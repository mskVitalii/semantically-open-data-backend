import asyncio
import uuid
from typing import List, Optional

from numpy import ndarray
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    QueryRequest,
    ScoredPoint,
    SearchParams,
    FusionQuery,
    Fusion,
    Prefetch,
)
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
    Range,
    SparseVector as QdrantSparseVector,
    HnswConfigDiff,
)

from src.datasets.datasets_metadata import DatasetMetadataWithFields
from src.infrastructure.config import (
    USE_GRPC,
    QDRANT_GRPC_PORT,
    QDRANT_HOST,
    QDRANT_HTTP_PORT,
    EmbedderModel,
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_SPARSE_DIM,
    DEFAULT_SPARSE_MODE,
    get_collection_name,
    get_embedding_dim,
)
from src.infrastructure.logger import get_prefixed_logger
from src.vector_search.embedder import (
    embed_batch_hybrid,
    SparseVector,
    HybridEmbedding,
)

logger = get_prefixed_logger(__name__, "VECTOR_DB")

# Named vector keys (every collection is hybrid)
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "lexical"

# Payload index fields
_KEYWORD_FIELDS = ["city", "state", "country", "organization", "embedder_model"]


class VectorDB:
    """Vector DB system with gRPC support and hybrid search.

    Every collection stores both dense and sparse (lexical) vectors.
    Index once, search with any strategy: dense / sparse / hybrid (RRF).
    """

    # region LIFE CYCLE

    def __init__(self):
        self.use_grpc = USE_GRPC
        self.qdrant: AsyncQdrantClient | None = None

    async def initialize(self):
        self.qdrant = await get_qdrant(self.use_grpc)
        await self._wait_for_qdrant()
        await self.setup_collection()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.warning("VectorDB __aexit__".upper())

    def __del__(self):
        logger.warning("VectorDB instance is being deleted".upper())

    async def _wait_for_qdrant(self, max_retries: int = 10, retry_delay: int = 2):
        for i in range(max_retries):
            try:
                await self.qdrant.get_collections()
                logger.info(
                    f"Qdrant is ready! (Using {'gRPC' if self.use_grpc else 'HTTP'})"
                )
                return
            except Exception as e:
                if i < max_retries - 1:
                    logger.info(f"Waiting for Qdrant... ({i + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                else:
                    raise RuntimeError(f"Qdrant failed to become ready: {e}")

    # endregion

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    async def _collection_exists(self, name: str) -> bool:
        collections = (await self.qdrant.get_collections()).collections
        return any(c.name == name for c in collections)

    async def _create_payload_indexes(self, collection_name: str):
        for field in _KEYWORD_FIELDS:
            await self.qdrant.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        await self.qdrant.create_payload_index(
            collection_name=collection_name,
            field_name="year",
            field_schema=PayloadSchemaType.INTEGER,
        )

    # ------------------------------------------------------------------
    # Collection setup (always hybrid: named dense + sparse)
    # ------------------------------------------------------------------

    async def setup_collection(
        self, embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL
    ):
        """Create hybrid Qdrant collection with named dense + sparse vectors."""
        collection_name = get_collection_name(embedder_model)
        embedding_dim = get_embedding_dim(embedder_model)
        logger.info(
            f"setup_collection for {embedder_model.value} (dense dim: {embedding_dim})"
        )

        if await self._collection_exists(collection_name):
            # Check if existing collection has the correct hybrid schema
            try:
                info = await self.qdrant.get_collection(collection_name)
                vectors_cfg = info.config.params.vectors
                # Old collections have a single unnamed VectorParams (not a dict)
                # New hybrid collections have a dict with "dense" key
                is_hybrid = isinstance(vectors_cfg, dict) and DENSE_VECTOR_NAME in vectors_cfg
                if is_hybrid:
                    logger.info(f"Collection {collection_name} already exists (hybrid)")
                    return
                else:
                    logger.warning(
                        f"Collection {collection_name} has legacy schema (unnamed vector). "
                        f"Recreating as hybrid..."
                    )
                    await self.qdrant.delete_collection(collection_name)
            except Exception as e:
                logger.error(f"Error checking collection schema: {e}")
                await self.qdrant.delete_collection(collection_name)

        logger.info(f"Creating collection {collection_name}")
        await self.qdrant.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=embedding_dim, distance=Distance.COSINE
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
            hnsw_config=HnswConfigDiff(m=64, ef_construct=256),
        )
        await self._create_payload_indexes(collection_name)
        logger.info(f"Collection {collection_name} created with indexes")

    # ------------------------------------------------------------------
    # Indexing (always hybrid — stores both dense + sparse)
    # ------------------------------------------------------------------

    async def index_datasets(
        self,
        datasets: List[DatasetMetadataWithFields],
        embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
        batch_size: int = 100,
        sparse_dim: int = DEFAULT_SPARSE_DIM,
        sparse_mode: str = DEFAULT_SPARSE_MODE,
    ):
        """Index datasets — always stores both dense and sparse vectors."""
        collection_name = get_collection_name(embedder_model)
        await self.setup_collection(embedder_model)

        try:
            texts = [ds.to_searchable_text() for ds in datasets]
        except Exception as e:
            logger.error(f"Error preparing texts for embedding: {e}", exc_info=e)
            return

        logger.debug(f"Generating hybrid embeddings for {len(texts)} texts...")
        hybrid_vecs = await embed_batch_hybrid(
            texts, embedder_model, sparse_dim, sparse_mode,
        )

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_VECTOR_NAME: hv.dense.tolist(),
                    SPARSE_VECTOR_NAME: QdrantSparseVector(
                        indices=hv.sparse.indices, values=hv.sparse.values,
                    ),
                },
                payload=ds.to_payload(),
            )
            for ds, hv in zip(datasets, hybrid_vecs)
        ]

        total_points = len(points)
        for i in range(0, total_points, batch_size):
            batch = points[i : i + batch_size]
            await self.qdrant.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True,
            )

        logger.debug(f"Indexed {len(datasets)} datasets to {collection_name}")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _build_filter(
        self,
        city_filter: Optional[str] = None,
        state_filter: Optional[str] = None,
        country_filter: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
    ) -> Optional[Filter]:
        conditions = []
        if city_filter:
            conditions.append(
                FieldCondition(key="city", match=MatchValue(value=city_filter))
            )
        if state_filter:
            conditions.append(
                FieldCondition(key="state", match=MatchValue(value=state_filter))
            )
        if country_filter:
            conditions.append(
                FieldCondition(key="country", match=MatchValue(value=country_filter))
            )
        if year_from is not None or year_to is not None:
            range_config = {}
            if year_from is not None:
                range_config["gte"] = year_from
            if year_to is not None:
                range_config["lte"] = year_to
            conditions.append(
                FieldCondition(key="year", range=Range(**range_config))
            )
        return Filter(must=conditions) if conditions else None

    # --- Dense search (named "dense" vector) ---

    async def search_dense(
        self,
        query_embedding: ndarray,
        embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
        city_filter: Optional[str] = None,
        state_filter: Optional[str] = None,
        country_filter: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        limit: int = 25,
    ) -> list[ScoredPoint]:
        """Dense-only search on the named 'dense' vector."""
        collection_name = get_collection_name(embedder_model)
        search_filter = self._build_filter(
            city_filter, state_filter, country_filter, year_from, year_to,
        )

        query_result = await self.qdrant.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            using=DENSE_VECTOR_NAME,
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=256),
        )
        return query_result.points

    # backward-compatible alias
    search = search_dense

    # --- Sparse search (named "lexical" vector) ---

    async def search_sparse(
        self,
        query_sparse: SparseVector,
        embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
        city_filter: Optional[str] = None,
        state_filter: Optional[str] = None,
        country_filter: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        limit: int = 25,
    ) -> list[ScoredPoint]:
        """Sparse-only search on the named 'lexical' vector."""
        collection_name = get_collection_name(embedder_model)
        search_filter = self._build_filter(
            city_filter, state_filter, country_filter, year_from, year_to,
        )

        qdrant_sparse = QdrantSparseVector(
            indices=query_sparse.indices, values=query_sparse.values,
        )

        query_result = await self.qdrant.query_points(
            collection_name=collection_name,
            query=qdrant_sparse,
            using=SPARSE_VECTOR_NAME,
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
        )
        return query_result.points

    # --- Hybrid search (RRF fusion) ---

    async def search_hybrid(
        self,
        query_hybrid: HybridEmbedding,
        embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
        city_filter: Optional[str] = None,
        state_filter: Optional[str] = None,
        country_filter: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        limit: int = 25,
        prefetch_limit: int = 100,
    ) -> list[ScoredPoint]:
        """Hybrid search: dense + sparse with RRF fusion."""
        collection_name = get_collection_name(embedder_model)
        search_filter = self._build_filter(
            city_filter, state_filter, country_filter, year_from, year_to,
        )

        qdrant_sparse = QdrantSparseVector(
            indices=query_hybrid.sparse.indices,
            values=query_hybrid.sparse.values,
        )

        query_result = await self.qdrant.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    query=query_hybrid.dense.tolist(),
                    using=DENSE_VECTOR_NAME,
                    filter=search_filter,
                    limit=prefetch_limit,
                ),
                Prefetch(
                    query=qdrant_sparse,
                    using=SPARSE_VECTOR_NAME,
                    filter=search_filter,
                    limit=prefetch_limit,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return query_result.points

    # ------------------------------------------------------------------
    # Batch search (dense)
    # ------------------------------------------------------------------

    async def batch_search(
        self,
        queries: List[str],
        embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
        city_filter: Optional[str] = None,
        limit: int = 5,
    ):
        """Batch dense search — efficient with gRPC."""
        from src.vector_search.embedder import embed_batch

        collection_name = get_collection_name(embedder_model)
        logger.info(
            f"\nBatch searching for {len(queries)} queries in {collection_name}"
        )

        query_embeddings = await embed_batch(queries, embedder_model=embedder_model)

        search_filter = None
        if city_filter:
            search_filter = Filter(
                must=[FieldCondition(key="city", match=MatchValue(value=city_filter))]
            )

        batch_results = await self.qdrant.query_batch_points(
            collection_name=collection_name,
            requests=[
                QueryRequest(
                    query=emb.tolist(),
                    using=DENSE_VECTOR_NAME,
                    filter=search_filter,
                    limit=limit,
                    with_payload=True,
                )
                for emb in query_embeddings
            ],
        )

        all_results = []
        for i, (query, result) in enumerate(zip(queries, batch_results)):
            logger.info(f"\nQuery {i + 1}: '{query}'")
            logger.info(f"Found {len(result.points)} results")
            all_results.append(result.points)

        return all_results

    # ------------------------------------------------------------------
    # Stats & management
    # ------------------------------------------------------------------

    async def get_stats(self, embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL):
        collection_name = get_collection_name(embedder_model)
        info = await self.qdrant.get_collection(collection_name)
        logger.info(f"\nCollection stats for {collection_name}:")
        logger.info(f"  Vectors count: {info.vectors_count}")
        logger.info(f"  Points count: {info.points_count}")
        logger.info(f"  Indexed vectors: {info.indexed_vectors_count}")
        logger.info(f"  Protocol: {'gRPC' if self.use_grpc else 'HTTP'}")
        return info

    async def remove_collection(
        self, embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL
    ) -> bool:
        collection_name = get_collection_name(embedder_model)
        try:
            if not await self._collection_exists(collection_name):
                logger.warning(f"Collection '{collection_name}' does not exist")
                return False

            logger.info(f"Removing collection '{collection_name}'...")
            await self.qdrant.delete_collection(collection_name=collection_name)
            logger.info(f"✅ Collection '{collection_name}' removed successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to remove collection '{collection_name}': {e}")
            raise


# region DI

vector_db: VectorDB | None = None


async def get_vector_db() -> VectorDB:
    global vector_db
    if vector_db is None:
        vector_db = VectorDB()
        await vector_db.initialize()
    return vector_db


qdrant: AsyncQdrantClient | None = None


async def init_qdrant(use_grpc: bool = True) -> AsyncQdrantClient:
    if use_grpc:
        logger.info(
            f"Connecting to Qdrant via gRPC at {QDRANT_HOST}:{QDRANT_GRPC_PORT}"
        )
        return AsyncQdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_GRPC_PORT,
            grpc_port=QDRANT_GRPC_PORT,
            prefer_grpc=True,
            timeout=30,
        )
    else:
        logger.info(
            f"Connecting to Qdrant via HTTP at {QDRANT_HOST}:{QDRANT_HTTP_PORT}"
        )
        return AsyncQdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_HTTP_PORT,
            prefer_grpc=False,
        )


async def get_qdrant(use_grpc: bool = True) -> AsyncQdrantClient:
    global qdrant
    if qdrant is None:
        qdrant = await init_qdrant(use_grpc)
    return qdrant


# endregion
