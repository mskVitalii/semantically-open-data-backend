import asyncio
import uuid
from typing import List, Optional

from numpy import ndarray
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import QueryRequest, ScoredPoint
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
    Range,
)

from src.datasets.datasets_metadata import DatasetMetadataWithFields
from src.infrastructure.config import (
    USE_GRPC,
    QDRANT_GRPC_PORT,
    QDRANT_HOST,
    QDRANT_HTTP_PORT,
    EmbedderModel,
    DEFAULT_EMBEDDER_MODEL,
    get_collection_name,
    get_embedding_dim,
)
from src.infrastructure.logger import get_prefixed_logger
from src.vector_search.embedder import embed_batch

logger = get_prefixed_logger(__name__, "VECTOR_DB")


class VectorDB:
    """Vector DB system with gRPC support"""

    # region LIFE CYCLE

    def __init__(self):
        """Initialize with gRPC or HTTP client"""
        # Use environment variable if not explicitly set
        self.use_grpc = USE_GRPC
        self.qdrant: AsyncQdrantClient | None = None

    async def initialize(self):
        """Async initialization of client and resources"""
        self.qdrant = await get_qdrant(self.use_grpc)
        await self._wait_for_qdrant()
        await self.setup_collection()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        logger.warning("VectorDB __aexit__".upper())

    def __del__(self):
        logger.warning("VectorDB instance is being deleted".upper())

    async def _wait_for_qdrant(self, max_retries: int = 10, retry_delay: int = 2):
        """Wait for Qdrant to be ready"""
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

    async def setup_collection(self, embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL):
        """Create Qdrant collection if not exists for specific embedder model"""
        collection_name = get_collection_name(embedder_model)
        embedding_dim = get_embedding_dim(embedder_model)
        logger.info(
            f"setup_collection for {embedder_model.value} (dimension: {embedding_dim})"
        )
        collections_response = await self.qdrant.get_collections()
        collections = collections_response.collections

        collection_exists = any(c.name == collection_name for c in collections)

        if collection_exists:
            # Check if existing collection has the correct dimension
            try:
                collection_info = await self.qdrant.get_collection(collection_name)
                existing_dim = collection_info.config.params.vectors.size

                if existing_dim != embedding_dim:
                    logger.warning(
                        f"Collection {collection_name} exists with wrong dimension "
                        f"(expected {embedding_dim}, got {existing_dim}). Recreating..."
                    )
                    await self.qdrant.delete_collection(collection_name)
                    collection_exists = False
                else:
                    logger.info(
                        f"Collection {collection_name} already exists with correct dimension {embedding_dim}"
                    )
            except Exception as e:
                logger.error(f"Error checking collection dimension: {e}")
                # If we can't check, assume it needs recreation
                await self.qdrant.delete_collection(collection_name)
                collection_exists = False

        if not collection_exists:
            logger.info(
                f"Creating collection {collection_name} with dimension {embedding_dim}"
            )
            await self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim, distance=Distance.COSINE
                ),
            )

            # Create indexes for filtering
            for field in ["city", "state", "country", "organization", "embedder_model"]:
                await self.qdrant.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )

            # Create integer index for year filtering
            await self.qdrant.create_payload_index(
                collection_name=collection_name,
                field_name="year",
                field_schema=PayloadSchemaType.INTEGER,
            )
            logger.info(f"Collection {collection_name} created with indexes")

    async def index_datasets(
        self,
        datasets: List[DatasetMetadataWithFields],
        embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
        batch_size: int = 100,
    ):
        """Index multiple datasets with batching for better performance using specified embedder"""
        collection_name = get_collection_name(embedder_model)

        # Ensure collection exists
        await self.setup_collection(embedder_model)

        # Prepare texts for embedding
        try:
            texts = [ds.to_searchable_text() for ds in datasets]
        except Exception as e:
            logger.error(f"Error preparing texts for embedding: {e}", exc_info=e)
            return

        # Generate embeddings in batches using specified embedder
        logger.debug(f"Generating embeddings for {len(texts)} texts...")
        embeddings = await embed_batch(texts, embedder_model=embedder_model)

        # Prepare points for Qdrant
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=dataset.to_payload(),
            )
            for dataset, embedding in zip(datasets, embeddings)
        ]

        # Upload to Qdrant in batches for better performance
        total_points = len(points)
        for i in range(0, total_points, batch_size):
            batch = points[i : i + batch_size]
            await self.qdrant.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True,  # Ensure consistency
            )

        logger.debug(f"Indexed {len(datasets)} datasets to {collection_name}")

    async def search(
        self,
        query_embedding: ndarray,
        embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
        city_filter: Optional[str] = None,
        state_filter: Optional[str] = None,
        country_filter: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        limit: int = 5,
    ) -> list[ScoredPoint]:
        """Search for datasets using query_points method in specific embedder collection"""
        collection_name = get_collection_name(embedder_model)

        # Build filter conditions
        filter_conditions = []
        if city_filter:
            filter_conditions.append(
                FieldCondition(key="city", match=MatchValue(value=city_filter))
            )
        if state_filter:
            filter_conditions.append(
                FieldCondition(key="state", match=MatchValue(value=state_filter))
            )
        if country_filter:
            filter_conditions.append(
                FieldCondition(key="country", match=MatchValue(value=country_filter))
            )

        # Add year range filter if specified
        if year_from is not None or year_to is not None:
            range_config = {}
            if year_from is not None:
                range_config["gte"] = year_from
            if year_to is not None:
                range_config["lte"] = year_to

            filter_conditions.append(
                FieldCondition(key="year", range=Range(**range_config))
            )

        # Create filter if any conditions exist
        search_filter = None
        if filter_conditions:
            search_filter = Filter(must=filter_conditions)

        # Use query_points (works with both gRPC and HTTP)
        query_result = await self.qdrant.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            query_filter=search_filter,
            limit=limit,
            with_payload=True,
        )

        # Extract points from the result
        results = query_result.points
        return results

    async def batch_search(
        self,
        queries: List[str],
        embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
        city_filter: Optional[str] = None,
        limit: int = 5,
    ):
        """Batch search - especially efficient with gRPC using specific embedder"""
        collection_name = get_collection_name(embedder_model)
        logger.info(
            f"\nBatch searching for {len(queries)} queries in {collection_name}"
        )

        # Generate embeddings for all queries concurrently using specified embedder
        query_embeddings = await embed_batch(queries, embedder_model=embedder_model)

        # Build filter
        search_filter = None
        if city_filter:
            search_filter = Filter(
                must=[FieldCondition(key="city", match=MatchValue(value=city_filter))]
            )

        # Batch query - very efficient with gRPC
        batch_results = await self.qdrant.query_batch_points(
            collection_name=collection_name,
            requests=[
                QueryRequest(
                    query=emb.tolist(),
                    filter=search_filter,
                    limit=limit,
                    with_payload=True,
                )
                for emb in query_embeddings
            ],
        )

        # Process results
        all_results = []
        for i, (query, result) in enumerate(zip(queries, batch_results)):
            logger.info(f"\nQuery {i + 1}: '{query}'")
            logger.info(f"Found {len(result.points)} results")
            all_results.append(result.points)

        return all_results

    async def get_stats(self, embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL):
        """Get collection statistics for specific embedder"""
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
        """
        Remove a collection from Qdrant for specific embedder model.

        Args:
            embedder_model: Embedder model whose collection should be removed.
                           Defaults to DEFAULT_EMBEDDER_MODEL.

        Returns:
            bool: True if collection was removed successfully, False otherwise.
        """
        collection_name = get_collection_name(embedder_model)
        try:
            # Check if collection exists
            collections_response = await self.qdrant.get_collections()
            collections = collections_response.collections

            if not any(c.name == collection_name for c in collections):
                logger.warning(f"Collection '{collection_name}' does not exist")
                return False

            # Delete the collection
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
    """Helper function to create and initialize the async Qdrant client"""
    global vector_db
    if vector_db is None:
        vector_db = VectorDB()
        await vector_db.initialize()
    return vector_db


qdrant: AsyncQdrantClient | None = None


async def init_qdrant(use_grpc: bool = True) -> AsyncQdrantClient:
    # Initialize Qdrant client
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
    """Helper function to create and initialize the async Qdrant client"""
    global qdrant
    if qdrant is None:
        qdrant = await init_qdrant(use_grpc)
    return qdrant


# endregion
