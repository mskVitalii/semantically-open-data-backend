# src/datasets_api/service.py
from fastapi import Depends
from numpy import ndarray

from src.datasets.bootstrap import bootstrap_data
from src.datasets.datasets_metadata import DatasetMetadataWithFields, make_field
from src.domain.repositories.dataset_repository import (
    DatasetRepository,
    get_dataset_repository,
)
from src.datasets_api.datasets_dto import (
    DatasetSearchRequest,
    DatasetSearchResponse,
    DatasetResponse,
)
from src.infrastructure.config import (
    EmbedderModel,
    DEFAULT_EMBEDDER_MODEL,
    SearchMode,
)
from src.infrastructure.logger import get_prefixed_logger
from src.infrastructure.paths import PROJECT_ROOT
from src.utils.datasets_utils import safe_delete
from src.vector_search.embedder import embed, embed_single, SparseVector, HybridEmbedding
from src.vector_search.reranker import rerank
from src.vector_search.vector_db import VectorDB, get_vector_db

logger = get_prefixed_logger(__name__, "DATASET_SERVICE")


class DatasetService:
    """Service for working with datasets"""

    def __init__(self, vector_db: VectorDB, repository: DatasetRepository):
        self.vector_db = vector_db
        self.repository = repository

    async def search_datasets(
        self,
        request: DatasetSearchRequest,
        embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
        search_mode: SearchMode = SearchMode.DENSE,
    ) -> DatasetSearchResponse:
        """Search for datasets using specified embedder model and search mode"""

        filter_kwargs = dict(
            embedder_model=embedder_model,
            city_filter=request.city,
            state_filter=request.state,
            country_filter=request.country,
            year_from=request.year_from,
            year_to=request.year_to,
            limit=request.limit,
        )

        # embed_single returns the right type for the requested mode
        query_vec = await embed_single(
            request.query, embedder_model=embedder_model, mode=search_mode,
        )

        if search_mode == SearchMode.SPARSE:
            datasets = await self.vector_db.search_sparse(query_vec, **filter_kwargs)
        elif search_mode == SearchMode.HYBRID:
            datasets = await self.vector_db.search_hybrid(query_vec, **filter_kwargs)
        else:
            datasets = await self.vector_db.search_dense(query_vec, **filter_kwargs)

        metadatas: list[DatasetResponse] = []
        for dataset in datasets:
            payload = dict(dataset.payload)
            # Remove year field as it's only used for filtering, not part of the model
            payload.pop("year", None)
            payload.pop(
                "fields", None
            )  # Remove fields as they're not needed in basic search

            # Convert embedder_model from string to EmbedderModel enum if needed
            if "embedder_model" in payload and isinstance(
                payload["embedder_model"], str
            ):
                payload["embedder_model"] = EmbedderModel(payload["embedder_model"])

            metadatas.append(
                DatasetResponse(
                    metadata=DatasetMetadataWithFields(**payload),
                    score=dataset.score,
                ),
            )

        return DatasetSearchResponse(
            datasets=metadatas,
            total=len(metadatas),
            limit=request.limit,
            offset=request.offset,
        )

    async def search_datasets_with_embeddings(
        self,
        embeddings: ndarray,
        embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
        city_filter: str = None,
        state_filter: str = None,
        country_filter: str = None,
        year_from: int = None,
        year_to: int = None,
    ) -> list[DatasetResponse]:
        """Search for datasets using pre-computed dense embeddings"""
        datasets = await self.vector_db.search_dense(
            embeddings,
            embedder_model=embedder_model,
            city_filter=city_filter,
            state_filter=state_filter,
            country_filter=country_filter,
            year_from=year_from,
            year_to=year_to,
        )

        results = []
        for dataset in datasets:
            payload = dict(dataset.payload)
            raw_fields = payload.pop("fields", {})
            # Remove year field as it's only used for filtering, not part of the model
            payload.pop("year", None)

            # Convert embedder_model from string to EmbedderModel enum if needed
            if "embedder_model" in payload and isinstance(
                payload["embedder_model"], str
            ):
                payload["embedder_model"] = EmbedderModel(payload["embedder_model"])

            metadata = DatasetMetadataWithFields(
                **payload,
                fields={k: make_field(v) for k, v in raw_fields.items()},
            )

            results.append(
                DatasetResponse(
                    metadata=metadata,
                    score=dataset.score,
                )
            )

        return results

    async def search_datasets_with_vector(
        self,
        vector,  # ndarray | SparseVector | HybridEmbedding
        search_mode: SearchMode = SearchMode.DENSE,
        embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
        city_filter: str = None,
        state_filter: str = None,
        country_filter: str = None,
        year_from: int = None,
        year_to: int = None,
        limit: int = 25,
    ) -> list[DatasetResponse]:
        """Search for datasets using a pre-computed vector (any mode)"""
        filter_kwargs = dict(
            embedder_model=embedder_model,
            city_filter=city_filter,
            state_filter=state_filter,
            country_filter=country_filter,
            year_from=year_from,
            year_to=year_to,
            limit=limit,
        )

        if search_mode == SearchMode.SPARSE:
            datasets = await self.vector_db.search_sparse(vector, **filter_kwargs)
        elif search_mode == SearchMode.HYBRID:
            datasets = await self.vector_db.search_hybrid(vector, **filter_kwargs)
        else:
            datasets = await self.vector_db.search_dense(vector, **filter_kwargs)

        results = []
        for dataset in datasets:
            payload = dict(dataset.payload)
            raw_fields = payload.pop("fields", {})
            payload.pop("year", None)

            if "embedder_model" in payload and isinstance(
                payload["embedder_model"], str
            ):
                payload["embedder_model"] = EmbedderModel(payload["embedder_model"])

            metadata = DatasetMetadataWithFields(
                **payload,
                fields={k: make_field(v) for k, v in raw_fields.items()},
            )
            results.append(
                DatasetResponse(metadata=metadata, score=dataset.score)
            )

        return results

    async def rerank_results(
        self,
        query: str,
        datasets: list[DatasetResponse],
        top_n: int | None = None,
    ) -> list[DatasetResponse]:
        """Rerank dataset results using the reranker service"""
        if not datasets:
            return datasets

        documents = [
            f"{ds.metadata.title}. {ds.metadata.description}"
            for ds in datasets
        ]

        ranked = await rerank(query=query, documents=documents, top_n=top_n)

        reranked = []
        for r in ranked:
            ds = datasets[r["index"]]
            reranked.append(
                DatasetResponse(
                    metadata=ds.metadata,
                    score=r["relevance_score"],
                )
            )
        return reranked

    async def bootstrap_datasets(
        self,
        clear_store: bool = True,
        clear_vector_db: bool = True,
        use_fs_cache: bool = True,
    ) -> bool:
        """Bootstrap datasets - clear and reload all data"""
        try:
            await self.clear_all_data(
                clear_store=clear_store,
                clear_vector_db=clear_vector_db,
                clear_fs=not use_fs_cache,
            )
            # Bootstrap data (this should populate both MongoDB and vector DB)
            await bootstrap_data(use_fs_cache)

            # Create indexes for better performance
            await self.repository.create_indexes()

            # Get statistics
            stats = await self.repository.get_statistics()
            logger.info(f"Bootstrap completed. Stats: {stats}")

            return True
        except Exception as e:
            logger.error(f"bootstrap_datasets error: {e}")
            return False

    async def index_existing_datasets(
        self,
        embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
    ) -> dict:
        """Index existing datasets from filesystem into vector DB (hybrid: dense + sparse)"""
        try:
            from src.utils.datasets_utils import load_metadata_from_file

            # Cities to scan
            cities = ["berlin", "chemnitz", "leipzig", "dresden"]
            datasets_to_index: list[DatasetMetadataWithFields] = []

            # Scan each city directory
            for city in cities:
                city_path = PROJECT_ROOT / "src" / "datasets" / city
                if not city_path.exists():
                    logger.warning(f"City directory not found: {city_path}")
                    continue

                logger.info(f"Scanning {city} datasets...")

                # Iterate through all dataset directories
                for dataset_dir in city_path.iterdir():
                    if not dataset_dir.is_dir():
                        continue

                    metadata_file = dataset_dir / "_metadata.json"
                    if not metadata_file.exists():
                        logger.debug(
                            f"Skipping {dataset_dir.name} - no _metadata.json found"
                        )
                        continue

                    # Load metadata from file
                    metadata = load_metadata_from_file(metadata_file)
                    if metadata is None:
                        logger.warning(f"Failed to load metadata from {metadata_file}")
                        continue

                    # Update embedder_model to the one we're indexing with
                    metadata.embedder_model = embedder_model

                    datasets_to_index.append(metadata)

            if not datasets_to_index:
                logger.warning("No datasets found in filesystem to index")
                return {
                    "ok": False,
                    "message": "No datasets found in filesystem",
                    "indexed": 0,
                }

            total_datasets = len(datasets_to_index)
            logger.info(
                f"Found {total_datasets} datasets in filesystem. Indexing with {embedder_model.value}..."
            )

            # Ensure collection exists for this embedder
            await self.vector_db.setup_collection(embedder_model)

            # Index datasets in batches of 10
            batch_size = 10
            total_batches = (total_datasets + batch_size - 1) // batch_size

            for i in range(0, total_datasets, batch_size):
                batch = datasets_to_index[i : i + batch_size]
                batch_num = i // batch_size + 1

                logger.info(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch)} datasets)..."
                )

                await self.vector_db.index_datasets(
                    batch, embedder_model=embedder_model,
                )

                logger.info(
                    f"✓ Batch {batch_num}/{total_batches} completed ({i + len(batch)}/{total_datasets} total)"
                )

            logger.info(
                f"Successfully indexed all {total_datasets} datasets with {embedder_model.value}"
            )

            return {
                "ok": True,
                "message": f"Indexed {total_datasets} datasets from filesystem",
                "indexed": total_datasets,
                "embedder_model": embedder_model.value,
            }

        except Exception as e:
            logger.error(f"index_existing_datasets error: {e}", exc_info=True)
            return {"ok": False, "message": str(e), "indexed": 0}

    async def clear_all_data(
        self,
        clear_store: bool = True,
        clear_vector_db: bool = True,
        clear_fs: bool = False,
    ) -> bool:
        """Clear all data from MongoDB and vector DB (all embedder collections)"""
        try:
            # Clear MongoDB
            if clear_store:
                await self.repository.delete_all()
                logger.warning("Deleted all MONGO collections")

            # Clear vector DB — one collection per embedder model
            if clear_vector_db:
                for embedder_model in EmbedderModel:
                    try:
                        await self.vector_db.remove_collection(embedder_model)
                    except Exception as e:
                        logger.error(
                            f"Error clearing collection for {embedder_model.value}: {e}"
                        )
                    await self.vector_db.setup_collection(embedder_model)
                    logger.warning(
                        f"Cleared vector DB collection for {embedder_model.value}"
                    )

            if clear_fs:
                safe_delete(PROJECT_ROOT / "src" / "datasets" / "dresden", logger)
                safe_delete(PROJECT_ROOT / "src" / "datasets" / "chemnitz", logger)
                safe_delete(PROJECT_ROOT / "src" / "datasets" / "berlin", logger)
                safe_delete(PROJECT_ROOT / "src" / "datasets" / "leipzig", logger)

            return True
        except Exception as e:
            logger.error(f"clear_all_data error: {e}")
            return False


# Dependency injection
async def get_dataset_service(
    vector_db: VectorDB = Depends(get_vector_db),
    repository: DatasetRepository = Depends(get_dataset_repository),
) -> DatasetService:
    """Get DatasetService instance with dependencies"""
    return DatasetService(vector_db, repository)
