import asyncio

from src.datasets.datasets_metadata import DatasetMetadataWithFields
from src.infrastructure.config import EmbedderModel, DEFAULT_EMBEDDER_MODEL
from src.infrastructure.logger import get_prefixed_logger
from src.utils.buffer_abc import AsyncBuffer
from src.vector_search.vector_db import VectorDB

logger = get_prefixed_logger(__name__, "VECTOR_BUFFER")


class VectorDBBuffer(AsyncBuffer[DatasetMetadataWithFields]):
    """Buffer for batching dataset indexing operations"""

    def __init__(
        self,
        vector_db: VectorDB,
        embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
        buffer_size: int = 150,
    ):
        """
        Initialize the buffer

        Args:
            vector_db: The AsyncVectorDB instance to use for indexing
            embedder_model: The embedder model to use for creating embeddings
            buffer_size: Maximum number of records to hold before auto-flushing
        """
        self.vector_db = vector_db
        self.embedder_model = embedder_model
        self.buffer_size = buffer_size
        self._buffer: list[DatasetMetadataWithFields] = []
        self._lock = asyncio.Lock()  # Async lock for thread safety
        self._total_indexed = 0

    # region Buffer logic

    async def add(self, dataset: DatasetMetadataWithFields) -> None:
        """
        Add a single dataset to the buffer

        Args:
            dataset: Dataset to add to the buffer
        """
        need_flush = False
        async with self._lock:
            self._buffer.append(dataset)
            logger.debug(f"Added dataset to buffer. Current size: {len(self._buffer)}")
            need_flush = len(self._buffer) >= self.buffer_size

        if need_flush:
            await self._flush_internal()

    async def flush(self) -> int:
        """
        Manually flush the buffer

        Returns:
            Number of datasets indexed
        """
        return await self._flush_internal()

    async def clear(self) -> None:
        """Clear the buffer without indexing"""
        async with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
        logger.info(f"Cleared {count} datasets from buffer without indexing")

    @property
    async def size(self) -> int:
        """Current number of items in the buffer"""
        async with self._lock:
            return len(self._buffer)

    @property
    def total_indexed(self) -> int:
        """Total number of datasets indexed through this buffer"""
        return self._total_indexed

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - always flush remaining data"""
        try:
            await self.flush()
        except Exception as e:
            logger.error(f"Error flushing buffer on exit: {e}")
            if exc_type is None:  # Only raise if there wasn't already an exception
                raise

    # endregion

    # region Data handle

    async def _flush_internal(self) -> int:
        """
        Internal flush method (must be called with lock held)

        Returns:
            Number of datasets indexed
        """
        async with self._lock:
            if not self._buffer:
                logger.debug("Buffer is empty, nothing to flush")
                return 0
            data_to_index = self._buffer[:]
            self._buffer.clear()
        data_count = len(data_to_index)

        try:
            logger.info(f"Flushing {data_count} datasets from buffer")

            async def index_and_cleanup():
                try:
                    await self.vector_db.index_datasets(
                        data_to_index,
                        embedder_model=self.embedder_model,
                        batch_size=self.buffer_size,
                    )
                    async with self._lock:
                        self._total_indexed += data_count
                    logger.info(
                        f"Successfully flushed {data_count} datasets to {self.embedder_model.value}. Total indexed: {self._total_indexed}"
                    )

                except Exception as _e:
                    logger.error(f"Background insert failed: {_e}")

            asyncio.create_task(index_and_cleanup())

            return data_count

        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")
            # Buffer is not cleared on error, so data is not lost
            raise

    # endregion
