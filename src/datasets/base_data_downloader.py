import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Set, Dict, Any, Optional
import asyncio
import aiohttp
from aiohttp import ClientTimeout, TCPConnector
import logging

from src.domain.repositories.dataset_repository import get_dataset_repository
from src.domain.services.dataset_buffer import DatasetDBBuffer
from src.infrastructure.mongo_db import get_mongo_database
from src.vector_search.vector_db import get_vector_db
from src.vector_search.vector_db_buffer import VectorDBBuffer


class BaseDataDownloader(ABC):
    """Abstract base class for async data downloaders"""

    # region INIT
    def __init__(
        self,
        output_dir: str = "data",
        max_workers: int = 8,
        delay: float = 0.05,
        is_file_system: bool = True,
        is_embeddings: bool = False,
        is_store: bool = False,
        connection_limit: int = 100,
        connection_limit_per_host: int = 30,
        batch_size: int = 50,
        max_retries: int = 1,
    ):
        """
        Initialize base downloader

        Args:
            output_dir: Directory to save data
            max_workers: Number of parallel workers
            delay: Delay between requests in seconds
            is_embeddings: Whether to generate embeddings
            is_store: Whether to save datasets to DB or not
            connection_limit: Total connection pool size
            connection_limit_per_host: Per-host connection limit
            batch_size: Size of dataset batches to process
            max_retries: Maximum retry attempts for failed requests
        """
        self.is_file_system = is_file_system
        self.output_dir = Path(output_dir)
        if is_file_system:
            self.output_dir.mkdir(exist_ok=True)

        self.max_workers = max_workers
        self.delay = delay
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.session: aiohttp.ClientSession
        self.is_embeddings = is_embeddings
        self.is_store = is_store

        # Connection configuration
        self.connection_limit = connection_limit
        self.connection_limit_per_host = connection_limit_per_host

        self.stats = {
            "datasets_found": 0,
            "datasets_processed": 0,
            "datasets_not_suitable": 0,
            "files_downloaded": 0,
            "geo_packages": 0,
            "errors": 0,
            "start_time": datetime.now(),
            "cache_hits": 0,
            "retries": 0,
            "failed_datasets": set(),
        }
        self.stats_lock = asyncio.Lock()

        # Cache for metadata to avoid redundant API calls
        self.cache: Dict[str, Any] = {}
        self.cache_lock = asyncio.Lock()

        # Track failed URLs for retry optimization
        self.failed_urls: Set[str] = set()
        self.failed_urls_lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)

        # Initialize buffer attributes (set in __aenter__ if needed)
        self.vector_db_buffer = None
        self.dataset_db_buffer = None

        # Cache unsuitable datasets to speed up processing
        self.unsuitable_datasets: Set[str] = set()
        self.unsuitable_datasets_lock = asyncio.Lock()
        self.unsuitable_datasets_file = self.output_dir / "unsuitable_datasets.json"

    async def __aenter__(self):
        """Async context manager entry with optimized session"""
        # Create connector with connection pooling
        connector = TCPConnector(
            limit=self.connection_limit,
            limit_per_host=self.connection_limit_per_host,
            ttl_dns_cache=300,  # DNS cache for 5 minutes
            enable_cleanup_closed=True,
            force_close=True,
        )

        # Optimized timeout settings
        timeout = ClientTimeout(
            total=120,  # Total timeout
            connect=30,  # Connection timeout
            sock_read=60,  # Socket read timeout
        )

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "Data Downloader (Python/aiohttp)",
                "Accept-Encoding": "gzip, deflate",  # Enable compression
            },
        )

        # Call child class initialization if needed
        """Initialize Dresden-specific resources"""
        if self.is_embeddings:
            vector_db = await get_vector_db(use_grpc=True)
            self.vector_db_buffer = VectorDBBuffer(vector_db)

        if self.is_store:
            database = await get_mongo_database()
            dataset_db = await get_dataset_repository(database=database)
            self.dataset_db_buffer = DatasetDBBuffer(dataset_db)

        await self.load_unsuitable_datasets()

        await self._initialize_resources()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Flush embeddings buffer if it exists
        if hasattr(self, "vector_db_buffer") and self.vector_db_buffer:
            try:
                await self.vector_db_buffer.flush()
            except Exception as e:
                self.logger.error(f"Error flushing VECTOR buffer: {e}")

        if hasattr(self, "dataset_db_buffer") and self.dataset_db_buffer:
            try:
                await self.dataset_db_buffer.flush()
            except Exception as e:
                self.logger.error(f"Error flushing MONGO buffer: {e}")

        # Close session
        if self.session:
            await self.session.close()

        # Clean up child class resources first
        await self._cleanup_resources()

    async def _initialize_resources(self):
        """
        Hook for child classes to initialize their specific resources.
        Called during __aenter__.
        """
        pass

    async def _cleanup_resources(self):
        """
        Hook for child classes to clean up their specific resources.
        Called during __aexit__.
        """
        pass

    # endregion

    # region LOGIC STEPS

    @abstractmethod
    async def process_all_datasets(self):
        """
        Abstract method to process all datasets.
        Must be implemented by child classes.
        """
        pass

    # endregion

    # region CACHE
    async def add_to_cache(self, key: str, value: Any):
        """
        Thread-safe cache addition

        Args:
            key: Cache key
            value: Value to cache
        """
        async with self.cache_lock:
            self.cache[key] = value

    async def get_from_cache(self, key: str) -> Optional[Any]:
        """
        Thread-safe cache retrieval

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        result = None
        async with self.cache_lock:
            if key in self.cache:
                result = self.cache.get(key)

        if result is not None:
            await self.update_stats("cache_hits")

        return result

    # endregion

    # region STATS
    async def update_stats(self, key: str, value: Any = 1, operation: str = "add"):
        """
        Thread-safe statistics update

        Args:
            key: Statistics key to update
            value: Value to add/set
            operation: 'add' to increment, 'set' to replace
        """
        async with self.stats_lock:
            if operation == "add":
                if key in self.stats:
                    if isinstance(self.stats[key], (int, float)):
                        self.stats[key] += value
                    elif isinstance(self.stats[key], set):
                        self.stats[key].add(value)
            elif operation == "set":
                self.stats[key] = value

    async def print_progress(self, current: int = None, total: int = None):
        """Unified progress reporting"""
        async with self.stats_lock:
            current = current or self.stats["datasets_processed"]
            total = total or self.stats["datasets_found"]

            if total > 0:
                percentage = (current / total) * 100
                elapsed = (datetime.now() - self.stats["start_time"]).total_seconds()
                rate = current / elapsed if elapsed > 0 else 0
                eta = (total - current) / rate if rate > 0 else 0

                metrics = [
                    f"Progress: {current}/{total} ({percentage:.1f}%)",
                    f"Files: {self.stats['files_downloaded']}",
                    f"Errors: {self.stats['errors']}",
                ]

                if "layers_downloaded" in self.stats:
                    metrics.append(f"Layers: {self.stats['layers_downloaded']}")
                if "playwright_downloads" in self.stats:
                    metrics.append(f"Playwright: {self.stats['playwright_downloads']}")

                metrics.extend([f"Rate: {rate:.1f} items/s", f"ETA: {eta:.0f}s"])

                self.logger.info("\t".join(metrics))

    async def progress_reporter(self):
        while True:
            await asyncio.sleep(5)  # More frequent updates
            async with self.stats_lock:
                if self.stats["datasets_processed"] >= self.stats["datasets_found"]:
                    break
            await self.print_progress()

    async def print_final_report(self):
        """Print comprehensive final report with statistics"""
        async with self.stats_lock:
            # Calculate final metrics
            elapsed = (datetime.now() - self.stats["start_time"]).total_seconds()
            processed = self.stats["datasets_processed"]
            total = self.stats["datasets_found"]
            files = self.stats["files_downloaded"]
            errors = self.stats["errors"]
            not_suitable = self.stats["datasets_not_suitable"]
            geo_packages = self.stats["geo_packages"]
            cache_hits = self.stats["cache_hits"]
            retries = self.stats["retries"]
            failed_count = len(self.stats["failed_datasets"])

            # Calculate rates and percentages
            success_rate = (
                ((processed - failed_count) / processed * 100) if processed > 0 else 0
            )
            cache_hit_rate = (
                (cache_hits / (cache_hits + files) * 100)
                if (cache_hits + files) > 0
                else 0
            )
            avg_rate = processed / elapsed if elapsed > 0 else 0

            # Format elapsed time
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            self.logger.debug("=" * 60)
            self.logger.debug("FINAL REPORT")
            self.logger.debug("=" * 60)
            self.logger.debug(f"Total runtime: {time_str}")
            self.logger.debug(f"Datasets found: {total}")
            self.logger.debug(
                f"Datasets processed: {processed}/{total} ({processed / total * 100:.1f}%)"
                if total > 0
                else "Datasets processed: 0"
            )
            self.logger.debug(f"Geo packages: {geo_packages}")
            self.logger.debug(f"Files downloaded: {files}")
            self.logger.debug("-" * 60)
            self.logger.debug("PERFORMANCE METRICS:")
            self.logger.debug(f"Average processing rate: {avg_rate:.2f} datasets/s")
            self.logger.debug(
                f"Cache hit rate: {cache_hit_rate:.1f}% ({cache_hits} hits)"
            )
            self.logger.debug(f"Retry attempts: {retries}")

            # Error analysis
            self.logger.debug("-" * 60)
            self.logger.debug("ERROR ANALYSIS:")
            self.logger.debug(f"Total errors: {errors}")
            self.logger.debug(f"Failed datasets: {failed_count}")
            self.logger.debug(f"Not suitable: {not_suitable}")
            self.logger.debug(f"Success rate: {success_rate:.1f}%")

            if 0 < failed_count:
                self.logger.debug("Failed dataset IDs:")
                for dataset_id in self.stats["failed_datasets"]:
                    self.logger.debug(f"  - {dataset_id}")

            # Get additional metrics from child classes
            additional_metrics = await self.get_additional_metrics()
            if additional_metrics:
                self.logger.debug("-" * 60)
                self.logger.debug("ADDITIONAL METRICS:")
                for key in additional_metrics:
                    value = self.stats[key]
                    self.logger.debug(f"{key}: {value}")

            self.logger.debug("=" * 60)

    async def get_additional_metrics(self) -> dict:
        """
        Override in child classes to provide additional metrics for the final report.

        Returns:
            Dictionary with metric names as keys and formatted values as strings
        """
        pass

    async def get_stat_value(self, key: str):
        async with self.stats_lock:
            return self.stats[key]

    # endregion

    # region FAILED URLS
    async def mark_url_failed(self, url: str):
        """
        Mark URL as failed for retry tracking

        Args:
            url: Failed URL
        """
        async with self.failed_urls_lock:
            self.failed_urls.add(url)

    async def is_url_failed(self, url: str) -> bool:
        """
        Check if URL has previously failed

        Args:
            url: URL to check

        Returns:
            True if URL has failed before
        """
        async with self.failed_urls_lock:
            return url in self.failed_urls

    # endregion

    # region UNSUITABLE DATASETS CACHE

    async def load_unsuitable_datasets(self):
        """Load unsuitable datasets from cache file"""
        if not self.unsuitable_datasets_file.exists():
            self.logger.info("No unsuitable datasets cache found, starting fresh")
            return

        try:
            with open(self.unsuitable_datasets_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                async with self.unsuitable_datasets_lock:
                    self.unsuitable_datasets = set(data)
        except Exception as e:
            self.logger.error(f"Error loading unsuitable datasets cache: {e}")
            self.unsuitable_datasets = set()

    async def save_unsuitable_datasets(self):
        """Save unsuitable datasets to cache file"""
        if not self.is_file_system:
            return

        try:
            async with self.unsuitable_datasets_lock:
                data = sorted(list(self.unsuitable_datasets))

            with open(self.unsuitable_datasets_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.error(f"Error saving unsuitable datasets cache: {e}")

    async def is_dataset_unsuitable(self, package_name: str) -> bool:
        """Check if dataset is in unsuitable cache"""
        async with self.unsuitable_datasets_lock:
            return package_name in self.unsuitable_datasets

    async def mark_dataset_unsuitable(self, package_name: str):
        """Mark dataset as unsuitable and save to cache"""
        async with self.unsuitable_datasets_lock:
            self.unsuitable_datasets.add(package_name)

        if self.is_file_system:
            await self.save_unsuitable_datasets()

    # endregion
