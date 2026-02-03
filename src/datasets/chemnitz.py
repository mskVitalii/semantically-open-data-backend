import asyncio
import io
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
from src.datasets.base_data_downloader import BaseDataDownloader
from src.datasets.datasets_metadata import DatasetMetadataWithFields
from src.utils.datasets_utils import (
    sanitize_title,
)
from src.infrastructure.logger import get_prefixed_logger
from src.utils.file import save_file_with_task


class Chemnitz(BaseDataDownloader):
    """Class for downloading Chemnitz open data"""

    # region INIT

    def __init__(
        self,
        csv_file_path: str,
        output_dir: str = "chemnitz",
        max_workers: int = 128,
        delay: float = 0.05,
        use_file_system: bool = True,
        use_embeddings: bool = False,
        use_store: bool = False,
        use_parallel: bool = True,
        use_playwright: bool = True,
        connection_limit: int = 100,
        connection_limit_per_host: int = 30,
        batch_size: int = 50,
        max_retries: int = 1,
    ):
        """
        Initialize optimized downloader

        Args:
            csv_file_path: Path to CSV file with dataset metadata
            output_dir: Directory to save data
            max_workers: Number of parallel workers
            delay: Delay between requests in seconds
            use_file_system: Whether to save datasets to filesystem
            use_embeddings: Whether to generate embeddings
            use_store: Whether to save datasets to DB or not
            connection_limit: Total connection pool size
            connection_limit_per_host: Per-host connection limit
            batch_size: Size of dataset batches to process
            max_retries: Maximum retry attempts for failed requests
            use_parallel: Whether to use parallel processing
            use_playwright: Whether to use Playwright for downloads
        """
        super().__init__(
            output_dir=output_dir,
            max_workers=max_workers,
            delay=delay,
            use_file_system=use_file_system,
            use_embeddings=use_embeddings,
            use_store=use_store,
            use_parallel=use_parallel,
            connection_limit=connection_limit,
            connection_limit_per_host=connection_limit_per_host,
            batch_size=batch_size,
            max_retries=max_retries,
        )
        self.csv_file_path = csv_file_path
        self.stats["layers_downloaded"] = 0
        self.logger = get_prefixed_logger(__name__, "CHEMNITZ")
        self.use_playwright = use_playwright

    # endregion

    # region STATS
    async def get_additional_metrics(self) -> list[str]:
        return ["layers_downloaded"]

    # endregion

    # region LOGIC STEPS
    # 6.
    async def download_layer_data_by_api(
        self,
        service_url: str,
        layer_id: int,
        layer_name: str,
    ) -> tuple[bool, dict | None]:
        """Download data for a single layer via query endpoint (JSON format).

        Returns:
            Tuple of (success, JSON_data)
        """
        query_url = f"{service_url}/{layer_id}/query"
        params = {
            "where": "1=1",
            "outFields": "*",
            "f": "json",
            "returnGeometry": "true",
        }

        for attempt in range(self.max_retries):
            try:
                async with self.session.get(query_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if "error" not in data:
                            await self.update_stats("layers_downloaded")
                            return True, data

                    return False, None

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await self.update_stats("retries")
                    await asyncio.sleep(2**attempt)
                else:
                    self.logger.error(f"Error downloading layer {layer_name}: {e}")
                    return False, None

        return False, None

    # 5.
    async def get_service_info_by_api(self, service_url: str) -> Optional[dict]:
        """Get service info with caching and retry logic"""
        # cached_service_info = await self.get_from_cache(service_url)
        # if cached_service_info is not None:
        #     return cached_service_info

        if await self.is_url_failed(service_url):
            return None

        for attempt in range(self.max_retries):
            try:
                async with self.session.get(f"{service_url}?f=json") as response:
                    response.raise_for_status()
                    result = await response.json()

                return result

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await self.update_stats("retries")
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    self.logger.error(
                        f"Error getting service info from {service_url}: {e}"
                    )
                    await self.mark_url_failed(service_url)
                    return None
        return None

    # 4a.
    async def download_map_service_metadata(
        self,
        csv_metadata: Dict[str, str],
    ) -> bool:
        """Download metadata for a Map Service (raster/WMS - no downloadable data)"""
        service_url = csv_metadata["url"]
        title = csv_metadata["title"]
        safe_title = sanitize_title(title)

        try:
            # Check if metadata file already exists
            if self.use_file_system:
                dataset_dir = self.output_dir / safe_title
                metadata_file = dataset_dir / "_metadata.json"
                if metadata_file.exists():
                    self.logger.debug(f"Metadata already exists: {title}")
                    await self.update_stats("datasets_processed")
                    await self.update_stats("files_downloaded")
                    return True

            # Get service info
            service_info = await self.get_service_info_by_api(service_url)
            if not service_info:
                await self.update_stats("datasets_processed")
                await self.update_stats("failed_datasets", title)
                return True

            # Parse tags and categories from CSV
            tags_str = csv_metadata.get("tags", "")
            tags = [t.strip() for t in tags_str.split(",")] if tags_str else None

            categories_str = csv_metadata.get("categories", "")
            groups = (
                [c.strip() for c in categories_str.split(",")]
                if categories_str
                else None
            )

            # Prepare metadata
            package_meta = DatasetMetadataWithFields(
                id=service_info.get("serviceItemId") or csv_metadata.get("id"),
                title=title,
                description=csv_metadata.get("description"),
                organization=csv_metadata.get("source"),
                metadata_created=csv_metadata.get("created"),
                metadata_modified=csv_metadata.get("modified"),
                city="Chemnitz",
                state="Saxony",
                country="Germany",
                tags=tags,
                groups=groups,
                url=csv_metadata.get("url"),
                author=csv_metadata.get("owner"),
                is_geo=True,
            )

            await self.update_stats("files_downloaded")

            if self.use_embeddings and self.vector_db_buffer:
                await self.vector_db_buffer.add(package_meta)

            if self.use_file_system:
                dataset_dir = self.output_dir / safe_title
                dataset_dir.mkdir(exist_ok=True)
                save_file_with_task(
                    dataset_dir / "_metadata.json", package_meta.to_json()
                )
                save_file_with_task(
                    dataset_dir / "_dataset_info.json",
                    json.dumps(service_info, ensure_ascii=False, indent=2),
                )

            await self.update_stats("datasets_processed")
            return True

        except Exception as e:
            self.logger.error(
                f"\tError processing Map Service {title}: {e}", exc_info=True
            )
            await self.update_stats("datasets_processed")
            await self.update_stats("failed_datasets", title)
            await self.update_stats("errors")
            return False

    # 4b.
    async def download_feature_service_data(
        self,
        csv_metadata: Dict[str, str],
    ) -> bool:
        """Download all data from a feature service with optimized concurrency.

        Each layer is saved under its own name in all available formats:
        - {layer_name}.json
        """
        service_url = csv_metadata["url"]
        title = csv_metadata["title"]
        safe_title = sanitize_title(title)

        try:
            # Get service info
            service_info = await self.get_service_info_by_api(service_url)
            if not service_info:
                await self.update_stats("datasets_processed")
                await self.update_stats("failed_datasets", title)
                return True

            # Parse tags and categories from CSV
            tags_str = csv_metadata.get("tags", "")
            tags = [t.strip() for t in tags_str.split(",")] if tags_str else None

            categories_str = csv_metadata.get("categories", "")
            groups = (
                [c.strip() for c in categories_str.split(",")]
                if categories_str
                else None
            )

            # Prepare metadata with all available fields
            package_meta = DatasetMetadataWithFields(
                id=service_info.get("serviceItemId") or csv_metadata.get("id"),
                title=title,
                description=csv_metadata.get("description"),
                organization=csv_metadata.get("source"),
                metadata_created=csv_metadata.get("created"),
                metadata_modified=csv_metadata.get("modified"),
                city="Chemnitz",
                state="Saxony",
                country="Germany",
                tags=tags,
                groups=groups,
                url=csv_metadata.get("url"),
                author=csv_metadata.get("owner"),
                is_geo=True,
            )

            # Get all features (layers and tables)
            layers = service_info.get("layers", [])
            tables = service_info.get("tables", [])
            all_layers = layers + tables

            if not all_layers:
                await self.update_stats("datasets_processed")
                await self.update_stats("failed_datasets", title)
                return True

            # Download layers concurrently with limited concurrency
            layer_semaphore = asyncio.Semaphore(
                self.max_workers if self.use_parallel else 1
            )

            async def download_with_semaphore(layer_info):
                _layer_id = layer_info.get("id", 0)
                _layer_name = layer_info.get("name", f"layer_{_layer_id}")
                _safe_layer_name = sanitize_title(_layer_name)

                # Skip if layer file already exists
                if self.use_file_system:
                    layer_file = (
                        self.output_dir / safe_title / f"{_safe_layer_name}.json"
                    )
                    if layer_file.exists():
                        self.logger.debug(f"Layer already exists: {_layer_name}")
                        await self.update_stats("layers_downloaded")
                        return _layer_name, True, None, True

                async with layer_semaphore:
                    _success, _data = await self.download_layer_data_by_api(
                        service_url, _layer_id, _layer_name
                    )
                    return _layer_name, _success, _data, False

            # Create download tasks
            download_tasks = [download_with_semaphore(layer) for layer in all_layers]

            # Wait for all downloads
            results = await asyncio.gather(*download_tasks, return_exceptions=True)

            # Process results and save files
            files_saved = 0
            dataset_dir = self.output_dir / safe_title
            if self.use_file_system:
                dataset_dir.mkdir(exist_ok=True)

            for result in results:
                if isinstance(result, Exception):
                    await self.update_stats("failed_datasets", title)
                    continue
                layer_name, success, data, exists = result
                if success and exists:
                    files_saved += 1
                    continue
                if not success:
                    await self.update_stats("failed_datasets", title)
                    continue

                if self.use_file_system:
                    safe_layer_name = sanitize_title(layer_name)
                    save_file_with_task(
                        dataset_dir / f"{safe_layer_name}.json",
                        json.dumps(data, ensure_ascii=False, indent=2),
                    )
                    files_saved += 1

            if files_saved > 0:
                await self.update_stats("files_downloaded", files_saved)

                if self.use_file_system:
                    save_file_with_task(
                        dataset_dir / "_metadata.json", package_meta.to_json()
                    )
                    save_file_with_task(
                        dataset_dir / "_dataset_info.json",
                        json.dumps(service_info, ensure_ascii=False, indent=2),
                    )

                if self.use_embeddings and self.vector_db_buffer:
                    await self.vector_db_buffer.add(package_meta)

            await self.update_stats("datasets_processed")
            return True

        except Exception as e:
            self.logger.error(f"\tError processing dataset {title}: {e}", exc_info=True)
            await self.update_stats("datasets_processed")
            await self.update_stats("failed_datasets", title)
            await self.update_stats("errors")
            return False

    # 3.
    async def process_dataset(self, metadata: Dict[str, str]) -> bool:
        """Process a single dataset"""
        title = metadata["title"]
        dataset_type = metadata["type"]

        try:
            if dataset_type == "Feature Service":
                return await self.download_feature_service_data(metadata)
            elif dataset_type == "Map Service":
                return await self.download_map_service_metadata(metadata)
            else:
                self.logger.warning(f"Unknown dataset type '{dataset_type}': {title}")
                await self.update_stats("datasets_processed")
                return True

        except Exception as e:
            self.logger.error(f"❌ Error processing {title}: {e}")
            await self.update_stats("failed_datasets", title)
            await self.update_stats("errors")
            return False

    # 2.
    async def load_datasets_metadata_from_csv(self) -> List[Dict[str, str]]:
        """Load dataset metadata from CSV file asynchronously"""
        datasets = []
        async with aiofiles.open(self.csv_file_path, "r", encoding="utf-8") as file:
            content = await file.read()

        # Parse CSV content
        csv_file = io.StringIO(content)
        reader = csv.DictReader(csv_file)

        for row in reader:
            if row.get("url") and row.get("url").strip():
                # Extract all available metadata from CSV
                datasets.append(
                    {
                        "id": row.get("id", "").strip(),
                        "owner": row.get("owner", "").strip(),
                        "created": row.get("created", "").strip(),
                        "modified": row.get("modified", "").strip(),
                        "title": row.get("title", "").strip(),
                        "type": row.get("type", "").strip(),
                        "description": row.get("description", "").strip(),
                        "tags": row.get("tags", "").strip(),
                        "snippet": row.get("snippet", "").strip(),
                        "categories": row.get("categories", "").strip(),
                        "accessInformation": row.get("accessInformation", "").strip(),
                        "licenseInfo": row.get("licenseInfo", "").strip(),
                        "culture": row.get("culture", "").strip(),
                        "url": row.get("url").strip(),
                        "access": row.get("access", "").strip(),
                        "license": row.get("license", "").strip(),
                        "source": row.get("source", "").strip(),
                        "extent": row.get("extent", "").strip(),
                        "industries": row.get("industries", "").strip(),
                    }
                )
        await self.update_stats("datasets_found", len(datasets))
        return datasets

    # 1.
    async def process_all_datasets(self):
        """Download all datasets with optimized batching and concurrency"""
        self.logger.info("Starting to download Chemnitz Open Data")

        # region Load datasets metadatas
        metadatas = await self.load_datasets_metadata_from_csv()
        if not metadatas:
            self.logger.error("No datasets found in CSV file")
            return

        self.logger.info(
            f"Found {len(metadatas)} datasets for download with {self.max_workers} workers"
        )
        # endregion

        progress_task = asyncio.create_task(self.progress_reporter())

        semaphore = asyncio.Semaphore(self.max_workers if self.use_parallel else 1)

        async def process_with_semaphore(metadata: Dict[str, str]):
            async with semaphore:
                return await self.process_dataset(metadata)

        # Process in batches to avoid overwhelming memory
        for i in range(0, len(metadatas), self.batch_size):
            batch = metadatas[i : i + self.batch_size]

            tasks = [process_with_semaphore(metadata) for metadata in batch]
            await asyncio.gather(*tasks, return_exceptions=True)

        # Cancel progress reporter
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

        if self.use_embeddings:
            await self.vector_db_buffer.flush()
        if self.use_store:
            await self.dataset_db_buffer.flush()

        # STATS
        self.logger.info("🎉 Download completed!")
        await self.print_final_report()

    # endregion


# region MAIN
async def async_main():
    """Async main function with optimized settings"""
    csv_file = "open_data_portal_stadt_chemnitz.csv"
    if not Path(csv_file).exists():
        print(f"❌ File {csv_file} not found!")
        print("Make sure that CSV with datasets links is in the same folder.")
        return 1

    import argparse

    parser = argparse.ArgumentParser(description="Download Chemnitz open data")
    parser.add_argument(
        "--output",
        "-o",
        default="./chemnitz",
        help="Output directory for downloaded datasets (default: ./chemnitz)",
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=128,
        help="Number of parallel workers (default: 128)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0.05,
        help="Delay between requests in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)",
    )
    parser.add_argument(
        "--connection-limit",
        type=int,
        default=100,
        help="Total connection pool size (default: 100)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum retry attempts for failed requests (default: 1)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Debug output")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    # Disable noisy third-party library logs
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)

    try:
        async with Chemnitz(
            csv_file,
            output_dir=args.output,
            max_workers=args.max_workers,
            delay=args.delay,
            use_embeddings=False,
            use_store=False,
            use_file_system=True,
            connection_limit=args.connection_limit,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            use_parallel=True,
            use_playwright=True,
        ) as downloader:
            await downloader.process_all_datasets()
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Download interrupted by user (Ctrl+C)")
        sys.exit(0)
    except asyncio.CancelledError:
        print("\n\n⚠️ Download cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


def main():
    """Synchronous entry point"""
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
# endregion
