import asyncio
import io
import csv
import json
import logging
import sys
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional

import aiofiles
import pandas as pd

from src.datasets.base_data_downloader import BaseDataDownloader
from src.datasets.datasets_metadata import (
    DatasetMetadataWithFields,
    Dataset,
)
from src.utils.datasets_utils import sanitize_title, safe_delete, extract_fields
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
        is_file_system: bool = True,
        is_embeddings: bool = False,
        is_store: bool = False,
        connection_limit: int = 100,
        connection_limit_per_host: int = 30,
        batch_size: int = 50,
        max_retries: int = 1,
        use_parallel: bool = True,
        use_playwright: bool = True,
    ):
        """
        Initialize optimized downloader

        Args:
            csv_file_path: Path to CSV file with dataset metadata
            output_dir: Directory to save data
            max_workers: Number of parallel workers
            delay: Delay between requests in seconds
            is_file_system: Whether to save datasets to filesystem
            is_embeddings: Whether to generate embeddings
            is_store: Whether to save datasets to DB or not
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
            is_file_system=is_file_system,
            is_embeddings=is_embeddings,
            is_store=is_store,
            connection_limit=connection_limit,
            connection_limit_per_host=connection_limit_per_host,
            batch_size=batch_size,
            max_retries=max_retries,
        )
        self.csv_file_path = csv_file_path
        self.stats["layers_downloaded"] = 0
        self.logger = get_prefixed_logger(__name__, "CHEMNITZ")

        self.use_parallel = use_parallel
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
    ) -> (bool, list[dict] | None):
        """Download data for a single layer with optimized retry logic"""
        formats_to_try = [
            ("geojson", "json"),
            ("csv", "csv"),
            ("json", "json"),
        ]

        for format_name, file_ext in formats_to_try:
            query_url = f"{service_url}/{layer_id}/query"
            params = {
                "where": "1=1",
                "outFields": "*",
                "f": format_name,
                "returnGeometry": "true",
            }

            for attempt in range(self.max_retries):
                try:
                    async with self.session.get(query_url, params=params) as response:
                        if response.status == 200:
                            if format_name == "geojson" or format_name == "json":
                                try:
                                    data = await response.json()
                                    features = data.get("features", [])
                                    await self.update_stats("layers_downloaded")
                                    return True, features

                                except json.JSONDecodeError:
                                    return False, None
                            else:
                                content = await response.read()
                                try:
                                    df = pd.read_csv(
                                        io.BytesIO(content),
                                        encoding="utf-8-sig",
                                        sep=None,
                                        engine="python",
                                    )
                                except UnicodeDecodeError:
                                    df = pd.read_csv(
                                        io.BytesIO(content),
                                        encoding="ISO-8859-1",
                                        sep=None,
                                        engine="python",
                                    )
                                features = df.to_dict("records")
                                await self.update_stats("layers_downloaded")
                                return True, features

                except Exception as e:
                    if attempt < self.max_retries - 1:
                        await self.update_stats("retries")
                        await asyncio.sleep(2**attempt)
                    else:
                        self.logger.error(
                            f"Error downloading layer {layer_name} with format {format_name}: {e}"
                        )
                        return False, None

        self.logger.error(f"⚠ Couldn't download layer {layer_name}")
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

    # 4.
    async def download_feature_service_data(
        self,
        service_url: str,
        title: str,
        description: str = "",
    ) -> bool:
        """Download all data from a feature service with optimized concurrency"""
        try:
            # Get service info
            service_info = await self.get_service_info_by_api(service_url)
            if not service_info:
                await self.update_stats("datasets_processed")
                await self.update_stats("failed_datasets", title)
                return True

                return False

            # Prepare metadata
            package_meta = DatasetMetadataWithFields(
                id=service_info.get("serviceItemId"),
                title=title,
                description=description,
                city="Chemnitz",
                state="Saxony",
                country="Germany",
            )

            # Get all features
            layers = service_info.get("layers", [])
            tables = service_info.get("tables", [])
            all_features = layers + tables

            if not all_features:
                await self.update_stats("datasets_processed")
                await self.update_stats("failed_datasets", title)
                return True

            # Download layers concurrently with limited concurrency
            layer_semaphore = asyncio.Semaphore(
                self.max_workers if self.use_parallel else 1
            )

            async def download_with_semaphore(feature):
                async with layer_semaphore:
                    layer_id = feature.get("id", 0)
                    layer_name = feature.get("name", f"layer_{layer_id}")
                    return await self.download_layer_data_by_api(
                        service_url, layer_id, layer_name
                    )

            # Create download tasks
            download_tasks = [download_with_semaphore(f) for f in all_features]

            # Wait for all downloads
            results = await asyncio.gather(*download_tasks, return_exceptions=True)

            # Count successful downloads
            success_count = sum(
                1
                for result in results
                if not isinstance(result, Exception)
                and isinstance(result, tuple)
                and len(result) == 2
                and result[0] is True
            )
            safe_title = sanitize_title(title)
            if success_count > 0:
                data = list(chain.from_iterable(res[1] for res in results))
                package_meta.fields = extract_fields(data)

                await self.update_stats("files_downloaded")

                if self.is_embeddings and self.vector_db_buffer:
                    await self.vector_db_buffer.add(package_meta)

                if self.is_store and self.dataset_db_buffer:
                    dataset = Dataset(metadata=package_meta, data=data)
                    await self.dataset_db_buffer.add(dataset)

                if self.is_file_system:
                    save_file_with_task(
                        self.output_dir / safe_title / "metadata.json",
                        package_meta.to_json(),
                    )
            else:
                # Clean up empty dataset
                if self.is_file_system:
                    safe_delete(self.output_dir / safe_title, self.logger)

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
        url = metadata["url"]
        dataset_type = metadata["type"]
        description = metadata.get("description", "")

        safe_title = sanitize_title(title)
        if await self.is_dataset_unsuitable(safe_title):
            self.logger.debug(f"Skipping unsuitable dataset from cache: {title}")
            await self.update_stats("datasets_processed")
            await self.update_stats("datasets_not_suitable")
            return True

        try:
            if "Feature Service" == dataset_type:
                return await self.download_feature_service_data(url, title, description)
            else:
                return False

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
                datasets.append(
                    {
                        "title": row.get("title", "").strip(),
                        "url": row.get("url").strip(),
                        "type": row.get("type", "").strip(),
                        "description": row.get("description", "").strip(),
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

        if self.is_embeddings:
            await self.vector_db_buffer.flush()
        if self.is_store:
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
        default=20,
        help="Number of parallel workers (default: 20)",
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
            is_embeddings=False,
            is_store=False,
            is_file_system=True,
            connection_limit=args.connection_limit,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            use_parallel=False,
            use_playwright=True,
        ) as downloader:
            await downloader.process_all_datasets()
        return 0

    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user (Ctrl+C)")
        sys.exit(0)
    except asyncio.CancelledError:
        print("\n\nDownload cancelled")
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
