import asyncio
import csv
import io
import json
import logging
import sys
import zipfile
from typing import Optional

import pandas as pd

from src.datasets.base_data_downloader import BaseDataDownloader
from src.datasets.datasets_metadata import (
    DatasetMetadataWithFields,
    Dataset,
)
from src.infrastructure.logger import get_prefixed_logger
from src.utils.datasets_utils import (
    sanitize_title,
    safe_delete,
    load_metadata_from_file,
    extract_fields,
)
from src.utils.file import save_file_with_task


class Leipzig(BaseDataDownloader):
    """Optimized async class for downloading Leipzig CSV/JSON data"""

    # region INIT

    def __init__(
        self,
        output_dir: str = "leipzig",
        max_workers: int = 128,
        delay: float = 0.05,
        use_file_system: bool = True,
        use_embeddings: bool = False,
        use_store: bool = False,
        use_parallel: bool = True,
        connection_limit: int = 100,
        connection_limit_per_host: int = 30,
        batch_size: int = 50,
        max_retries: int = 3,
    ):
        """
        Initialize optimized downloader

        Args:
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
        self.base_url = "https://opendata.leipzig.de"
        self.api_url = f"{self.base_url}/api/3/action"
        self.logger = get_prefixed_logger(__name__, "LEIPZIG")

    # endregion

    # region PARSING

    def _parse_csv(self, content: bytes) -> list[dict] | None:
        """Parse CSV content trying multiple encodings"""
        encodings_to_try = ["utf-8-sig", "utf-8", "ISO-8859-1", "latin1", "cp1252"]
        last_error = None

        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(
                    io.BytesIO(content),
                    encoding=encoding,
                    sep=None,
                    engine="python",
                    on_bad_lines="skip",
                )
                return df.to_dict("records")
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                last_error = e
                continue

        self.logger.error(
            f"\t❌ Failed to parse CSV with all encodings. Last error: {last_error}"
        )
        return None

    def _parse_json(self, content: bytes) -> list[dict] | None:
        """Parse JSON content"""
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return data.get("features", [])
            elif isinstance(data, list):
                return data
            else:
                self.logger.warning(f"Unknown JSON type: {type(data)}")
                return None
        except json.JSONDecodeError as e:
            self.logger.error(f"\t❌ JSON parsing error: {e}")
            return None

    def _parse_zip(self, content: bytes) -> list[dict] | None:
        """Extract and parse CSV/JSON/GeoJSON files from a ZIP archive"""
        try:
            all_features = []
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                for name in zf.namelist():
                    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
                    file_content = zf.read(name)

                    if ext == "csv":
                        features = self._parse_csv(file_content)
                    elif ext in {"json", "geojson"}:
                        features = self._parse_json(file_content)
                    else:
                        continue

                    if features:
                        all_features.extend(features)

            if not all_features:
                self.logger.warning("\t⚠️ ZIP archive contained no parseable files")
                return None
            return all_features
        except zipfile.BadZipFile as e:
            self.logger.error(f"\t❌ Bad ZIP file: {e}")
            return None

    # endregion

    # region LOGIC STEPS

    # 7.
    async def download_resource_by_api(
        self, resource: dict
    ) -> tuple[bool, list[dict] | str | None, str]:
        """Download a single resource with retry logic.

        Returns:
            Tuple of (success, data, format) where:
            - format is "csv" or "json"
            - data is raw CSV string or list of dicts for JSON
        """
        try:
            url = resource.get("url")
            resource_format = resource.get("format", "").lower()

            # Download with retry
            for attempt in range(self.max_retries):
                try:
                    async with self.session.get(url) as response:
                        response.raise_for_status()
                        content = await response.read()

                        if resource_format == "csv":
                            # Return raw CSV string for native storage
                            for encoding in ["utf-8-sig", "utf-8", "ISO-8859-1", "latin1"]:
                                try:
                                    csv_text = content.decode(encoding)
                                    # Validate it's parseable
                                    pd.read_csv(
                                        io.StringIO(csv_text), sep=None, engine="python", nrows=1
                                    )
                                    await self.update_stats("layers_downloaded")
                                    return True, csv_text, "csv"
                                except (UnicodeDecodeError, pd.errors.ParserError):
                                    continue
                            return False, None, "csv"

                        elif resource_format == "zip":
                            features = self._parse_zip(content)
                            if features is None:
                                return False, None, "json"
                            await self.update_stats("layers_downloaded")
                            return True, features, "json"

                        else:
                            # JSON/GeoJSON
                            try:
                                data = json.loads(content)
                                if isinstance(data, dict):
                                    features = data.get("features", [data])
                                elif isinstance(data, list):
                                    features = data
                                else:
                                    features = [data]

                                await self.update_stats("layers_downloaded")
                                return True, features, "json"
                            except json.JSONDecodeError as e:
                                self.logger.error(f"\t❌ JSON parsing error: {e}")
                                return False, None, "json"

                except Exception as e:
                    if attempt < self.max_retries - 1:
                        await self.update_stats("retries")
                        await asyncio.sleep(2**attempt)
                    else:
                        self.logger.error(f"\t❌ Error downloading {url}: {e}")
                        await self.mark_url_failed(url)
                        return False, None, "json"

            return False, None, "json"
        except Exception as e:
            self.logger.error(f"\t❌ Unexpected error: {e}")
            return False, None, "json"

    # 6.
    async def process_dataset(self, metadata: dict) -> bool:
        """Download all resources for a package"""
        try:
            package_data = metadata["package_data"]
            target_resources = metadata["target_resources"]
            package_title = package_data.get("title", metadata["package_id"])
            organization = package_data.get("organization", {}).get("title", "Unknown")
            safe_title = sanitize_title(package_title)

            if self.use_file_system:
                dataset_dir = self.output_dir / safe_title
                metadata_file = dataset_dir / "_metadata.json"
                if metadata_file.exists():
                    self.logger.debug(f"Dataset already processed: {package_title}")
                    await self.update_stats("datasets_processed")
                    await self.update_stats("files_downloaded")

                    package_meta = load_metadata_from_file(metadata_file)
                    if package_meta and self.use_embeddings and self.vector_db_buffer:
                        await self.vector_db_buffer.add(package_meta)

                    return True

            self.logger.debug(
                f"Processing dataset: {package_title} ({len(target_resources)} resources)"
            )

            # Prepare metadata
            package_meta = DatasetMetadataWithFields(
                id=package_data.get("id"),
                title=package_title,
                organization=organization,
                author=package_data.get("author"),
                description=package_data.get("notes"),
                metadata_created=package_data.get("metadata_created"),
                metadata_modified=package_data.get("metadata_modified"),
                tags=[tag.get("name") for tag in package_data.get("tags", [])],
                groups=[group.get("title") for group in package_data.get("groups", [])],
                url=f"{self.base_url}/dataset/{package_data.get('name')}",
                city="Leipzig",
                state="Saxony",
                country="Germany",
            )

            async def download_with_semaphore(_resource):
                url = _resource.get("url")
                if not url:
                    return False, None, "json"
                if await self.is_url_failed(url):
                    return False, None, "json"

                return await self.download_resource_by_api(_resource)

            # Sort resources to prioritize CSV
            csv_resources = [r for r in target_resources if r.get("format", "").lower() == "csv"]
            other_resources = [r for r in target_resources if r.get("format", "").lower() != "csv"]
            sorted_resources = csv_resources + other_resources

            # Download all resources
            all_records: list[dict] = []
            any_success = False
            is_geo = False

            dataset_dir = self.output_dir / safe_title
            if self.use_file_system:
                dataset_dir.mkdir(exist_ok=True)

            for resource in sorted_resources:
                resource_format = resource.get("format", "").lower()
                resource_name = sanitize_title(
                    resource.get("name", resource.get("id", "unnamed"))
                )

                # Check if file already exists
                csv_path = dataset_dir / f"{resource_name}.csv"
                json_path = dataset_dir / f"{resource_name}.json"
                if csv_path.exists() or json_path.exists():
                    self.logger.debug(f"File already exists: {resource_name}")
                    await self.update_stats("files_downloaded")
                    any_success = True
                    continue

                success, data, fmt = await download_with_semaphore(resource)
                if success:
                    await self.update_stats("files_downloaded")
                    any_success = True

                    # Collect records for field extraction
                    if data:
                        if fmt == "csv" and isinstance(data, str):
                            try:
                                sample = data[:4096]
                                dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
                                reader = csv.DictReader(io.StringIO(data), dialect=dialect)
                            except csv.Error:
                                reader = csv.DictReader(io.StringIO(data))
                            all_records.extend(reader)
                        elif isinstance(data, list):
                            all_records.extend(data)

                        # Save file in native format
                        if self.use_file_system:
                            if fmt == "csv":
                                save_file_with_task(csv_path, data)
                            else:
                                save_file_with_task(
                                    json_path,
                                    json.dumps(data, ensure_ascii=False, indent=2),
                                )

                    if resource_format == "geojson":
                        is_geo = True
                else:
                    await self.update_stats("errors")

            if any_success:
                package_meta.is_geo = is_geo

                # Extract fields from in-memory records
                if all_records:
                    package_meta.fields = extract_fields(all_records)

                # Save metadata
                if self.use_file_system:
                    metadata_file = dataset_dir / "_metadata.json"
                    save_file_with_task(metadata_file, package_meta.to_json())
                    dataset_info_file = dataset_dir / "_dataset_info.json"
                    save_file_with_task(
                        dataset_info_file,
                        json.dumps(metadata, ensure_ascii=False, indent=2),
                    )

                if self.use_embeddings and self.vector_db_buffer:
                    await self.vector_db_buffer.add(package_meta)

                if self.use_store and self.dataset_db_buffer:
                    dataset = Dataset(metadata=package_meta, data=all_records)
                    await self.dataset_db_buffer.add(dataset)
            else:
                # Clean up empty dataset
                if self.use_file_system:
                    dataset_dir = self.output_dir / safe_title
                    safe_delete(dataset_dir, self.logger)

            return any_success

        except Exception as e:
            self.logger.error(f"\t❌ Error processing package: {e}", exc_info=True)
            await self.update_stats("errors")
            return False

    # 5.
    async def get_package_details_by_api(self, package_id: str) -> Optional[dict]:
        """Get package details with caching and retry logic"""
        await asyncio.sleep(self.delay)  # Rate limiting

        for attempt in range(self.max_retries):
            try:
                async with self.session.get(
                    f"{self.api_url}/package_show", params={"id": package_id}
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    if data["success"]:
                        result = data["result"]
                        return result
                    return None

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await self.update_stats("retries")
                    await asyncio.sleep(2**attempt)
                else:
                    self.logger.warning(
                        f"Failed to get package details for {package_id}: {e}"
                    )
                    return None
        return None

    # 4.
    async def analyze_package(self, package_id: str) -> Optional[dict]:
        """Analyze a package and return metadata if it has target formats"""
        package_data = await self.get_package_details_by_api(package_id)
        if not package_data:
            return None

        target_resources = [
            r
            for r in package_data.get("resources", [])
            if r.get("format", "").lower() in {"csv", "json", "geojson", "zip"}
        ]

        if target_resources:
            await self.update_stats("datasets_found", len(target_resources))

            return {
                "package_id": package_id,
                "package_data": package_data,
                "target_resources": target_resources,
            }

        return None

    # 3.
    async def get_package_list_by_api(self) -> list[str]:
        """Get list of all package IDs with retry logic"""
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(f"{self.api_url}/package_list") as response:
                    response.raise_for_status()
                    data = await response.json()

                    if not data["success"]:
                        self.logger.error(f"API error: {data.get('error')}")
                        return []

                    packages = data["result"]
                    self.logger.info(f"📊 Total packages found: {len(packages)}")
                    return packages

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await self.update_stats("retries")
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    self.logger.error(f"❌ Error fetching package list: {e}")
                    return []
        return []

    # 2.
    async def get_packages_with_target_formats(self) -> list[dict]:
        """Get all packages that contain CSV/JSON/GeoJSON resources"""
        try:
            # Get all package IDs
            all_packages = await self.get_package_list_by_api()
            if not all_packages:
                return []

            # Process packages in batches
            target_packages = []

            semaphore = asyncio.Semaphore(self.max_workers if self.use_parallel else 1)

            async def analyze_with_semaphore(package_id: str):
                async with semaphore:
                    return await self.analyze_package(package_id)

            for i in range(0, len(all_packages), self.batch_size):
                batch = all_packages[i : i + self.batch_size]

                # Analyze batch
                tasks = [analyze_with_semaphore(pkg_id) for pkg_id in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Collect valid results
                for result in results:
                    if isinstance(result, dict) and result:
                        target_packages.append(result)

            datasets_found = await self.get_stat_value("datasets_found")
            self.logger.info(
                f"✅ Found {len(target_packages)} CSV/JSON/GeoJSON packages with {datasets_found} target resources"
            )
            return target_packages

        except Exception as e:
            self.logger.error(f"❌ Error searching datasets: {e}")
            return []

    # 1.
    async def process_all_datasets(self):
        """Main download method"""
        self.logger.info("Start Leipzig Open Data Downloader")

        # Get target packages
        target_packages = await self.get_packages_with_target_formats()

        if not target_packages:
            self.logger.error("❌ No packages found with CSV/JSON data")
            return

        # Process packages in batches
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_with_semaphore(metadata: dict):
            async with semaphore:
                return await self.process_dataset(metadata)

        for i in range(0, len(target_packages), self.batch_size):
            batch = target_packages[i : i + self.batch_size]

            # Process batch
            tasks = [process_with_semaphore(metadata) for metadata in batch]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Progress update
            processed = min(i + self.batch_size, len(target_packages))
            await self.print_progress(processed, len(target_packages))

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
    """Async main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Leipzig open data (CSV/JSON)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./leipzig",
        help="Output directory for downloaded datasets (default: ./leipzig)",
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
        default=128,
        help="Batch size for processing (default: 128)",
    )
    parser.add_argument(
        "--connection-limit",
        type=int,
        default=100,
        help="Total connection pool size (default: 100)",
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
        async with Leipzig(
            output_dir=args.output,
            max_workers=args.max_workers,
            delay=args.delay,
            batch_size=args.batch_size,
            connection_limit=args.connection_limit,
            use_parallel=True,
            use_file_system=True,
            use_embeddings=False,
            use_store=False,
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
