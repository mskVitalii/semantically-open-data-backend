import asyncio
import io
import json
import logging
import sys

import pandas as pd

from src.datasets.base_data_downloader import BaseDataDownloader
from src.datasets.datasets_metadata import (
    DatasetMetadataWithFields,
    Dataset,
)
from src.infrastructure.logger import get_prefixed_logger
from src.utils.datasets_utils import sanitize_title
from src.utils.file import save_file_with_task


class Dresden(BaseDataDownloader):
    """Optimized async class for downloading Dresden open data"""

    # region INIT
    def __init__(
        self,
        output_dir: str = "dresden",
        max_workers: int = 20,
        delay: float = 0.05,
        use_file_system: bool = True,
        use_embeddings: bool = False,
        use_store: bool = False,
        use_parallel: bool = True,
        connection_limit: int = 100,
        connection_limit_per_host: int = 30,
        batch_size: int = 50,
        max_retries: int = 1,
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
        self.base_url = "https://register.opendata.sachsen.de"
        self.search_endpoint = f"{self.base_url}/store/search"
        self.logger = get_prefixed_logger(__name__, "DRESDEN")

    # endregion

    # region LOGIC STEPS

    # 8.
    async def download_file(self, url: str) -> tuple[bool, list[dict] | None]:
        """
        Download and parse file content into list of dicts.

        Tries JSON first, then CSV. Result is always structured data.

        Args:
            url: File URL

        Returns:
            Tuple of (success, parsed data)
        """
        # Check if URL previously failed
        if await self.is_url_failed(url):
            return False, None

        for attempt in range(self.max_retries):
            try:
                async with self.session.get(url) as response:
                    response.raise_for_status()

                    content_type = response.headers.get("content-type", "").lower()
                    content_length = response.headers.get("content-length")

                    if (
                        content_length
                        and int(content_length) < 100
                        and "html" in content_type
                    ):
                        self.logger.warning(
                            f"Response appears to be an error page: {url}"
                        )
                        await self.mark_url_failed(url)
                        return False, None

                    content = await response.read()

                    # Try JSON first
                    try:
                        data = json.loads(content)
                        features = (
                            data.get("features", []) if isinstance(data, dict) else data
                        )
                        return True, features
                    except (json.JSONDecodeError, ValueError):
                        pass

                    # Try CSV
                    try:
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
                        return True, df.to_dict("records")
                    except Exception:
                        return False, None
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await self.update_stats("retries")
                    await asyncio.sleep(2**attempt)
                else:
                    self.logger.error(f"Error downloading {url}: {e}")
                    await self.update_stats("errors")
                    await self.mark_url_failed(url)
                    return False, None
        return False, None

    # 5.
    def extract_download_urls(
        self, dataset_metadata: dict, dataset_uri: str
    ) -> list[dict]:
        """
        Extract download URLs from distribution metadata.

        Reads available formats from DCAT distributions and constructs
        download URLs using the portal's content{extension} pattern.
        Falls back to all known formats if metadata has no format info.

        Args:
            dataset_metadata: Dataset metadata
            dataset_uri: URI of the dataset

        Returns:
            list of dictionaries with download file information
        """
        if not dataset_uri or not dataset_metadata:
            return []

        format_to_ext = {
            "csv": ".csv",
            "json": ".json",
            "xml": ".xml",
            "xlsx": ".xlsx",
            "excel": ".xlsx",
        }

        downloads = []

        for subject, predicates in dataset_metadata.items():
            distributions = predicates.get("http://www.w3.org/ns/dcat#distribution", [])

            if not distributions:
                continue

            # Format info lives on the dataset level, not distribution level
            dataset_format_values = predicates.get(
                "http://purl.org/dc/terms/format", []
            ) or predicates.get("http://www.w3.org/ns/dcat#mediaType", [])

            for dist in distributions:
                if dist.get("type") != "uri":
                    continue

                dist_uri = dist.get("value", "")

                # Extract format from distribution URI fragment (#dist-csv → csv)
                extension = None
                format_name = None
                if "#dist-" in dist_uri:
                    format_name = dist_uri.split("#dist-")[-1]
                    extension = format_to_ext.get(format_name)

                # Fallback: use dataset-level dc:format
                if not extension and dataset_format_values:
                    format_name = dataset_format_values[0].get("value", "")
                    for key, ext in format_to_ext.items():
                        if key in format_name.lower():
                            extension = ext
                            break

                if not extension:
                    continue

                downloads.append(
                    {
                        "url": f"{dataset_uri}/content{extension}",
                        "title": f"content_{len(downloads) + 1}",
                        "format": format_name,
                        "extension": extension,
                    }
                )

        # Fallback: if no formats found in metadata, try all known formats
        if not downloads:
            self.logger.debug(
                f"No formats in distribution metadata for {dataset_uri}, "
                "falling back to all known formats"
            )
            for ext in [".json", ".csv"]:
                downloads.append(
                    {
                        "url": f"{dataset_uri}/content{ext}",
                        "title": "content",
                        "format": ext.lstrip("."),
                        "extension": ext,
                    }
                )

        return downloads

    # 4.
    async def process_dataset(self, dataset_info: dict) -> bool:
        """Process a single dataset with optimized async operations"""

        context_id = dataset_info.get("contextId")
        entry_id = dataset_info.get("entryId")

        if not context_id or not entry_id:
            self.logger.warning("Missing contextId or entryId")
            return False

        await asyncio.sleep(self.delay)  # Minimal delay to respect server

        # Use metadata from dataset_info
        dataset_metadata = dataset_info.get("metadata", {})
        if not dataset_metadata:
            self.logger.warning(f"Missing metadata for dataset {context_id}/{entry_id}")
            return False

        # Extract entry metadata (created/modified dates, etc.)
        info_metadata = dataset_info.get("info", {})
        created_date = None
        modified_date = None

        # Extract created and modified dates from info section
        for uri, predicates in info_metadata.items():
            if uri.startswith("http"):
                created_info = predicates.get("http://purl.org/dc/terms/created", [])
                if created_info:
                    created_date = created_info[0].get("value")

                modified_info = predicates.get("http://purl.org/dc/terms/modified", [])
                if modified_info:
                    modified_date = modified_info[0].get("value")

        # Extract dataset information
        title = "Unknown Dataset"
        dataset_uri = None
        keywords = []
        description = None
        publisher = None
        license_info = None
        author = None
        organization = None

        # Find dataset URI, title, keywords, and description
        for uri, predicates in dataset_metadata.items():
            if (
                uri.startswith("http")
                and "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" in predicates
            ):
                rdf_types = predicates[
                    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                ]
                if any(
                    t.get("value") == "http://www.w3.org/ns/dcat#Dataset"
                    for t in rdf_types
                ):
                    dataset_uri = uri

                    # Extract title
                    title_info = predicates.get("http://purl.org/dc/terms/title", [])
                    if title_info:
                        title = title_info[0].get("value", title)

                    # Extract keywords as an array
                    keyword_info = predicates.get(
                        "http://www.w3.org/ns/dcat#keyword", []
                    )
                    keywords = [
                        kw.get("value") for kw in keyword_info if kw.get("value")
                    ]

                    # Extract description
                    description_info = predicates.get(
                        "http://purl.org/dc/terms/description", []
                    )
                    if description_info:
                        description = description_info[0].get("value")

                    # Extract publisher URI
                    publisher_info = predicates.get(
                        "http://purl.org/dc/terms/publisher", []
                    )
                    if publisher_info:
                        publisher = publisher_info[0].get("value")

                    # Extract license
                    license_data = predicates.get(
                        "http://purl.org/dc/terms/license", []
                    )
                    if license_data:
                        license_info = license_data[0].get("value")

                    # Extract modified date from dataset metadata (fallback)
                    if not modified_date:
                        mod_info = predicates.get(
                            "http://purl.org/dc/terms/modified", []
                        )
                        if mod_info:
                            modified_date = mod_info[0].get("value")

                    # Extract maintainer (author)
                    maintainer_info = predicates.get(
                        "http://dcat-ap.de/def/dcatde/maintainer", []
                    )
                    if maintainer_info:
                        maintainer_uri = maintainer_info[0].get("value")
                        # Try to get maintainer name from metadata
                        if maintainer_uri and maintainer_uri.startswith("_:"):
                            maintainer_data = dataset_metadata.get(maintainer_uri, {})
                            name_info = maintainer_data.get(
                                "http://xmlns.com/foaf/0.1/name", []
                            )
                            if name_info:
                                author = name_info[0].get("value")

                    # Try to extract organization from publisher
                    if publisher and not organization:
                        # Publisher is a URI, try to get its name from metadata
                        if publisher in dataset_metadata:
                            publisher_data = dataset_metadata[publisher]
                            org_name = publisher_data.get(
                                "http://xmlns.com/foaf/0.1/name", []
                            )
                            if org_name:
                                organization = org_name[0].get("value")

                    break

        if not dataset_uri:
            self.logger.warning(
                f"Dataset URI not found in metadata {context_id}/{entry_id}"
            )
            return False

        # Create safe_title after extracting title
        safe_title = sanitize_title(title)
        ds_name = f"{context_id}_{entry_id}_{safe_title}"
        dataset_dir = self.output_dir / ds_name

        self.logger.debug(f"Processing dataset: {ds_name}")

        # Prepare metadata with all available fields
        package_meta = DatasetMetadataWithFields(
            id=f"{context_id}/{entry_id}",
            url=dataset_uri,
            title=title,
            description=description,
            organization=organization,
            metadata_created=created_date,
            metadata_modified=modified_date,
            tags=keywords,
            author=author,
            city="Dresden",
            state="Saxony",
            country="Germany",
        )

        # Extract download links
        downloads = self.extract_download_urls(dataset_metadata, dataset_uri)
        if not downloads:
            await self.update_stats("datasets_processed")
            await self.update_stats("failed_datasets", ds_name)
            return False

        # Sort downloads to prioritize JSON files
        json_downloads = [d for d in downloads if d.get("extension") == ".json"]
        other_downloads = [d for d in downloads if d.get("extension") != ".json"]
        sorted_downloads = json_downloads + other_downloads

        # Download files with limited concurrency
        download_semaphore = asyncio.Semaphore(
            self.max_workers if self.use_parallel else 1
        )

        async def download_with_semaphore(_download_info):
            url = _download_info["url"]
            file_title = _download_info.get("title", "file")
            filename = sanitize_title(f"{file_title}.json")
            filepath = dataset_dir / filename
            if filepath.exists():
                self.logger.debug(f"File already exists: {filename}")
                await self.update_stats("files_downloaded")
                return True, None
            async with download_semaphore:
                success, data = await self.download_file(url)
            if success and data is not None:
                dataset_dir.mkdir(exist_ok=True)
                save_file_with_task(
                    filepath,
                    json.dumps(data, ensure_ascii=False, indent=2),
                )
                await self.update_stats("files_downloaded")
            return success, data

        # Download all resources
        any_success = False
        all_data = []

        for download_info in sorted_downloads:
            success, data = await download_with_semaphore(download_info)
            if success:
                any_success = True
                if data:
                    all_data.extend(data)

        if any_success:
            # Save metadata
            if self.use_file_system:
                dataset_dir.mkdir(exist_ok=True)
                metadata_file = dataset_dir / "_metadata.json"
                content = package_meta.to_json()
                save_file_with_task(metadata_file, content)

                # Save dataset data
                dataset_info_file = dataset_dir / "_dataset_info.json"
                save_file_with_task(
                    dataset_info_file,
                    json.dumps(dataset_info, ensure_ascii=False, indent=2),
                )

            if self.use_embeddings and self.vector_db_buffer:
                await self.vector_db_buffer.add(package_meta)

            if self.use_store and self.dataset_db_buffer:
                dataset = Dataset(metadata=package_meta, data=all_data)
                await self.dataset_db_buffer.add(dataset)

        await self.update_stats("datasets_processed")
        return any_success

    # 3.
    async def search_datasets_by_api(self, limit: int = 100, offset: int = 0) -> dict:
        """
        Search Dresden datasets via API with retry logic

        Args:
            limit: Number of results per request
            offset: Offset for pagination

        Returns:
            API response with datasets
        """
        params = {
            "type": "solr",
            "query": "rdfType:http\\://www.w3.org/ns/dcat#Dataset AND public:true AND resource:*dresden*",
            "limit": limit,
            "offset": offset,
            "sort": "modified desc",
        }

        for attempt in range(self.max_retries):
            try:
                async with self.session.get(
                    self.search_endpoint, params=params
                ) as response:
                    response.raise_for_status()
                    json_ = await response.json()
                    return json_
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await self.update_stats("retries")
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Error searching datasets: {e}")
                    return {}
        return {}

    # 2.
    async def collect_datasets(self) -> list[dict]:
        """Collect all datasets from the API"""
        all_datasets = []
        offset = 0
        limit = 100

        while True:
            # Search for datasets
            search_result = await self.search_datasets_by_api(
                limit=limit, offset=offset
            )

            if not search_result or "resource" not in search_result:
                self.logger.warning("Empty response from API or invalid format")
                break

            children = search_result["resource"].get("children", [])
            total_results = search_result.get("results", 0)

            if not children:
                break

            if offset == 0:
                await self.update_stats("datasets_found", total_results, "set")

            all_datasets.extend(children)

            # Stop if we've collected all results
            if len(all_datasets) >= total_results:
                break

            # Move to next page by actual page size
            offset += len(children)

        return all_datasets

    # 1.
    async def process_all_datasets(self):
        """Download all datasets with optimized async processing"""
        self.logger.info("Starting optimized Dresden Open Data download")

        # First, collect all datasets
        all_datasets = await self.collect_datasets()

        if not all_datasets:
            self.logger.error("No datasets found")
            return

        progress_task = asyncio.create_task(self.progress_reporter())

        semaphore = asyncio.Semaphore(self.max_workers if self.use_parallel else 1)

        async def process_with_semaphore(dataset: dict):
            async with semaphore:
                try:
                    return await self.process_dataset(dataset)
                except Exception as e:
                    self.logger.error(f"Error processing dataset: {e}")
                    await self.update_stats("errors")
                    return False

        for i in range(0, len(all_datasets), self.batch_size):
            batch = all_datasets[i : i + self.batch_size]
            tasks = [process_with_semaphore(dataset) for dataset in batch]
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
    import argparse

    parser = argparse.ArgumentParser(description="Download Dresden open data")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="dresden",
        help="Directory to save data (default: dresden)",
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
        default=200,
        help="Total connection pool size (default: 200)",
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
        async with Dresden(
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            delay=args.delay,
            use_file_system=True,
            use_embeddings=False,
            use_store=False,
            use_parallel=True,
            connection_limit=args.connection_limit,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
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
