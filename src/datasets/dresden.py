import asyncio
import io
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from aiohttp import (
    ClientTimeout,
    ClientError,
    ClientConnectionError,
    ClientResponseError,
)

from src.datasets.base_data_downloader import BaseDataDownloader
from src.datasets.datasets_metadata import (
    DatasetMetadataWithFields,
    Dataset,
)
from src.infrastructure.logger import get_prefixed_logger
from src.utils.datasets_utils import (
    sanitize_title,
    extract_fields,
    load_metadata_from_file,
)
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
    async def download_file(
        self, url: str, filename: str, dataset_dir: Path
    ) -> (bool, list[dict] | None):
        """
        Download file with optimized async streaming

        Args:
            url: File URL
            filename: Filename to save
            dataset_dir: Dataset directory

        Returns:
            True if file successfully downloaded
        """
        filepath = dataset_dir / filename

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

                    if filepath.suffix == ".csv":
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
                        await self.update_stats("files_downloaded")
                        return True, features
                    else:
                        try:
                            data = await response.json()
                            features = data.get("features", [])
                            await self.update_stats("files_downloaded")
                            return True, features
                        except json.JSONDecodeError:
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

    # 7.
    @staticmethod
    def extract_from_distributions(metadata: dict) -> list[dict]:
        """
        Fallback method to extract download URLs from distribution metadata

        Args:
            metadata: Dataset metadata

        Returns:
            list of download information
        """
        downloads = []

        if not metadata:
            return downloads

        # Search for distributions in metadata
        for subject, predicates in metadata.items():
            # Look for dcat:distribution
            distributions = predicates.get("http://www.w3.org/ns/dcat#distribution", [])

            for dist in distributions:
                if dist.get("type") == "uri":
                    dist_uri = dist.get("value")

                    # Get distribution information
                    dist_info = metadata.get(dist_uri, {})

                    # Look for download URL
                    download_url = None
                    access_urls = dist_info.get(
                        "http://www.w3.org/ns/dcat#downloadURL", []
                    )
                    if not access_urls:
                        access_urls = dist_info.get(
                            "http://www.w3.org/ns/dcat#accessURL", []
                        )

                    if access_urls and access_urls[0].get("type") == "uri":
                        download_url = access_urls[0].get("value")

                    # Look for file format
                    format_info = dist_info.get("http://purl.org/dc/terms/format", [])
                    media_type = dist_info.get(
                        "http://www.w3.org/ns/dcat#mediaType", []
                    )

                    file_format = None
                    if format_info and format_info[0].get("value"):
                        file_format = format_info[0]["value"]
                    elif media_type and media_type[0].get("value"):
                        file_format = media_type[0]["value"]

                    # Look for title
                    title = dist_info.get("http://purl.org/dc/terms/title", [])
                    file_title = (
                        title[0].get("value", "untitled") if title else "untitled"
                    )

                    # Determine file extension
                    extension = ""
                    if file_format:
                        format_lower = file_format.lower()
                        if "csv" in format_lower:
                            extension = ".csv"
                        elif "json" in format_lower:
                            extension = ".json"
                        # elif "xml" in format_lower:
                        #     extension = ".xml"
                        # elif "xlsx" in format_lower or "excel" in format_lower:
                        #     extension = ".xlsx"

                    if download_url:
                        downloads.append(
                            {
                                "url": download_url,
                                "title": file_title,
                                "format": file_format,
                                "extension": extension,
                                "distribution_uri": dist_uri,
                            }
                        )

        return downloads

    # 6.
    async def check_url_availability_by_api(
        self, url: str, format_info: dict
    ) -> Optional[dict]:
        """Check if a URL is available and return download info"""
        try:
            async with self.session.head(
                url, timeout=ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    return {
                        "url": url,
                        "title": "content",
                        "format": format_info["format"],
                        "extension": format_info["ext"],
                    }
        except (
            ClientError,
            ClientConnectionError,
            ClientResponseError,
            asyncio.TimeoutError,
        ):
            pass
        return None

    # 5.
    async def extract_download_urls(
        self, dataset_metadata: dict, dataset_uri: str
    ) -> list[dict]:
        """
        Extract download URLs from metadata using direct content.csv approach

        Args:
            dataset_metadata: Dataset metadata
            dataset_uri: URI of the dataset

        Returns:
            list of dictionaries with download file information
        """
        downloads = []

        if not dataset_uri:
            self.logger.warning("No dataset URI provided")
            return downloads

        # Formats to try in priority order
        format_attempts = [
            {"suffix": "/content.json", "format": "application/json", "ext": ".json"},
            {
                "suffix": "/content.json",
                "format": "application/json",
                "ext": ".geojson",
            },
            {"suffix": "/content.csv", "format": "text/csv", "ext": ".csv"},
            # {"suffix": "/content.xml", "format": "application/xml", "ext": ".xml"},
            # {
            #     "suffix": "/content.xlsx",
            #     "format": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            #     "ext": ".xlsx",
            # },
        ]

        # Check each format concurrently
        check_tasks = []
        for format_info in format_attempts:
            url = f"{dataset_uri}{format_info['suffix']}"
            task = self.check_url_availability_by_api(url, format_info)
            check_tasks.append(task)

        results = await asyncio.gather(*check_tasks, return_exceptions=True)

        # Add successful results
        for result in results:
            if isinstance(result, dict) and result:
                downloads.append(result)

        # Fallback: try to extract from distribution metadata if no direct files found
        if not downloads:
            downloads = self.extract_from_distributions(dataset_metadata)

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
                    publisher_info = predicates.get("http://purl.org/dc/terms/publisher", [])
                    if publisher_info:
                        publisher = publisher_info[0].get("value")

                    # Extract license
                    license_data = predicates.get("http://purl.org/dc/terms/license", [])
                    if license_data:
                        license_info = license_data[0].get("value")

                    # Extract modified date from dataset metadata (fallback)
                    if not modified_date:
                        mod_info = predicates.get("http://purl.org/dc/terms/modified", [])
                        if mod_info:
                            modified_date = mod_info[0].get("value")

                    # Extract maintainer (author)
                    maintainer_info = predicates.get("http://dcat-ap.de/def/dcatde/maintainer", [])
                    if maintainer_info:
                        maintainer_uri = maintainer_info[0].get("value")
                        # Try to get maintainer name from metadata
                        if maintainer_uri and maintainer_uri.startswith("_:"):
                            maintainer_data = dataset_metadata.get(maintainer_uri, {})
                            name_info = maintainer_data.get("http://xmlns.com/foaf/0.1/name", [])
                            if name_info:
                                author = name_info[0].get("value")

                    # Try to extract organization from publisher
                    if publisher and not organization:
                        # Publisher is a URI, try to get its name from metadata
                        if publisher in dataset_metadata:
                            publisher_data = dataset_metadata[publisher]
                            org_name = publisher_data.get("http://xmlns.com/foaf/0.1/name", [])
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

        # Check if dataset already processed by finding directory that starts with {context_id}_{entry_id}_
        if self.use_file_system:
            metadata_file = dataset_dir / "metadata.json"
            if metadata_file.exists():
                self.logger.debug(f"Dataset already processed: {ds_name}")
                await self.update_stats("datasets_processed")
                await self.update_stats("files_downloaded")

                package_meta = load_metadata_from_file(metadata_file)
                if package_meta and self.use_embeddings and self.vector_db_buffer:
                    await self.vector_db_buffer.add(package_meta)

                return True

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
        downloads = await self.extract_download_urls(dataset_metadata, dataset_uri)
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
            async with download_semaphore:
                url = _download_info["url"]
                file_title = _download_info.get("title", "file")
                extension = _download_info.get("extension", "")
                filename = sanitize_title(f"{file_title}{extension}")
                return await self.download_file(url, filename, dataset_dir)

        # Try to download files
        success = False
        data = []

        for download_info in sorted_downloads:
            success, data = await download_with_semaphore(download_info)
            if success:
                break  # Stop after first successful download

        if success:
            package_meta.fields = extract_fields(data)

            # Save metadata
            if self.use_file_system:
                dataset_dir.mkdir(exist_ok=True)
                metadata_file = dataset_dir / "metadata.json"
                content = package_meta.to_json()
                save_file_with_task(metadata_file, content)

            if self.use_embeddings and self.vector_db_buffer:
                await self.vector_db_buffer.add(package_meta)

            if self.use_store and self.dataset_db_buffer:
                dataset = Dataset(metadata=package_meta, data=data)
                await self.dataset_db_buffer.add(dataset)

        await self.update_stats("datasets_processed")
        return success

    # 3.
    async def search_datasets_by_api(self, limit: int = 150, offset: int = 0) -> dict:
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
                    return await response.json()
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
        limit = 150

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

            # Move to next page
            offset += limit

            # If we got fewer results than requested, this is the last page
            if len(children) < limit:
                break

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
