import asyncio
import io
import json
import logging
import random
import sys
from itertools import chain
from typing import Dict, Optional, Set

import aiohttp
import pandas as pd
from aiohttp import ClientTimeout
from playwright.async_api import async_playwright, ViewportSize

from src.datasets.base_data_downloader import BaseDataDownloader
from src.datasets.datasets_metadata import (
    DatasetMetadataWithFields,
    Dataset,
)
from src.infrastructure.logger import get_prefixed_logger
from src.utils.datasets_utils import (
    extract_fields,
)
from src.utils.file import save_file_with_task


class Berlin(BaseDataDownloader):
    """Optimized async class for downloading Berlin open data"""

    # region INIT

    def __init__(
        self,
        output_dir: str = "berlin",
        max_workers: int = 128,
        delay: float = 0.05,
        use_file_system: bool = False,
        use_embeddings: bool = False,
        use_store: bool = False,
        connection_limit: int = 100,
        connection_limit_per_host: int = 30,
        batch_size: int = 50,
        use_parallel: bool = True,
        use_playwright: bool = True,
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
            batch_size: Size of package batches to process
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
        )
        self.base_url = "https://datenregister.berlin.de"
        self.api_url = f"{self.base_url}/api/3/action"
        self.logger = get_prefixed_logger(__name__, "BERLIN")
        self.stats["playwright_downloads"] = 0

        # Track domains that require Playwright
        self.playwright_domains: Set[str] = set()
        self.playwright_lock = asyncio.Lock()

        # Playwright browser instance
        self.browser = None
        self.browser_lock = asyncio.Lock()
        self.browser_context = None
        self.context_lock = asyncio.Lock()

        self.use_parallel = use_parallel
        self.use_playwright = use_playwright

    async def _cleanup_resources(self):
        if self.browser:
            await self.browser.close()

    # endregion

    # region STATS
    async def get_additional_metrics(self) -> list[str]:
        return ["playwright_downloads"]

    # endregion

    # region PLAYWRIGHT
    async def get_or_create_browser(self):
        """Get or create a shared Playwright browser instance with optimizations"""
        async with self.browser_lock:
            if not self.browser:
                p = await async_playwright().start()
                self.browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage",
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        # Performance optimizations
                        "--disable-gpu",
                        "--disable-dev-tools",
                        "--disable-software-rasterizer",
                        "--disable-extensions",
                        "--disable-background-networking",
                        "--disable-background-timer-throttling",
                        "--disable-backgrounding-occluded-windows",
                        "--disable-renderer-backgrounding",
                        "--disable-sync",
                        "--metrics-recording-only",
                        "--mute-audio",
                        "--no-first-run",
                        "--no-default-browser-check",
                    ],
                )
            return self.browser

    async def get_or_create_context(self):
        """Get or create a shared browser context for faster page loads"""
        async with self.context_lock:
            if not self.browser_context:
                browser = await self.get_or_create_browser()
                self.browser_context = await browser.new_context(
                    accept_downloads=True,
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    viewport=ViewportSize(width=1280, height=720),  # Smaller viewport
                    java_script_enabled=True,
                    # Disable unnecessary resources for speed
                    bypass_csp=True,
                    ignore_https_errors=True,
                )

                # Block images, fonts, and stylesheets to speed up loading
                await self.browser_context.route(
                    "**/*",
                    lambda route: route.abort()
                    if route.request.resource_type
                    in ["image", "stylesheet", "font", "media"]
                    else route.continue_(),
                )

            return self.browser_context

    async def _find_download_link(self, page) -> str | None:
        """Find a downloadable link on the page that likely contains data"""
        try:
            # Look for links containing download keywords or file extensions
            download_selectors = [
                # Links with download-related text
                'a:has-text("download")',
                'a:has-text("Download")',
                'a:has-text("herunterladen")',  # German for download
                # Links with file extensions
                'a[href*=".csv"]',
                'a[href*=".json"]',
                'a[href*=".xlsx"]',
                'a[href*=".xls"]',
                # Buttons with download attributes
                "a[download]",
                "button[download]",
            ]

            for selector in download_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        href = await element.get_attribute("href")
                        if href:
                            self.logger.debug(f"Found download link: {href}")
                            return selector
                except Exception:
                    continue

            return None
        except Exception as e:
            self.logger.debug(f"Error finding download link: {e}")
            return None

    async def download_file_playwright(self, url: str) -> bytes | None:
        """Optimized file download using Playwright with reusable context"""
        context = await self.get_or_create_context()

        if "github.com" in url:
            self.logger.error(f"Playwright error for {url}: github.com not supported")
            return None

        try:
            page = await context.new_page()

            try:
                # Load the page first
                await page.goto(url, wait_until="networkidle", timeout=10000)

                # Strategy 1: Check if there are download links on the page
                download_selector = await self._find_download_link(page)

                if download_selector:
                    # Click the download link and wait for download
                    try:
                        async with page.expect_download(timeout=10000) as download_info:
                            await page.click(download_selector)
                            download = await download_info.value
                            path = await download.path()
                            with open(path, "rb") as f:
                                content = f.read()

                        self.logger.debug(
                            f"Downloaded via Playwright (link): {download.suggested_filename}"
                        )
                        await self.update_stats("playwright_downloads")
                        return content
                    except Exception as e:
                        self.logger.debug(f"Click download failed: {e}")

                # Strategy 2: No download links found, wait for automatic download
                # This handles cases where download starts automatically without visible links
                try:
                    async with page.expect_download(timeout=15000) as download_info:
                        download = await download_info.value
                        path = await download.path()
                        with open(path, "rb") as f:
                            content = f.read()

                    self.logger.debug(
                        f"Downloaded via Playwright (auto): {download.suggested_filename}"
                    )
                    await self.update_stats("playwright_downloads")
                    return content
                except Exception as e:
                    self.logger.debug(f"Auto-download timeout: {e}")
                    return None

            finally:
                # Always close the page to free resources
                await page.close()

        except Exception as e:
            self.logger.error(f"Playwright error for {url}: {e}")
            return None

    # endregion

    # region LOGIC STEPS

    @staticmethod
    def get_file_extension(url: str, format_hint: str = None) -> str:
        """Optimized file extension detection"""
        # Quick lookup table

        url_lower = url.lower()
        extensions = [
            ".csv",
            ".json",
            ".xml",
            ".xlsx",
            ".xls",
            ".pdf",
            ".txt",
        ]
        for ext in extensions:
            if url_lower.endswith(ext):
                return ext

        # Format hint lookup
        if format_hint:
            format_lower = format_hint.lower()
            format_extension_map = {
                "csv": ".csv",
                "json": ".json",
                "xml": ".xml",
                "xlsx": ".xlsx",
                "excel": ".xlsx",
                "xls": ".xls",
                "pdf": ".pdf",
                "txt": ".txt",
                "text": ".txt",
            }

            for fmt, ext in format_extension_map.items():
                if fmt in format_lower:
                    return ext

        return ""

    @staticmethod
    def should_skip_resource(resource: Dict) -> bool:
        """Optimized resource skip check"""
        url = resource.get("url", "").lower()
        pkg_format = resource.get("format", "").lower()
        formats = ["csv", "json", "xls", "xlsx"]

        # Check for allowed formats
        return not any(
            pkg_format in formats or url.endswith(f".{indicator}")
            for indicator in formats
        )

    @staticmethod
    def is_geo_resource(resource: Dict) -> bool:
        url = resource.get("url", "").lower()
        pkg_format = resource.get("format", "").lower()

        # Geographic/geospatial formats to skip
        geo_formats = {
            "wms",
            "wfs",
            "wcs",
            "wmts",  # Web map services
            "geojson",
            "kml",
            "kmz",
            "gml",
            "gpx",  # Vector formats
            "shp",
            "shx",
            "dbf",
            "prj",
            "shz",  # Shapefile components
            "gpkg",
            "geopackage",  # GeoPackage
            "geotiff",
            "tif",
            "tiff",  # Raster formats
            "ecw",
            "mrsid",
            "sid",  # Compressed imagery
        }

        # Check format field
        if pkg_format and any(geo_fmt in pkg_format for geo_fmt in geo_formats):
            return True

        # Check URL for geographic extensions
        if url and any(url.endswith(f".{geo_fmt}") for geo_fmt in geo_formats):
            return True

        return False

    # 5.
    async def download_resource_by_api(
        self, resource: dict
    ) -> (bool, list[dict] | None):
        """Optimized file download with streaming"""
        url = resource.get("url")
        try:
            async with self.session.get(
                url, timeout=ClientTimeout(total=600), ssl=False
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()

                if "html" in content_type:
                    # Use Playwright for HTML content that triggers download
                    if not self.use_playwright:
                        return False, None

                    self.logger.debug(f"Using Playwright for: {url}")
                    content = await self.download_file_playwright(url)
                    if content is None:
                        return False, None

                    # Try to detect file type and parse
                    try:
                        # Try CSV first
                        df = pd.read_csv(
                            io.BytesIO(content),
                            encoding="utf-8-sig",
                            sep=None,
                            engine="python",
                        )
                        features = df.to_dict("records")
                        await self.update_stats("files_downloaded")
                        return True, features
                    except Exception:
                        try:
                            # Try Excel
                            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
                            features = df.to_dict("records")
                            await self.update_stats("files_downloaded")
                            return True, features
                        except Exception:
                            # Try JSON
                            try:
                                data = json.loads(content.decode("utf-8"))
                                features = data.get("features", [])
                                await self.update_stats("files_downloaded")
                                return True, features
                            except Exception:
                                return False, None
                elif (
                    "csv" in content_type
                    or "text/plain" in content_type
                    or url.endswith(".csv")
                ):
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
                elif (
                    "json" in content_type
                    or "application/octet-stream" in content_type
                    or url.endswith(".json")
                ):
                    raw = await response.read()
                    try:
                        data = json.loads(raw.decode("utf-8"))
                        features = data.get("features", [])
                        await self.update_stats("files_downloaded")
                        return True, features
                    except Exception:
                        return False, None
                elif (
                    "vnd.ms-excel" in content_type
                    or "spreadsheetml.sheet" in content_type
                    or url.endswith(".xls")
                    or url.endswith(".xlsx")
                ):
                    content = await response.read()
                    try:
                        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
                    except Exception:
                        # Fallback to xlrd for old .xls files
                        try:
                            df = pd.read_excel(io.BytesIO(content), engine="xlrd")
                        except Exception as e:
                            self.logger.error(f"Failed to read Excel file {url}: {e}")
                            return False, None
                    features = df.to_dict("records")
                    await self.update_stats("files_downloaded")
                    return True, features
                else:
                    self.logger.warning(
                        f"Unknown content type {content_type} for {url}"
                    )
                    return False, None

        except Exception as e:
            if "Not Found" in str(e):
                return False, None

            self.logger.error(f"\t❌ Unexpected error: {e}. For {url}")
            return False, None

    # 4.
    async def get_package_details_by_api(self, package_name: str) -> Optional[dict]:
        """Get package details with caching"""
        cached_service_info = await self.get_from_cache(package_name)
        if cached_service_info is not None:
            return cached_service_info
        delay = 1
        retries = 3
        for attempt in range(retries):
            try:
                async with self.session.get(
                    f"{self.api_url}/package_show",
                    params={"id": package_name},
                    timeout=ClientTimeout(total=600),
                    ssl=False,
                ) as response:
                    response.raise_for_status()
                    raw = bytearray()
                    async for chunk in response.content.iter_chunked(4096 * 16):
                        raw.extend(chunk)
                    data = json.loads(raw)

                    if data.get("success"):
                        result = data.get("result")
                        await self.add_to_cache(package_name, result)
                        return result

                    self.logger.warning(f"Package not found or error: {package_name}")
                    return None
            except (
                aiohttp.ClientError,
                aiohttp.http_exceptions.HttpProcessingError,
            ) as e:
                self.logger.error(
                    f"Error fetching package details for {package_name}: {e}"
                )
                if attempt == retries - 1:
                    return None
                await asyncio.sleep(delay + random.random())
                delay *= 2
        return None

    # 3.
    async def process_dataset(self, package_name: str) -> bool:
        """Process dataset with optimized resource handling"""

        # Skip if dataset is known to be unsuitable
        if await self.is_dataset_unsuitable(package_name):
            self.logger.debug(f"Skipping unsuitable dataset from cache: {package_name}")
            await self.update_stats("datasets_processed")
            await self.update_stats("datasets_not_suitable")
            return True

        # Create dataset directory
        dataset_dir = self.output_dir / f"{package_name}"
        metadata_file = dataset_dir / "metadata.json"

        # Skip if already processed successfully
        # if self.use_file_system:
        #     if metadata_file.exists():
        #         self.logger.debug(f"Dataset already processed: {package_name}")
        #         await self.update_stats("datasets_processed")
        #         await self.update_stats("files_downloaded")
        #
        #         # Load metadata from file
        #         package_meta = load_metadata_from_file(metadata_file)
        #         if package_meta and self.use_embeddings and self.vector_db_buffer:
        #             await self.vector_db_buffer.add(package_meta)
        #         # I don't need to store the data...
        #         # if self.use_store and self.dataset_db_buffer:
        #         #     dataset = Dataset(metadata=package_meta, data=)
        #         #     await self.dataset_db_buffer.add(dataset)
        #         return True

        # Minimal delay to respect server
        await asyncio.sleep(self.delay)

        try:
            package = await self.get_package_details_by_api(package_name)
            if not package:
                await self.update_stats("errors")
                return False

            meta_id = package.get("id")
            title = package.get("title", package_name)
            resources = package.get("resources", [])

            if not resources:
                self.logger.debug(f"No resources in dataset: {title}")
                await self.mark_dataset_unsuitable(package_name)
                await self.update_stats("datasets_processed")
                await self.update_stats("datasets_not_suitable")
                await self.update_stats("failed_datasets", package_name)
                return True

            self.logger.debug(
                f"Processing dataset: {package_name} ({len(resources)} resources)"
            )

            # Filter and prioritize resources
            valid_resources = []
            is_geo_format = False
            for i, resource in enumerate(resources):
                if self.is_geo_resource(resource):
                    is_geo_format = True
                    continue

                if not resource.get("url") or self.should_skip_resource(resource):
                    continue

                # Prioritize by format
                format_priority = {"json": 1, "csv": 2, "xlsx": 3, "xls": 4}
                format_str = resource.get("format", "").lower()
                priority = min(
                    format_priority.get(fmt, 999)
                    for fmt in format_priority
                    if fmt in format_str
                )
                valid_resources.append((priority, i, resource))

            # Sort by priority
            valid_resources.sort(key=lambda x: x[0])

            if is_geo_format:
                await self.update_stats("geo_packages")

            if len(valid_resources) == 0:
                self.logger.debug(f"Dataset {package_name} has no suitable resources")
                await self.mark_dataset_unsuitable(package_name)
                await self.update_stats("datasets_processed")
                await self.update_stats("datasets_not_suitable")
                return True

            # Download resources concurrently (but limit to avoid overwhelming)
            download_tasks = []

            for priority, i, resource in valid_resources:
                # resource_name = resource.get("name", f"resource_{i}")
                # resource_format = resource.get("format", "")
                # extension = self.get_file_extension(url, resource_format)

                # filename = sanitize_title(f"{resource_name}_{i}{extension}")
                # filepath = dataset_dir / filename

                task = self.download_resource_by_api(resource)
                download_tasks.append(task)

            # Wait for downloads
            results = await asyncio.gather(*download_tasks, return_exceptions=True)
            success_count = sum(
                1 for r in results if not isinstance(r, Exception) and r[0] is True
            )

            if success_count > 0:
                # Extract author/maintainer
                author = package.get("author")
                maintainer = package.get("maintainer")
                maintainer_email = package.get("maintainer_email")

                # Combine author information
                author_str = author if author else None
                if maintainer:
                    if author_str:
                        author_str = f"{author_str}, {maintainer}"
                    else:
                        author_str = maintainer

                package_meta = DatasetMetadataWithFields(
                    id=meta_id,
                    title=title,
                    groups=[group.get("title") for group in package.get("groups", [])],
                    organization=package.get("organization", {}).get("title"),
                    tags=[tag.get("name") for tag in package.get("tags", [])],
                    description=package.get("notes"),
                    metadata_created=package.get("metadata_created"),
                    metadata_modified=package.get("metadata_modified"),
                    author=author_str,
                    url=package.get("url"),
                    city="Berlin",
                    state="Berlin",
                    country="Germany",
                )

                data = list(
                    chain.from_iterable(
                        res[1]
                        for res in results
                        if not isinstance(res, Exception) and res[0] is True and res[1]
                    )
                )
                package_meta.fields = extract_fields(data)

                if self.use_file_system:
                    dataset_dir.mkdir(exist_ok=True)
                    save_file_with_task(metadata_file, package_meta.to_json())

                if self.use_embeddings and self.vector_db_buffer:
                    await self.vector_db_buffer.add(package_meta)

                if self.use_store and self.dataset_db_buffer:
                    dataset = Dataset(metadata=package_meta, data=data)
                    await self.dataset_db_buffer.add(dataset)

            await self.update_stats("datasets_processed")
            return True

        except Exception as e:
            self.logger.error(f"Error processing dataset {package_name}: {e}")
            await self.update_stats("failed_datasets", package_name)
            await self.update_stats("errors")
            return False

    # 2.
    async def get_all_packages_by_api(self) -> list[str]:
        """Get list of all package names with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.debug("Fetching list of all datasets...")
                async with self.session.get(f"{self.api_url}/package_list") as response:
                    response.raise_for_status()
                    data = await response.json()

                    if data.get("success"):
                        packages = data.get("result", [])
                        await self.update_stats("datasets_found", len(packages))
                        self.logger.info(f"Found {len(packages)} datasets")
                        return packages
                    else:
                        self.logger.error(f"API error: {data.get('error')}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2**attempt)  # Exponential backoff
                        continue

            except aiohttp.ClientError as e:
                self.logger.error(
                    f"Error fetching package list (attempt {attempt + 1}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)

        return []

    # 1.
    async def process_all_datasets(self):
        """Optimized download with batching and better concurrency"""
        self.logger.info("Starting optimized Berlin Open Data download")

        # Get list of all packages
        packages = await self.get_all_packages_by_api()
        if not packages:
            self.logger.error("No packages found or error fetching package list")
            return

        self.logger.debug(
            f"Starting download of {len(packages)} datasets with {self.max_workers} workers"
        )

        # Progress reporting task
        progress_task = asyncio.create_task(self.progress_reporter())

        semaphore = asyncio.Semaphore(self.max_workers if self.use_parallel else 1)

        async def process_with_semaphore(package_name: str):
            async with semaphore:
                return await self.process_dataset(package_name)

        try:
            # Process in batches to avoid overwhelming memory
            for i in range(0, len(packages), self.batch_size):
                batch = packages[i : i + self.batch_size]
                tasks = [process_with_semaphore(package) for package in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle exceptions
                for package, result in zip(batch, results):
                    if isinstance(result, Exception):
                        if not isinstance(result, asyncio.CancelledError):
                            self.logger.error(
                                f"Exception in task for {package}: {result}"
                            )
                            await self.update_stats("errors")

                self.logger.debug(
                    f"Completed batch {i // self.batch_size + 1}/{(len(packages) + self.batch_size - 1) // self.batch_size}"
                )

        except asyncio.CancelledError:
            self.logger.info("Processing cancelled by user")
            raise
        finally:
            # Cancel progress reporter
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

            # Final flush of embeddings buffer
            if hasattr(self, "vector_db_buffer") and self.vector_db_buffer:
                await self.vector_db_buffer.flush()

            # STATS
            self.logger.info("Download stopped")
            await self.print_final_report()

    # endregion


# region MAIN
async def main():
    """Main async function with optimized settings"""
    import argparse

    parser = argparse.ArgumentParser(description="Download Berlin open data")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="berlin",
        help="Directory to save data (default: berlin)",
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
        async with Berlin(
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            use_file_system=True,
            delay=args.delay,
            use_store=False,
            use_embeddings=False,
            connection_limit=args.connection_limit,
            batch_size=args.batch_size,
            use_parallel=True,
            use_playwright=False,
        ) as downloader:
            await downloader.process_all_datasets()

    except KeyboardInterrupt:
        print("\n\n⚠️ Download interrupted by user (Ctrl+C)")
        sys.exit(0)
    except asyncio.CancelledError:
        print("\n\n⚠️ Download cancelled")
        sys.exit(0)

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
# endregion
