import asyncio
import time
from pathlib import Path

from src.datasets.berlin import Berlin
from src.datasets.chemnitz import Chemnitz
from src.datasets.dresden import Dresden
from src.datasets.leipzig import Leipzig
from src.infrastructure.logger import get_prefixed_logger
from src.infrastructure.paths import PROJECT_ROOT
from src.utils.datasets_utils import safe_delete

logger = get_prefixed_logger(__name__, "BOOTSTRAP")


async def download_berlin():
    """Download Berlin datasets."""
    path = PROJECT_ROOT / "src" / "datasets" / "berlin"
    safe_delete(path, logger)
    start_time = time.perf_counter()

    async with Berlin(
        output_dir=path, use_file_system=False, use_embeddings=True, use_store=True
    ) as downloader:
        await downloader.process_all_datasets()

    elapsed = time.perf_counter() - start_time
    logger.info(f"✅ Berlin download completed in {elapsed:.2f} seconds!")


async def download_chemnitz():
    """Download Chemnitz datasets."""
    csv_file = PROJECT_ROOT / "src" / "datasets" / "open_data_portal_stadt_chemnitz.csv"
    if not Path(csv_file).exists():
        logger.error(f"❌ File {csv_file} not found!")
        return

    path = PROJECT_ROOT / "src" / "datasets" / "chemnitz"
    safe_delete(path, logger)
    start_time = time.perf_counter()

    async with Chemnitz(
        csv_file,
        output_dir=path,
        batch_size=25,
        use_file_system=False,
        use_embeddings=True,
        use_store=True,
    ) as downloader:
        await downloader.process_all_datasets()

    elapsed = time.perf_counter() - start_time
    logger.info(f"✅ Chemnitz download completed in {elapsed:.2f} seconds!")


async def download_leipzig():
    """Download Leipzig datasets."""
    path = PROJECT_ROOT / "src" / "datasets" / "leipzig"
    safe_delete(path, logger)
    start_time = time.perf_counter()

    async with Leipzig(
        output_dir=path,
        use_file_system=False,
        use_embeddings=True,
        use_store=True,
    ) as downloader:
        await downloader.process_all_datasets()

    elapsed = time.perf_counter() - start_time
    logger.info(f"✅ Leipzig download completed in {elapsed:.2f} seconds!")


async def download_dresden():
    """Download Dresden datasets."""
    path = PROJECT_ROOT / "src" / "datasets" / "dresden"
    safe_delete(path, logger)
    start_time = time.perf_counter()

    async with Dresden(
        output_dir=path,
        use_file_system=False,
        use_embeddings=True,
        use_store=True,
    ) as downloader:
        await downloader.process_all_datasets()

    elapsed = time.perf_counter() - start_time
    logger.info(f"✅ Dresden download completed in {elapsed:.2f} seconds!")


async def bootstrap_data():
    """Run all city downloads in parallel."""
    logger.info("🚀 Starting parallel download for all cities...")
    start_time = time.perf_counter()

    # Create tasks for each city
    tasks = [
        download_chemnitz(),
        download_berlin(),
        download_leipzig(),
        download_dresden(),
    ]

    # Run all tasks concurrently
    try:
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start_time
        logger.info(
            f"✅ All downloads completed successfully in {elapsed:.2f} seconds!"
        )
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(
            f"❌ Error during parallel download after {elapsed:.2f} seconds: {e}"
        )
        raise


# if __name__ == "__main__":
#     asyncio.run(bootstrap_data())
