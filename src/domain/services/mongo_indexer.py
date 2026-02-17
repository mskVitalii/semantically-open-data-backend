import csv
import io
import json
from pathlib import Path

from src.domain.repositories.dataset_repository import DatasetRepository
from src.infrastructure.logger import get_prefixed_logger
from src.infrastructure.paths import PROJECT_ROOT
from src.utils.datasets_utils import load_metadata_from_file, sanitize_title

logger = get_prefixed_logger(__name__, "MONGO_INDEXER")

SKIP_FILES = {"_metadata.json", "_dataset_info.json", "web_services.json"}
CITIES = ["berlin", "chemnitz", "leipzig", "dresden"]
# MongoDB BSON document limit is 16 MB; use a conservative threshold.
_MAX_DOC_SIZE = 15_000_000


def _sanitize_collection_name(name: str) -> str:
    """Make *name* safe for use as a MongoDB collection name.

    MongoDB forbids empty segments created by consecutive dots (``..``),
    the ``$`` character, and null bytes.
    """
    name = name.replace("..", "_").replace("$", "_").replace("\x00", "_")
    # Strip leading/trailing dots — they produce empty namespace segments.
    return name.strip(".")


def _flatten_records(records: list[dict]) -> list[dict]:
    """Unwrap oversized wrapper-dicts that contain the real rows inside.

    Example: ``[{"veranstaltungen": {"veranstaltung": [<10 000 rows>]}}]``
    becomes the 10 000 inner dicts.
    """
    result: list[dict] = []
    for record in records:
        size = len(json.dumps(record, default=str))
        if size <= _MAX_DOC_SIZE:
            result.append(record)
            continue

        # Try to find a nested list of dicts to use instead
        nested = _extract_nested_list(record)
        if nested:
            result.extend(nested)
        else:
            logger.warning(
                f"Skipping oversized record ({size / 1_000_000:.1f} MB), "
                f"could not extract nested rows"
            )
    return result


def _extract_nested_list(obj: dict) -> list[dict] | None:
    """Recursively find the first large list[dict] inside a nested dict."""
    for value in obj.values():
        if isinstance(value, list) and len(value) > 1:
            if value and isinstance(value[0], dict):
                return value
        if isinstance(value, dict):
            found = _extract_nested_list(value)
            if found:
                return found
    return None


def _read_data_file(path: Path) -> list[dict]:
    """Read a CSV or JSON data file into a list of flat dicts.

    - CSV: each row → one dict; extra columns get ``unnamed_N`` keys.
    - JSON: arrays / ArcGIS ``features`` / nested wrappers are unwrapped
      so that every element fits into a single MongoDB document.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            text = path.read_text(encoding="utf-8")
            try:
                sample = text[:4096]
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                reader = csv.DictReader(io.StringIO(text), dialect=dialect)
            except csv.Error:
                reader = csv.DictReader(io.StringIO(text))
            rows = []
            for row in reader:
                extra = row.pop(None, None)
                if extra:
                    for i, val in enumerate(extra):
                        row[f"unnamed_{i}"] = val
                rows.append(row)
            return rows

        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return _flatten_records(data)
            if isinstance(data, dict):
                if "features" in data:
                    return _flatten_records(data["features"])
                return _flatten_records([data])
            return []

    except Exception as e:
        logger.warning(f"Cannot read {path.name}: {e}")
    return []


async def index_datasets_to_mongo(
    repository: DatasetRepository,
    clear_before: bool = False,
) -> dict:
    """Scan city directories and upsert _metadata.json + data files into MongoDB.

    Does NOT touch Qdrant — works only with MongoDB via *repository*.
    """
    if clear_before:
        await repository.delete_all()
        logger.warning("Deleted all MongoDB collections")

    indexed = 0
    skipped = 0
    errors = 0
    total_rows = 0

    for city in CITIES:
        city_path = PROJECT_ROOT / "src" / "datasets" / city
        if not city_path.exists():
            logger.warning(f"City directory not found: {city_path}")
            continue

        for dataset_dir in city_path.iterdir():
            if not dataset_dir.is_dir():
                continue

            metadata_file = dataset_dir / "_metadata.json"
            if not metadata_file.exists():
                skipped += 1
                continue

            metadata = load_metadata_from_file(metadata_file)
            if metadata is None:
                logger.warning(f"Failed to load metadata: {metadata_file}")
                errors += 1
                continue

            try:
                # 1. Upsert metadata (without fields to stay under 16 MB)
                meta_dict = metadata.to_payload()
                meta_dict.pop("fields", None)
                meta_id = await repository.upsert_metadata(meta_dict)

                # 2. Insert data files into a per-dataset collection
                safe_title = sanitize_title(metadata.title)
                collection_name = _sanitize_collection_name(
                    f"{safe_title}_{meta_id}"
                )

                if await repository.collection_has_data(collection_name):
                    logger.debug(f"Collection already has data, skipping: {collection_name}")
                    indexed += 1
                    continue

                data_files = [
                    f for f in dataset_dir.iterdir()
                    if f.is_file() and f.name not in SKIP_FILES
                ]

                rows_for_dataset = 0
                for data_file in data_files:
                    rows = _read_data_file(data_file)
                    if not rows:
                        continue
                    inserted = await repository.insert_data_rows(
                        collection_name, rows,
                    )
                    rows_for_dataset += inserted

                total_rows += rows_for_dataset
                indexed += 1
                logger.debug(f"Indexed {metadata.title}: {rows_for_dataset} rows")

            except Exception as e:
                logger.error(f"Error indexing {dataset_dir.name}: {e}")
                errors += 1

    logger.info(
        f"Done: indexed={indexed}, rows={total_rows}, skipped={skipped}, errors={errors}"
    )
    return {
        "ok": errors == 0,
        "indexed": indexed,
        "total_rows": total_rows,
        "skipped": skipped,
        "errors": errors,
    }
