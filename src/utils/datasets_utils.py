import csv
import io
import json
import logging
import shutil
from collections import Counter
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any
import numpy as np
from scipy import stats
import warnings

from src.datasets.datasets_metadata import (
    Field,
    FieldString,
    FieldDate,
    FieldNumeric,
    DatasetMetadataWithFields,
    make_field,
)
from src.infrastructure.logger import get_prefixed_logger

logger = get_prefixed_logger(__name__, "DATASET_UTILS")


def safe_delete(path: Path, _logger: Logger | logging.LoggerAdapter):
    if path.exists():
        if path.is_file():
            path.unlink()
            # logger.debug(f"Deleted file: {path}")
        elif path.is_dir():
            shutil.rmtree(path)
            # logger.debug(f"Deleted folder: {path}")
    else:
        _logger.debug(f"Path doesn't exist: {path}")


def sanitize_title(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    invalid_chars = '<>:"/\\|?* '
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Limit length
    if len(filename) > 200:
        filename = filename[:200]

    return filename


skip_formats = [
    "atom",
    "wms",
    "wfs",
    "geojson",
    "wmts",
    "api",
    "html",
    "htm",
    "zip",
]

allowed_formats = [
    "csv",
    "json",
    "txt",
    "xlsx",
    "xls",
]
allowed_extensions = [f".{f}" for f in allowed_formats]


logging.getLogger("fitter").setLevel(logging.WARNING)


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def detect_distribution(
    series: np.ndarray, min_size: int = 30, max_size: int = 1000
) -> str:
    """
    Detect distribution of data. For performance, skips fitter on large datasets.

    Args:
        series: Numeric data array
        min_size: Minimum number of samples required
        max_size: Maximum size for expensive fitter operations (default: 1000)

    Returns:
        Distribution name ('norm', 'expon', 'lognorm', 'gamma', or 'none')
    """
    try:
        data = series.astype(float)

        if len(data) < min_size:
            return "none"

        if np.ptp(data) == 0:  # range = max - min
            return "none"

        # Quick normality test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(data) <= 5000:
                stat, p_norm = stats.shapiro(data)
                if p_norm > 0.05:
                    return "norm"
            else:
                stat, p_norm = stats.kstest(
                    data, "norm", args=(data.mean(), data.std())
                )
                if p_norm > 0.05:
                    return "norm"

        # Skip expensive fitter operations on large datasets
        if len(data) > max_size:
            # logger.debug(
            #     f"Skipping distribution fitting for large dataset ({len(data)} > {max_size})"
            # )
            return "none"

        # Sample data if still large
        sample_data = data
        if len(data) > 500:
            np.random.seed(42)
            sample_data = np.random.choice(data, size=500, replace=False)

        from fitter import Fitter

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = Fitter(
                sample_data.tolist(), distributions=["expon", "lognorm", "gamma"]
            )
            f.fit()

        # Check if any distributions were fitted successfully
        if not f.fitted_param:
            return "none"

        best = f.get_best(method="sumsquare_error")
        if not best:
            return "none"

        name = list(best.keys())[0]
        min_error = f.summary().sumsquare_error.min()

        if min_error > 0.01:
            return "none"

        if name == "expon":
            stat, p_exp = stats.kstest(data, "expon", args=(data.min(), data.std()))
            if p_exp < 0.05:
                return "none"

        return name
    except Exception:
        logger.debug("Error detecting distribution => none")
        return "none"


def safe_unique_count(values: list[Any]) -> int:
    """
    Safely count unique values, handling unhashable types like dictionaries.
    """
    if not values:
        return 0

    try:
        return len(set(values))
    except TypeError:
        # Handle unhashable types by converting to JSON strings (silently)
        try:
            json_values = []
            for v in values:
                if isinstance(v, (dict, list)):
                    try:
                        json_values.append(
                            json.dumps(v, sort_keys=True, ensure_ascii=False)
                        )
                    except (TypeError, ValueError, OverflowError):
                        json_values.append(f"<unserializable_{type(v).__name__}>")
                elif v is None:
                    json_values.append("null")
                else:
                    try:
                        json_values.append(str(v))
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        json_values.append(f"<unconvertible_{type(v).__name__}>")

            return len(set(json_values))

        except (MemoryError, RecursionError) as critical_error:
            # Critical errors that we should propagate
            logger.error(f"Critical error in safe_unique_count: {critical_error}")
            raise

        except (AttributeError, RuntimeError) as runtime_error:
            # Runtime issues with the data structure
            logger.warning(f"Runtime error processing values: {runtime_error}")
            return len(values)  # Fallback to total count


def extract_fields(data: list[dict]) -> dict[str, Field]:
    if not data:
        return {}

    data = [flatten_dict(record) for record in data]
    fields: dict[str, Field] = {}
    keys = data[0].keys()

    for key in keys:
        values = [record.get(key) for record in data if record.get(key) is not None]
        null_count = sum(1 for record in data if record.get(key) is None)
        unique_count = safe_unique_count(values)

        if all(isinstance(v, (int, float)) for v in values) and values:
            arr = np.array(values)
            q0, q25, q50, q75, q100 = np.percentile(arr, [0, 25, 50, 75, 100])
            mean = float(arr.mean())
            std = float(arr.std())
            fields[key] = FieldNumeric(
                name=key,
                mean=float(mean),
                std=float(std),
                quantile_0_min=float(q0),
                quantile_25=float(q25),
                quantile_50_median=float(q50),
                quantile_75=float(q75),
                quantile_100_max=float(q100),
                unique_count=unique_count,
                null_count=null_count,
            )

        elif all(isinstance(v, str) for v in values) and values:
            try:
                dates = [datetime.fromisoformat(v) for v in values]
                timestamps = [d.timestamp() for d in dates]
                min_date = datetime.fromtimestamp(min(timestamps))
                max_date = datetime.fromtimestamp(max(timestamps))
                mean_date = datetime.fromtimestamp(sum(timestamps) / len(timestamps))
                fields[key] = FieldDate(
                    name=key,
                    min=min_date,
                    max=max_date,
                    mean=mean_date,
                    unique_count=unique_count,
                    null_count=null_count,
                )
            except ValueError:
                top_values = dict(Counter(values).most_common(25))
                fields[key] = FieldString(
                    name=key,
                    unique_count=unique_count,
                    null_count=null_count,
                    top_values=top_values,
                )
        else:
            # Non-string values - convert to str for top_values
            str_values = [str(v) for v in values if v is not None]
            top_values = dict(Counter(str_values).most_common(25)) if str_values else None
            fields[key] = FieldString(
                name=key,
                unique_count=unique_count,
                null_count=null_count,
                top_values=top_values,
            )

    return fields


def _parse_csv_to_records(file_path: Path) -> list[dict]:
    """Parse a CSV file into a list of dicts, trying multiple encodings and delimiters."""
    encodings = ["utf-8-sig", "utf-8", "ISO-8859-1", "latin1", "cp1252"]
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            # Auto-detect delimiter
            try:
                sample = content[:4096]
                dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
                reader = csv.DictReader(io.StringIO(content), dialect=dialect)
            except csv.Error:
                reader = csv.DictReader(io.StringIO(content))
            records = list(reader)
            if records:
                return records
        except (UnicodeDecodeError, csv.Error):
            continue
    return []


def _parse_json_to_records(file_path: Path) -> list[dict]:
    """Parse a JSON file into a list of dicts. Handles ArcGIS and plain formats."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    # ArcGIS format: {"features": [{"attributes": {...}}, ...]}
    if isinstance(data, dict) and "features" in data:
        return data["features"]

    # Plain list of dicts
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]

    return []


def extract_fields_from_folder(
    folder: Path, overwrite: bool = False
) -> dict[str, Field] | None:
    """Extract fields from all data files in a folder.

    1. If overwrite=False and _metadata.json already has fields, returns them as-is.
    2. Reads all .csv and .json data files (skipping _-prefixed service files).
    3. Runs extract_fields on the combined records.
    """
    # 1. Check existing fields
    metadata_file = folder / "_metadata.json"
    if not overwrite and metadata_file.exists():
        metadata = load_metadata_from_file(metadata_file)
        if metadata and metadata.fields:
            return metadata.fields

    # 2. Collect records from all data files
    all_records: list[dict] = []
    for file_path in sorted(folder.iterdir()):
        if file_path.name.startswith("_") or not file_path.is_file():
            continue
        if file_path.suffix == ".csv":
            all_records.extend(_parse_csv_to_records(file_path))
        elif file_path.suffix == ".json":
            all_records.extend(_parse_json_to_records(file_path))

    if not all_records:
        return None

    # 3. Extract fields
    return extract_fields(all_records)


# ---------------------------------------------------------------------------
# Web services lookup (cached id → directory mapping)
# ---------------------------------------------------------------------------

_web_services_index: dict[str, Path] | None = None


def _build_web_services_index() -> dict[str, Path]:
    """Scan all dataset dirs, return {dataset_id: dir_path} for dirs with web_services.json."""
    from src.infrastructure.paths import PROJECT_ROOT

    index: dict[str, Path] = {}
    datasets_root = PROJECT_ROOT / "src" / "datasets"
    for city_dir in datasets_root.iterdir():
        if not city_dir.is_dir() or city_dir.name.startswith("_"):
            continue
        for dataset_dir in city_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            ws_file = dataset_dir / "web_services.json"
            meta_file = dataset_dir / "_metadata.json"
            if ws_file.exists() and meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        data = json.loads(f.read())
                    dataset_id = data.get("id")
                    if dataset_id:
                        index[dataset_id] = dataset_dir
                except Exception:
                    continue
    logger.info(f"Web services index built: {len(index)} geo datasets")
    return index


def get_web_services(dataset_id: str) -> list[dict] | None:
    """Load web_services.json for a given dataset ID. Returns None if not found."""
    global _web_services_index
    if _web_services_index is None:
        _web_services_index = _build_web_services_index()

    dataset_dir = _web_services_index.get(dataset_id)
    if dataset_dir is None:
        return None

    ws_file = dataset_dir / "web_services.json"
    try:
        with open(ws_file, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except Exception:
        return None


def load_metadata_from_file(metadata_file: Path) -> DatasetMetadataWithFields | None:
    """
    Load DatasetMetadataWithFields from JSON file

    Args:
        metadata_file: Path to metadata.json file

    Returns:
        DatasetMetadataWithFields object or None if file doesn't exist or error occurs
    """
    if not metadata_file.exists():
        return None

    # Check if file is empty
    if metadata_file.stat().st_size == 0:
        logger.warning(f"Metadata file is empty: {metadata_file}")
        return None

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            content = f.read().strip()

            # Check if content is empty after stripping whitespace
            if not content:
                logger.warning(f"Metadata file has no content: {metadata_file}")
                return None

            data = json.loads(content)

        # Extract fields separately
        raw_fields = data.pop("fields", {})

        # Create metadata object without fields
        metadata = DatasetMetadataWithFields(**data)

        # Restore fields with proper types
        if raw_fields:
            metadata.fields = {k: make_field(v) for k, v in raw_fields.items()}

        return metadata
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in metadata file {metadata_file}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading metadata from {metadata_file}: {e}", exc_info=True)
        return None
