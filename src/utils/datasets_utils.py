import json
import logging
import shutil
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
            distribution = detect_distribution(arr)
            fields[key] = FieldNumeric(
                name=key,
                mean=float(mean),
                std=float(std),
                quantile_0_min=float(q0),
                quantile_25=float(q25),
                quantile_50_median=float(q50),
                quantile_75=float(q75),
                quantile_100_max=float(q100),
                distribution=distribution,
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
                fields[key] = FieldString(
                    name=key, unique_count=unique_count, null_count=null_count
                )
        else:
            fields[key] = FieldString(
                name=key, unique_count=unique_count, null_count=null_count
            )

    return fields


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

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract fields separately
        raw_fields = data.pop("fields", {})

        # Create metadata object without fields
        metadata = DatasetMetadataWithFields(**data)

        # Restore fields with proper types
        if raw_fields:
            metadata.fields = {k: make_field(v) for k, v in raw_fields.items()}

        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata from {metadata_file}: {e}", exc_info=True)
        return None
