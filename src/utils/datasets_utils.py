import json
import logging
import shutil
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any

# import matplotlib
#
# matplotlib.use("Agg")  # Use non-interactive backend to prevent plots from showing

import numpy as np
from scipy import stats
import warnings

from src.datasets.datasets_metadata import Field, FieldString, FieldDate, FieldNumeric
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


def detect_distribution(series: np.ndarray, min_size: int = 30) -> str:
    try:
        data = series.astype(float)

        if len(data) < min_size:
            return "none"

        if np.ptp(data) == 0:  # range = max - min
            return "none"

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

        from fitter import Fitter

        f = Fitter(data.tolist(), distributions=["expon", "lognorm", "gamma"])
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
        logger.error("Error detecting distribution => none", exc_info=True)
        return "none"


def safe_unique_count(values: list[Any]) -> int:
    """
    Safely count unique values, handling unhashable types like dictionaries.
    """
    if not values:
        return 0

    try:
        return len(set(values))
    except TypeError as e:
        # Handle unhashable types by converting to JSON strings
        logging.debug(f"Unhashable types detected, converting to JSON: {e}")

        try:
            json_values = []
            for v in values:
                if isinstance(v, (dict, list)):
                    try:
                        json_values.append(
                            json.dumps(v, sort_keys=True, ensure_ascii=False)
                        )
                    except (TypeError, ValueError, OverflowError) as json_error:
                        # Specific JSON serialization errors
                        logging.debug(
                            f"Failed to serialize value {type(v)}: {json_error}"
                        )
                        json_values.append(f"<unserializable_{type(v).__name__}>")
                elif v is None:
                    json_values.append("null")
                else:
                    try:
                        json_values.append(str(v))
                    except (UnicodeEncodeError, UnicodeDecodeError) as str_error:
                        logging.debug(f"Failed to convert to string: {str_error}")
                        json_values.append(f"<unconvertible_{type(v).__name__}>")

            return len(set(json_values))

        except (MemoryError, RecursionError) as critical_error:
            # Critical errors that we should propagate
            logging.error(f"Critical error in safe_unique_count: {critical_error}")
            raise

        except (AttributeError, RuntimeError) as runtime_error:
            # Runtime issues with the data structure
            logging.warning(f"Runtime error processing values: {runtime_error}")
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
