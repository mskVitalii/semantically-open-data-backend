import json
import math
from abc import ABC
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Optional

from src.infrastructure.config import EmbedderModel, DEFAULT_EMBEDDER_MODEL
from src.utils.embeddings_utils import format_metadata_text


def extract_year_from_date(date_str: Optional[str]) -> Optional[int]:
    """Extract year from date string in various formats"""
    if not date_str:
        return None

    try:
        # Try parsing GMT format: "Wed, 01 Dec 2021 13:29:47 GMT"
        from datetime import datetime

        # Try different date formats
        formats = [
            "%a, %d %b %Y %H:%M:%S %Z",  # GMT format
            "%Y-%m-%dT%H:%M:%S",  # ISO format without timezone
            "%Y-%m-%d %H:%M:%S",  # Standard datetime
            "%Y-%m-%d",  # Just date
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.year
            except ValueError:
                continue

        # If all formats fail, try to extract year with regex
        import re

        year_match = re.search(r"\b(19|20)\d{2}\b", date_str)
        if year_match:
            return int(year_match.group(0))

    except Exception:
        pass

    return None


@dataclass
class DatasetMetadata:
    """Simple dataset metadata structure"""

    id: str
    title: str
    description: Optional[str] = None
    organization: Optional[str] = None
    metadata_created: Optional[str] = None
    metadata_modified: Optional[str] = None
    city: Optional[str] = None  # Chemnitz
    state: Optional[str] = None  # Saxony
    country: Optional[str] = None  # Germany
    tags: Optional[list[str]] = None
    groups: Optional[list[str]] = None
    url: Optional[str] = None
    author: Optional[str] = None
    embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL

    def to_searchable_text(self) -> str:
        """Combine title and description for embedding"""
        payload = {
            "title": self.title,
            "description": self.description,
            "organization": self.organization,
            "metadata_created": self.metadata_created,
            "metadata_modified": self.metadata_modified,
            "city": self.city,
            "state": self.state,
            "country": self.country,
            "tags": self.tags,
            "groups": self.groups,
        }
        return format_metadata_text(payload)

        # return f"{self.title}\n{self.description}" if self.description else self.title

    def to_payload(self) -> dict[str, Any]:
        """Convert to Qdrant payload"""
        # Extract year from metadata_created for filtering
        year = extract_year_from_date(self.metadata_created)

        # Handle embedder_model - it can be either EmbedderModel enum or string
        embedder_model_value = (
            self.embedder_model.value
            if isinstance(self.embedder_model, EmbedderModel)
            else self.embedder_model
        )

        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "organization": self.organization,
            "metadata_created": self.metadata_created,
            "metadata_modified": self.metadata_modified,
            "city": self.city,
            "state": self.state,
            "country": self.country,
            "tags": self.tags,
            "groups": self.groups,
            "url": self.url,
            "author": self.author,
            "embedder_model": embedder_model_value,
            "year": year,  # Add year for range filtering
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (for JSON serialization)"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())


@dataclass
class Field(ABC):
    type: str
    name: str
    unique_count: int
    null_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(
            self.to_dict(),
            indent=2,
            ensure_ascii=False,
            cls=DatasetJSONEncoder,
        )


@dataclass
class FieldNumeric(Field):
    type: str = field(default="Numeric", init=False)
    mean: float
    std: float
    quantile_0_min: float
    quantile_25: float
    quantile_50_median: float
    quantile_75: float
    quantile_100_max: float
    distribution: str


@dataclass
class FieldString(Field):
    type: str = field(default="String", init=False)


@dataclass
class FieldDate(Field):
    type: str = field(default="Date", init=False)
    min: datetime
    max: datetime
    mean: datetime


def safe_value(val):
    if val is None:
        return ""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return 0.0
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, list):
        return [safe_value(v) for v in val]
    if isinstance(val, dict):
        return {k: safe_value(v) for k, v in val.items()}
    return val


def make_field(data: dict) -> Field:
    type_map = {
        "Numeric": FieldNumeric,
        "String": FieldString,
        "Date": FieldDate,
    }
    cls = type_map.get(data.get("type"))
    if not cls:
        raise ValueError(f"Unknown field type: {data}")
    return cls(**{k: v for k, v in data.items() if k != "type"})


@dataclass
class DatasetMetadataWithFields(DatasetMetadata):
    """Dataset metadata with additional content field"""

    fields: Optional[dict[str, Field]] = None

    def to_searchable_text(self) -> str:
        """Combine title, description, and content for embedding"""
        base_text = super().to_searchable_text()

        # If you want to include content in the searchable text
        if self.fields:
            return f"{base_text}\n{','.join(self.fields.keys())}"
        return base_text

    def to_payload(self) -> dict[str, Any]:
        """Convert to Qdrant payload including content"""
        payload = super().to_payload()
        if self.fields:
            payload["fields"] = {
                k: safe_value(v.to_dict()) for k, v in self.fields.items()
            }
        return payload

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(
            self,
            indent=2,
            ensure_ascii=False,
            cls=DatasetJSONEncoder,
        )


class DatasetJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles DatasetMetadata objects and datetime"""

    def default(self, obj):
        if isinstance(obj, (DatasetMetadata, DatasetMetadataWithFields)):
            return obj.to_dict()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


@dataclass
class Dataset:
    metadata: DatasetMetadata
    data: list[dict]
