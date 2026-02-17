import json

from pydantic import BaseModel, Field
from typing import Optional, List
from dataclasses import dataclass

from src.datasets.datasets_metadata import (
    DatasetMetadataWithFields,
    DatasetJSONEncoder,
)
from src.infrastructure.config import EmbedderModel, DEFAULT_EMBEDDER_MODEL, SearchMode


class DatasetSearchRequest(BaseModel):
    """DTO for dataset search request"""

    query: Optional[str] = Field(None, description="Search query")
    embedder_model: EmbedderModel = Field(
        DEFAULT_EMBEDDER_MODEL, description="Embedder model to use for search"
    )
    search_mode: SearchMode = Field(
        SearchMode.DENSE,
        description="Vectorization mode: dense, sparse, or hybrid (RRF fusion)",
    )
    tags: Optional[List[str]] = Field(None, description="List of tags for filtering")
    city: Optional[str] = Field(None, description="Filter by city")
    state: Optional[str] = Field(None, description="Filter by state/region")
    country: Optional[str] = Field(None, description="Filter by country")
    year_from: Optional[int] = Field(None, description="Filter datasets created from this year (inclusive)")
    year_to: Optional[int] = Field(None, description="Filter datasets created until this year (inclusive)")
    use_multi_query: bool = Field(
        False, description="Enable multi-query RAG: LLM generates research questions, searches for each, merges results"
    )
    use_reranker: bool = Field(False, description="Rerank results using cross-encoder reranker")
    reranker_candidates: Optional[int] = Field(
        None, ge=10, le=200,
        description="How many candidates to fetch before reranking (default: limit * 3)",
    )
    limit: int = Field(10, ge=1, le=100, description="Number of results")
    offset: int = Field(0, ge=0, description="Offset for pagination")


class DatasetResponse(BaseModel):
    """DTO for dataset information response"""

    score: float
    metadata: DatasetMetadataWithFields

    model_config = {"from_attributes": True}

    def to_json(self) -> str:
        data = self.model_dump()
        if isinstance(self.metadata, DatasetMetadataWithFields):
            data["metadata"] = self.metadata.to_json()
        return json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            cls=DatasetJSONEncoder,
        )

    def to_dict(self):
        data = self.model_dump()
        if isinstance(self.metadata, DatasetMetadataWithFields):
            data["metadata"] = self.metadata.to_payload()
        return data


class DatasetSearchResponse(BaseModel):
    """DTO for dataset search response"""

    datasets: list[DatasetResponse]
    total: int
    limit: int
    offset: int


@dataclass
class SearchCriteria:
    """Search criteria for datasets"""

    query: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = 10
    offset: int = 0

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
