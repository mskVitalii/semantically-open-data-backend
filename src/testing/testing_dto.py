from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

from src.infrastructure.config import EmbedderModel, DEFAULT_EMBEDDER_MODEL


class DatasetResultItem(BaseModel):
    """Single dataset result with title and score"""

    title: str = Field(..., description="Dataset title")
    score: float = Field(..., description="Qdrant similarity score")
    dataset_id: str = Field(..., description="Dataset ID")
    relevance_rating: Optional[float] = Field(
        None,
        description="Manual relevance rating: 0 (not relevant), 0.5 (partially relevant), 1 (relevant), null (not rated, treated as 1 in calculations)",
    )


class TestQuestion(BaseModel):
    """Test question for bulk testing"""

    id: str = Field(..., description="Unique question ID")
    question: str = Field(..., description="Question text")
    created_at: datetime = Field(default_factory=datetime.now)


class TestConfig(BaseModel):
    """Configuration for a single test run"""

    embedder_model: EmbedderModel = Field(
        DEFAULT_EMBEDDER_MODEL, description="Embedder model to use"
    )
    city: Optional[str] = Field(None, description="Filter by city")
    state: Optional[str] = Field(None, description="Filter by state/region")
    country: Optional[str] = Field(None, description="Filter by country")
    use_multi_query: Optional[bool] = Field(False, description="Enable multi-query RAG")
    use_llm_interpretation: Optional[bool] = Field(
        False, description="Enable LLM interpretation"
    )
    limit: Optional[int] = Field(
        25, ge=1, le=25, description="Number of results per query"
    )


class TestResult(BaseModel):
    """Result of a single test execution"""

    question: str
    config: TestConfig
    datasets_found: int
    datasets: List[DatasetResultItem] = Field(
        default_factory=list, description="Found datasets with scores"
    )
    execution_time_seconds: float
    research_questions: Optional[List[str]] = None
    error: Optional[str] = None


class BulkTestRequest(BaseModel):
    """Request for bulk testing"""

    question_indices: Optional[List[int]] = Field(
        None, description="Indices of questions to test (None = all)"
    )
    test_configs: List[TestConfig] = Field(
        ..., description="List of configurations to test"
    )


class TestReport(BaseModel):
    """Complete test report"""

    report_id: str
    created_at: datetime
    total_tests: int
    successful_tests: int
    failed_tests: int
    total_execution_time_seconds: float
    results: List[TestResult]


class AddQuestionRequest(BaseModel):
    """Request to add a new test question"""

    question: str


class QuestionListResponse(BaseModel):
    """Response with list of test questions"""

    questions: List[TestQuestion]
    total: int


class UpdateRelevanceRequest(BaseModel):
    """Request to update relevance rating for a dataset in a test result"""

    report_id: str = Field(..., description="Report ID")
    question: str = Field(..., description="Question text")
    dataset_id: str = Field(..., description="Dataset ID to rate")
    relevance_rating: float = Field(
        ..., ge=0, le=1, description="Relevance rating: 0, 0.5, or 1"
    )


class ExperimentComparison(BaseModel):
    """Comparison data for Excel export"""

    experiment_name: str = Field(
        ..., description="Name of the experiment configuration"
    )
    embedder_model: str = Field(..., description="Embedder model used")
    config: TestConfig = Field(..., description="Test configuration")
    report_id: str = Field(..., description="Report ID")
