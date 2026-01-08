from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

from src.infrastructure.config import EmbedderModel, DEFAULT_EMBEDDER_MODEL


class DatasetResultItem(BaseModel):
    """Single dataset result with title and score"""

    title: str = Field(..., description="Dataset title")
    score: float = Field(..., description="Qdrant similarity score")
    dataset_id: str = Field(..., description="Dataset ID")


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
    use_multi_query: bool = Field(True, description="Enable multi-query RAG")
    use_llm_interpretation: bool = Field(
        True, description="Enable LLM interpretation"
    )
    limit: int = Field(5, ge=1, le=20, description="Number of results per query")


class TestResult(BaseModel):
    """Result of a single test execution"""

    question: str
    config: TestConfig
    datasets_found: int
    datasets: List[DatasetResultItem] = Field(default_factory=list, description="Found datasets with scores")
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