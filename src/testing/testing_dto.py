from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field

from src.infrastructure.config import EmbedderModel, DEFAULT_EMBEDDER_MODEL, SearchMode


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
    question_en: str = Field(..., description="Question text")
    question_de: str = Field(..., description="Question deutsch text")
    question_ru: str = Field(..., description="Question ru text")
    city: Optional[str] = Field(None, description="Filter by city")
    state: Optional[str] = Field(None, description="Filter by state/region")
    country: Optional[str] = Field(None, description="Filter by country")
    year_from: Optional[int] = Field(
        None, description="Filter datasets from this year (inclusive)"
    )
    year_to: Optional[int] = Field(
        None, description="Filter datasets up to this year (inclusive)"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    expected_datasets: Optional[Dict[str, float]] = Field(
        None,
        description="Expected dataset IDs with relevance ratings (0-1). Used to automatically set relevance_rating in test results.",
    )


class TestConfig(BaseModel):
    """Configuration for a single test run

    Each configuration will be automatically tested in up to 8 variants:
    {WITH/WITHOUT location filters} x {WITH/WITHOUT multi-query} x {WITH/WITHOUT reranker}
    """

    embedder_model: EmbedderModel = Field(
        DEFAULT_EMBEDDER_MODEL, description="Embedder model to use"
    )
    limit: Optional[int] = Field(
        10, ge=1, le=25, description="Number of results per query"
    )


class TestResult(BaseModel):
    """Result of a single test execution"""

    question: str
    question_language: str = Field(
        "en", description="Language of the question (en/de/ru)"
    )
    config: TestConfig
    datasets_found: int
    datasets: List[DatasetResultItem] = Field(
        default_factory=list, description="Found datasets with scores"
    )
    execution_time_seconds: float
    research_questions: Optional[List[str]] = None
    error: Optional[str] = None
    # Applied filters (to distinguish with/without filter variants)
    applied_city_filter: Optional[str] = Field(
        None, description="City filter that was applied"
    )
    applied_state_filter: Optional[str] = Field(
        None, description="State filter that was applied"
    )
    applied_country_filter: Optional[str] = Field(
        None, description="Country filter that was applied"
    )
    applied_year_from: Optional[int] = Field(
        None, description="Year from filter that was applied"
    )
    applied_year_to: Optional[int] = Field(
        None, description="Year to filter that was applied"
    )
    # Applied multi-query flag (to distinguish multi-query vs single-query variants)
    used_multi_query: bool = Field(
        False, description="Whether multi-query RAG was used for this test"
    )
    applied_search_mode: SearchMode = Field(
        SearchMode.DENSE, description="Search mode that was used for this test"
    )
    used_reranker: bool = Field(
        False, description="Whether reranker was used for this test"
    )


class BulkTestRequest(BaseModel):
    """Request for bulk testing"""

    question_indices: Optional[List[int]] = Field(
        None, description="Indices of questions to test (None = all)"
    )
    test_configs: List[TestConfig] = Field(
        ..., description="List of configurations to test"
    )
    # Control which test variants to run
    filters: Optional[bool] = Field(
        None,
        description="None = both variants (with/without filters), True = only WITH filters, False = only WITHOUT filters",
    )
    multiquery: Optional[bool] = Field(
        None,
        description="None = both variants (with/without multiquery), True = only WITH multiquery, False = only WITHOUT multiquery",
    )
    reranker: Optional[bool] = Field(
        None,
        description="None = both variants (with/without reranker), True = only WITH reranker, False = only WITHOUT reranker",
    )
    reranker_candidates: Optional[int] = Field(
        30,
        ge=10,
        le=200,
        description="How many candidates to fetch before reranking (default: limit * 3)",
    )
    search_modes: Optional[List[SearchMode]] = Field(
        [SearchMode.DENSE, SearchMode.SPARSE, SearchMode.HYBRID],
        description="Search modes to test. None = all three (dense, sparse, hybrid). Specify list to run only selected modes.",
    )
    languages: List[str] = Field(
        ["en", "de", "ru"],
        description="Languages to test: ['en', 'de', 'ru']. Each language runs as a separate test variant.",
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

    question_en: str = Field(..., description="Question text in English")
    question_de: str = Field(..., description="Question text in German")
    question_ru: str = Field(..., description="Question text in Russian")
    city: Optional[str] = Field(None, description="Filter by city")
    state: Optional[str] = Field(None, description="Filter by state/region")
    country: Optional[str] = Field(None, description="Filter by country")
    year_from: Optional[int] = Field(
        None, description="Filter datasets from this year (inclusive)"
    )
    year_to: Optional[int] = Field(
        None, description="Filter datasets up to this year (inclusive)"
    )
    expected_datasets: Optional[Dict[str, float]] = Field(
        None,
        description="Expected dataset IDs with relevance ratings (0-1)",
    )


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
