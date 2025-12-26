from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional

from src.domain.services.dataset_service import DatasetService, get_dataset_service
from src.domain.services.llm_service import LLMService, get_llm_service_dep
from src.infrastructure.logger import get_prefixed_logger
from src.testing.testing_dto import (
    AddQuestionRequest,
    QuestionListResponse,
    BulkTestRequest,
    TestReport,
    TestConfig,
)
from src.testing.testing_service import get_testing_service

logger = get_prefixed_logger("API /testing")

router = APIRouter(prefix="/testing", tags=["testing"])


# region Questions Management


@router.post("/questions")
async def add_question(
    request: AddQuestionRequest,
    dataset_service: DatasetService = Depends(get_dataset_service),
    llm_service: LLMService = Depends(get_llm_service_dep),
):
    """
    Add a new test question to the test suite

    Parameters:
    - question: The question text
    """
    try:
        testing_service = get_testing_service(dataset_service, llm_service)
        question = testing_service.add_question(question=request.question)
        return {"ok": True, "question": question}
    except Exception as e:
        logger.error(f"Failed to add question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/questions", response_model=QuestionListResponse)
async def get_questions(
    dataset_service: DatasetService = Depends(get_dataset_service),
    llm_service: LLMService = Depends(get_llm_service_dep),
):
    """
    Get all test questions
    """
    try:
        testing_service = get_testing_service(dataset_service, llm_service)
        questions = testing_service.get_all_questions()
        return QuestionListResponse(questions=questions, total=len(questions))
    except Exception as e:
        logger.error(f"Failed to get questions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# endregion


# region Testing


@router.post("/run", response_model=TestReport)
async def run_bulk_test(
    request: BulkTestRequest,
    dataset_service: DatasetService = Depends(get_dataset_service),
    llm_service: LLMService = Depends(get_llm_service_dep),
):
    """
    Run bulk testing with specified configurations

    This endpoint executes all or selected test questions with multiple configurations
    and returns a comprehensive report.

    Parameters:
    - question_indices: Optional list of question indices to test (None = all questions)
    - test_configs: List of test configurations to apply

    Example test_configs:
    ```json
    [
        {
            "city": null,
            "state": null,
            "country": null,
            "use_multi_query": true,
            "use_llm_interpretation": false,
            "limit": 5
        },
        {
            "city": "Chemnitz",
            "state": "Saxony",
            "country": "Germany",
            "use_multi_query": false,
            "use_llm_interpretation": true,
            "limit": 10
        }
    ]
    ```
    """
    try:
        testing_service = get_testing_service(dataset_service, llm_service)
        report = await testing_service.run_bulk_test(request)
        return report
    except Exception as e:
        logger.error(f"Failed to run bulk test: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/quick")
async def run_quick_test(
    question_indices: Optional[str] = Query(
        None, description="Comma-separated question indices (e.g., '0,1,2')"
    ),
    use_multi_query: bool = Query(True, description="Enable multi-query RAG"),
    use_llm_interpretation: bool = Query(True, description="Enable LLM interpretation"),
    city: Optional[str] = Query(None, description="Filter by city"),
    state: Optional[str] = Query(None, description="Filter by state"),
    country: Optional[str] = Query(None, description="Filter by country"),
    limit: int = Query(5, ge=1, le=20, description="Results per query"),
    dataset_service: DatasetService = Depends(get_dataset_service),
    llm_service: LLMService = Depends(get_llm_service_dep),
):
    """
    Quick test run with single configuration

    Convenience endpoint for running tests with a single configuration.
    Use the /run endpoint for testing multiple configurations.
    """
    try:
        # Parse question indices
        indices = None
        if question_indices:
            indices = [int(i.strip()) for i in question_indices.split(",")]

        # Create request
        request = BulkTestRequest(
            question_indices=indices,
            test_configs=[
                TestConfig(
                    city=city,
                    state=state,
                    country=country,
                    use_multi_query=use_multi_query,
                    use_llm_interpretation=use_llm_interpretation,
                    limit=limit,
                )
            ],
        )

        testing_service = get_testing_service(dataset_service, llm_service)
        report = await testing_service.run_bulk_test(request)
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid question_indices: {e}")
    except Exception as e:
        logger.error(f"Failed to run quick test: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# endregion


# region Reports


@router.get("/reports")
async def get_all_reports(
    dataset_service: DatasetService = Depends(get_dataset_service),
    llm_service: LLMService = Depends(get_llm_service_dep),
):
    """
    Get list of all test reports (metadata only)

    Returns list of reports sorted by creation time (newest first)
    """
    try:
        testing_service = get_testing_service(dataset_service, llm_service)
        reports = testing_service.get_all_reports()
        return {"reports": reports, "total": len(reports)}
    except Exception as e:
        logger.error(f"Failed to get reports: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/{report_id}", response_model=TestReport)
async def get_report(
    report_id: str,
    dataset_service: DatasetService = Depends(get_dataset_service),
    llm_service: LLMService = Depends(get_llm_service_dep),
):
    """
    Get specific test report by ID

    Returns full report with all test results
    """
    try:
        testing_service = get_testing_service(dataset_service, llm_service)
        report = testing_service.get_report(report_id)

        if report is None:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

        return report
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# endregion
