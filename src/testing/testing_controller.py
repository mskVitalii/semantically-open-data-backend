from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import FileResponse

from src.domain.services.dataset_service import DatasetService, get_dataset_service
from src.domain.services.llm_service import LLMService, get_llm_service_dep
from src.infrastructure.logger import get_prefixed_logger
from src.testing.testing_dto import (
    AddQuestionRequest,
    QuestionListResponse,
    BulkTestRequest,
    TestReport,
    UpdateRelevanceRequest,
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
    - question_en: The question text in English
    - question_de: The question text in German
    - question_ru: The question text in Russian
    - city: Optional city filter
    - state: Optional state/region filter
    - country: Optional country filter
    - year_from: Optional year from filter (inclusive)
    - year_to: Optional year to filter (inclusive)
    - expected_datasets: Optional dict of dataset IDs with relevance ratings (0-1)
    """
    try:
        testing_service = get_testing_service(dataset_service, llm_service)
        question = testing_service.add_question(
            question_en=request.question_en,
            question_de=request.question_de,
            question_ru=request.question_ru,
            city=request.city,
            state=request.state,
            country=request.country,
            year_from=request.year_from,
            year_to=request.year_to,
            expected_datasets=request.expected_datasets,
        )
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

    Location filters (city, state, country) are stored in questions themselves.
    By default, each configuration is tested in up to 8 variants:
    {WITH/WITHOUT location filters} x {WITH/WITHOUT multi-query} x {WITH/WITHOUT reranker}

    You can control which variants to run using:
    - filters: None (both), True (only with filters), False (only without filters)
    - multiquery: None (both), True (only with multiquery), False (only without multiquery)
    - reranker: None (both), True (only with reranker), False (only without reranker)

    Parameters:
    - question_indices: Optional list of question indices to test (None = all questions)
    - test_configs: List of test configurations to apply
    - filters: Control filter variants (default: None = both)
    - multiquery: Control multi-query variants (default: None = both)
    - reranker: Control reranker variants (default: None = both)

    Examples:
    - filters=None, multiquery=None, reranker=None → all 8 variants (default)
    - filters=True, multiquery=None, reranker=False → 2 variants (WITH filters, WITHOUT reranker)
    - reranker=True → only WITH reranker variants

    Search modes are controlled globally via `search_modes`:
    - null → all three modes (dense, sparse, hybrid)
    - ["dense", "hybrid"] → only dense and hybrid

    Example test_configs:
    ```json
    [
        {"embedder_model": "jinaai-jina-embeddings-v3", "limit": 25},
        {"embedder_model": "baai-bge-m3", "limit": 25},
        {"embedder_model": "intfloat-multilingual-e5-base", "limit": 25},
        {"embedder_model": "sentence-transformers-labse", "limit": 25}
    ]
    ```

    All tests use maximum accuracy settings (HNSW ef=256, m=64, ef_construct=256).
    """
    try:
        testing_service = get_testing_service(dataset_service, llm_service)
        report = await testing_service.run_bulk_test(request)
        return report
    except Exception as e:
        logger.error(f"Failed to run bulk test: {e}", exc_info=True)
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


# region Relevance Rating


@router.post("/reports/{report_id}/relevance")
async def update_relevance_rating(
    report_id: str,
    request: UpdateRelevanceRequest,
    dataset_service: DatasetService = Depends(get_dataset_service),
    llm_service: LLMService = Depends(get_llm_service_dep),
):
    """
    Update relevance rating for a dataset in a test result

    Relevance rating scale:
    - 0: Not relevant
    - 0.5: Partially relevant
    - 1: Fully relevant

    This rating is used to calculate weighted scores when comparing experiments.
    """
    try:
        testing_service = get_testing_service(dataset_service, llm_service)
        success = testing_service.update_relevance_rating(
            report_id=report_id,
            question=request.question,
            dataset_id=request.dataset_id,
            relevance_rating=request.relevance_rating,
        )

        if success:
            return {"ok": True, "message": "Relevance rating updated successfully"}
        else:
            raise HTTPException(
                status_code=404, detail="Dataset not found in the specified question"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update relevance rating: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reports/{report_id}/collect-candidates")
async def collect_unrated_candidates(
    report_id: str,
    dataset_service: DatasetService = Depends(get_dataset_service),
    llm_service: LLMService = Depends(get_llm_service_dep),
):
    """
    Extract datasets with null relevance_rating from a report into test_data/candidates.json.

    Run this after each test to accumulate unrated dataset candidates.
    Manually fill in ratings (0, 0.5, 1) in the file, then run again — rated entries get removed.
    """
    try:
        testing_service = get_testing_service(dataset_service, llm_service)
        candidates = testing_service.collect_unrated_candidates(report_id)
        total = sum(len(titles) for titles in candidates.values())
        return {
            "ok": True,
            "questions": len(candidates),
            "unrated_datasets": total,
        }
    except Exception as e:
        logger.error(f"Failed to collect candidates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# endregion


# region Excel Export


@router.get("/export/excel")
async def export_to_excel(
    report_ids: str = Query(
        ...,
        description="Comma-separated list of report IDs to compare (e.g., 'report1,report2,report3')",
    ),
    dataset_service: DatasetService = Depends(get_dataset_service),
    llm_service: LLMService = Depends(get_llm_service_dep),
):
    """
    Export comparison of multiple reports to Excel

    Creates an Excel file with five sheets:
    1. **Weighted Scores**: Average weighted score (score * relevance_rating) per question per experiment
    2. **Relevance Metrics**: Percentage of relevant datasets (rating >= 0.5) per question per experiment
    3. **Normalized Weighted**: Average normalized weighted score per question per experiment
       - Scores normalized to 0-100 using global min/max across ALL questions in the experiment
       - Then multiplied by relevance_rating
       - Use this to compare performance across questions within same experiment
    4. **Normalized Per Question**: Average normalized weighted score per question per experiment
       - Scores normalized to 0-100 using min/max within EACH question separately
       - Then multiplied by relevance_rating
       - Use this to compare experiments within same question
    5. **Detailed Ratings**: All datasets with their scores and relevance ratings

    The Excel file is structured as:
    - Sheets 1-4: Rows = Questions, Columns = Experiments (different configurations/models)
    - Sheet 5: Detailed list of all datasets with ratings

    Example usage:
    GET /testing/export/excel?report_ids=report1,report2,report3

    Returns the Excel file for download.
    """
    try:
        # Parse report IDs
        report_id_list = [rid.strip() for rid in report_ids.split(",")]

        if not report_id_list:
            raise HTTPException(
                status_code=400, detail="At least one report ID must be provided"
            )

        testing_service = get_testing_service(dataset_service, llm_service)
        excel_file = testing_service.export_to_excel(report_ids=report_id_list)

        # Get the filename from the path
        from pathlib import Path

        filename = Path(excel_file).name

        return FileResponse(
            excel_file,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=filename,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to export to Excel: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# endregion
