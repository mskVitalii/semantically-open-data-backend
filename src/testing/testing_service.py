import json
import time
import uuid
from datetime import datetime
from typing import List, Optional

from src.domain.services.dataset_service import DatasetService
from src.domain.services.llm_dto import LLMQuestion
from src.domain.services.llm_service import LLMService
from src.infrastructure.logger import get_prefixed_logger
from src.infrastructure.paths import PROJECT_ROOT
from src.testing.testing_dto import (
    TestQuestion,
    TestConfig,
    TestResult,
    TestReport,
    BulkTestRequest,
    DatasetResultItem,
)
from src.vector_search.embedder import embed_batch_with_ids

logger = get_prefixed_logger(__name__, "TESTING_SERVICE")

# File paths
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
QUESTIONS_FILE = TEST_DATA_DIR / "questions.json"
REPORTS_DIR = TEST_DATA_DIR / "reports"


class TestingService:
    """Service for bulk testing of dataset search"""

    def __init__(self, dataset_service: DatasetService, llm_service: LLMService):
        self.dataset_service = dataset_service
        self.llm_service = llm_service
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure test data directories exist"""
        TEST_DATA_DIR.mkdir(exist_ok=True)
        REPORTS_DIR.mkdir(exist_ok=True)

        # Create empty questions file if it doesn't exist
        if not QUESTIONS_FILE.exists():
            QUESTIONS_FILE.write_text("[]")
            logger.info(f"Created empty questions file: {QUESTIONS_FILE}")

    # region Questions Management

    def add_question(self, question: str) -> TestQuestion:
        """Add a new test question"""
        questions = self.get_all_questions()

        new_question = TestQuestion(
            id=str(uuid.uuid4()),
            question=question,
        )

        questions.append(new_question)
        self._save_questions(questions)

        logger.info(f"Added new test question: {question}")
        return new_question

    def get_all_questions(self) -> List[TestQuestion]:
        """Get all test questions"""
        try:
            with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [TestQuestion(**q) for q in data]
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            return []

    def _save_questions(self, questions: List[TestQuestion]):
        """Save questions to file"""
        with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                [q.model_dump(mode="json") for q in questions],
                f,
                ensure_ascii=False,
                indent=2,
            )

    # endregion

    # region Test Execution

    async def run_single_test(self, question: str, config: TestConfig) -> TestResult:
        """Execute a single test with given configuration"""
        start_time = time.perf_counter()
        research_questions = None

        try:
            # Step 1: Generate research questions if multi-query enabled
            if config.use_multi_query:
                research_questions_objs = await self.llm_service.get_research_questions(
                    question
                )
                research_questions = [q.question for q in research_questions_objs]
            else:
                research_questions_objs = [
                    LLMQuestion(question=question, reason="Direct user query")
                ]

            # Step 2: Generate embeddings with specified embedder model
            questions_list = [
                {"text": q.question, "id": q.question_hash}
                for q in research_questions_objs
            ]
            embeddings = await embed_batch_with_ids(
                questions_list, embedder_model=config.embedder_model
            )

            # Step 3: Search for each research question using specified embedder
            all_datasets = []
            for emb in embeddings:
                datasets = await self.dataset_service.search_datasets_with_embeddings(
                    emb["embedding"],
                    embedder_model=config.embedder_model,
                    city_filter=config.city,
                    state_filter=config.state,
                    country_filter=config.country,
                )
                all_datasets.extend(datasets[: config.limit])

            # Remove duplicates by dataset ID
            seen_ids = set()
            unique_datasets = []
            for ds in all_datasets:
                ds_id = ds.metadata.id
                if ds_id not in seen_ids:
                    seen_ids.add(ds_id)
                    unique_datasets.append(ds)

            # Limit to config.limit
            datasets_found = unique_datasets[: config.limit]

            execution_time = time.perf_counter() - start_time

            return TestResult(
                question=question,
                config=config,
                datasets_found=len(datasets_found),
                datasets=[
                    DatasetResultItem(
                        title=ds.metadata.title,
                        score=ds.score,
                        dataset_id=ds.metadata.id,
                    )
                    for ds in datasets_found
                ],
                execution_time_seconds=execution_time,
                research_questions=research_questions,
                error=None,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"Test failed for question '{question}': {e}", exc_info=True)
            return TestResult(
                question=question,
                config=config,
                datasets_found=0,
                datasets=[],
                execution_time_seconds=execution_time,
                research_questions=research_questions,
                error=str(e),
            )

    async def run_bulk_test(self, request: BulkTestRequest) -> TestReport:
        """Execute bulk testing with multiple configurations"""
        logger.info("Starting bulk test execution")
        start_time = time.perf_counter()

        # Get questions to test
        all_questions = self.get_all_questions()
        if request.question_indices:
            questions_to_test = [
                all_questions[i]
                for i in request.question_indices
                if i < len(all_questions)
            ]
        else:
            questions_to_test = all_questions

        logger.info(
            f"Testing {len(questions_to_test)} questions with {len(request.test_configs)} configurations"
        )

        # Run all tests
        results = []
        total_tests = len(questions_to_test) * len(request.test_configs)
        current_test = 0

        for question in questions_to_test:
            for config in request.test_configs:
                current_test += 1
                logger.info(f"Running test {current_test}/{total_tests}")

                result = await self.run_single_test(question.question, config)
                results.append(result)

        # Calculate statistics
        execution_time = time.perf_counter() - start_time
        successful_tests = sum(1 for r in results if r.error is None)
        failed_tests = sum(1 for r in results if r.error is not None)

        # Create report
        report = TestReport(
            report_id=str(uuid.uuid4()),
            created_at=datetime.now(),
            total_tests=total_tests,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            total_execution_time_seconds=execution_time,
            results=results,
        )

        # Save report
        self._save_report(report)

        logger.info(
            f"Bulk test completed: {successful_tests}/{total_tests} successful in {execution_time:.2f}s"
        )
        return report

    # endregion

    # region Reports Management

    def _save_report(self, report: TestReport):
        """Save test report to file with timestamp-based filename"""
        # Format: 2025_12_26_01_28.json
        timestamp = report.created_at.strftime("%Y_%m_%d_%H_%M")
        report_file = REPORTS_DIR / f"{timestamp}.json"

        # If file exists, add seconds to make it unique
        if report_file.exists():
            timestamp = report.created_at.strftime("%Y_%m_%d_%H_%M_%S")
            report_file = REPORTS_DIR / f"{timestamp}.json"

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved report: {report_file}")

    def get_all_reports(self) -> List[dict]:
        """Get list of all reports (metadata only)"""
        reports = []
        for report_file in REPORTS_DIR.glob("*.json"):
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    reports.append(
                        {
                            "report_id": data["report_id"],
                            "created_at": data["created_at"],
                            "total_tests": data["total_tests"],
                            "successful_tests": data["successful_tests"],
                            "failed_tests": data["failed_tests"],
                        }
                    )
            except Exception as e:
                logger.error(f"Error loading report {report_file}: {e}")

        # Sort by creation time (newest first)
        reports.sort(key=lambda x: x["created_at"], reverse=True)
        return reports

    def get_report(self, report_id: str) -> Optional[TestReport]:
        """Get specific test report by searching through all reports"""
        # Search through all report files to find the one with matching report_id
        for report_file in REPORTS_DIR.glob("*.json"):
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("report_id") == report_id:
                        return TestReport(**data)
            except Exception as e:
                logger.error(f"Error loading report {report_file}: {e}")
                continue

        logger.warning(f"Report {report_id} not found")
        return None

    # endregion


# region Dependency injection
_testing_service: Optional[TestingService] = None


def get_testing_service(
    dataset_service: DatasetService, llm_service: LLMService
) -> TestingService:
    """Get TestingService instance"""
    global _testing_service
    if _testing_service is None:
        _testing_service = TestingService(dataset_service, llm_service)
    return _testing_service


# endregion
