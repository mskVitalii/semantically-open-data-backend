"""
CLI script for running search benchmarks.

This script provides an easy way to:
1. Run tests with different embedder models and configurations
2. Compare multiple experiments
3. Export results to Excel

Usage examples:
    # Test all questions with a single model
    python -m src.scripts.run_benchmark --model jinaai-jina-embeddings-v3 --limit 10

    # Test all available models
    python -m src.scripts.run_benchmark --all-models --limit 10

    # Test specific questions
    python -m src.scripts.run_benchmark --questions 0,1,2 --model baai-bge-m3

    # Test with multiple configurations
    python -m src.scripts.run_benchmark --model jinaai-jina-embeddings-v3 --limits 5,10,25

    # Export existing reports to Excel
    python -m src.scripts.run_benchmark --export report1,report2,report3
"""

import asyncio
import argparse
import sys
from typing import List
from datetime import datetime

from src.infrastructure.config import EmbedderModel, DEFAULT_EMBEDDER_MODEL
from src.infrastructure.logger import get_prefixed_logger
from src.testing.testing_dto import BulkTestRequest, TestConfig
from src.domain.services.dataset_service import DatasetService, get_dataset_service
from src.domain.services.llm_service import LLMService, get_llm_service_dep
from src.testing.testing_service import get_testing_service

logger = get_prefixed_logger(__name__, "BENCHMARK_CLI")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run search benchmarks with different configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Test execution options
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        choices=[m.value for m in EmbedderModel],
        help="Embedder model to test (default: current default model)",
    )

    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Test all available embedder models",
    )

    parser.add_argument(
        "--limits",
        type=str,
        default="10",
        help="Comma-separated list of limit values to test (e.g., '5,10,25')",
    )

    parser.add_argument(
        "--questions",
        "-q",
        type=str,
        help="Comma-separated question indices to test (e.g., '0,1,2'). If not specified, tests all questions.",
    )

    parser.add_argument(
        "--multi-query",
        action="store_true",
        default=False,
        help="Enable multi-query RAG (generates multiple search queries)",
    )

    parser.add_argument(
        "--city",
        type=str,
        help="Filter results by city",
    )

    parser.add_argument(
        "--country",
        type=str,
        help="Filter results by country",
    )

    parser.add_argument(
        "--state",
        type=str,
        help="Filter results by state",
    )

    # Export options
    parser.add_argument(
        "--export",
        type=str,
        help="Export existing reports to Excel. Provide comma-separated report IDs (e.g., 'report1,report2')",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path for Excel export",
    )

    # Display options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output including individual test results",
    )

    return parser.parse_args()


async def run_tests(
    models: List[EmbedderModel],
    limits: List[int],
    question_indices: List[int] = None,
    city: str = None,
    country: str = None,
    state: str = None,
    multi_query: bool = False,
    verbose: bool = False,
) -> List[str]:
    """Run tests with specified configurations and return report IDs"""

    # Initialize services
    dataset_service = await get_dataset_service()
    llm_service = await get_llm_service_dep()
    testing_service = get_testing_service(dataset_service, llm_service)

    # Create test configurations
    test_configs = []
    for model in models:
        for limit in limits:
            config = TestConfig(
                embedder_model=model,
                limit=limit,
                city=city,
                country=country,
                state=state,
                use_multi_query=multi_query,
                use_llm_interpretation=False,  # Not used in current flow
            )
            test_configs.append(config)

    logger.info(
        f"\n{'='*60}\n"
        f"Starting benchmark with {len(test_configs)} configuration(s)\n"
        f"{'='*60}"
    )

    # Print configurations
    for i, config in enumerate(test_configs, 1):
        logger.info(
            f"Config {i}: {config.embedder_model.value}, limit={config.limit}, "
            f"multi_query={config.use_multi_query}"
        )

    # Run bulk test
    request = BulkTestRequest(
        question_indices=question_indices,
        test_configs=test_configs,
    )

    report = await testing_service.run_bulk_test(request)

    # Display results
    logger.info(f"\n{'='*60}")
    logger.info("BENCHMARK COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Report ID: {report.report_id}")
    logger.info(f"Total tests: {report.total_tests}")
    logger.info(f"Successful: {report.successful_tests}")
    logger.info(f"Failed: {report.failed_tests}")
    logger.info(f"Total time: {report.total_execution_time_seconds:.2f}s")
    logger.info(f"{'='*60}\n")

    if verbose:
        # Show detailed results
        logger.info("\nDetailed Results:")
        for i, result in enumerate(report.results, 1):
            logger.info(f"\n{i}. Question: {result.question[:80]}...")
            logger.info(
                f"   Config: {result.config.embedder_model.value}, limit={result.config.limit}"
            )
            logger.info(f"   Datasets found: {result.datasets_found}")
            if result.datasets:
                logger.info(f"   Top 3 results:")
                for j, ds in enumerate(result.datasets[:3], 1):
                    logger.info(f"     {j}. {ds.title[:60]} (score: {ds.score:.4f})")

    return [report.report_id]


async def export_reports(report_ids: List[str], output_file: str = None):
    """Export reports to Excel"""
    dataset_service = await get_dataset_service()
    llm_service = await get_llm_service_dep()
    testing_service = get_testing_service(dataset_service, llm_service)

    logger.info(f"\nExporting {len(report_ids)} report(s) to Excel...")

    excel_file = testing_service.export_to_excel(
        report_ids=report_ids, output_file=output_file
    )

    logger.info(f"Excel file created: {excel_file}")
    logger.info(
        "\nThe Excel file contains three color-coded sheets:\n"
        "1. Weighted Scores: avg(score * relevance_rating) per question\n"
        "   Color scale: Red (0.0) → Yellow (0.5) → Green (1.0)\n"
        "2. Relevance Metrics: % of relevant datasets per question\n"
        "   Color scale: Red (0%) → Yellow (50%) → Green (100%)\n"
        "3. Detailed Ratings: all datasets with their scores and ratings\n"
        "   Score column uses Red-Yellow-Green color scale\n"
    )


async def main():
    args = parse_args()

    try:
        # Export mode
        if args.export:
            report_ids = [rid.strip() for rid in args.export.split(",")]
            await export_reports(report_ids, args.output)
            return

        # Test mode
        # Parse limits
        limits = [int(x.strip()) for x in args.limits.split(",")]

        # Parse question indices
        question_indices = None
        if args.questions:
            question_indices = [int(x.strip()) for x in args.questions.split(",")]

        # Determine models to test
        if args.all_models:
            models = list(EmbedderModel)
        elif args.model:
            models = [EmbedderModel(args.model)]
        else:
            models = [DEFAULT_EMBEDDER_MODEL]

        # Run tests
        report_ids = await run_tests(
            models=models,
            limits=limits,
            question_indices=question_indices,
            city=args.city,
            country=args.country,
            state=args.state,
            multi_query=args.multi_query,
            verbose=args.verbose,
        )

        # Auto-export if multiple configs tested
        if len(models) * len(limits) > 1:
            logger.info("\nMultiple configurations tested. Creating comparison Excel...")
            await export_reports(report_ids, args.output)

        logger.info(
            f"\nNext steps:\n"
            f"1. Review the report: {report_ids[0]}\n"
            f"2. Rate datasets using the API or web interface\n"
            f"3. Export to Excel for analysis\n"
        )

    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running benchmark: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
