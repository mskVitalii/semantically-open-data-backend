import json
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.responses import StreamingResponse

from .datasets_dto import DatasetSearchRequest, DatasetSearchResponse, DatasetResponse
from .qa_cache.qa_cache import check_qa_cache, set_qa_cache
from ..domain.repositories.dataset_repository import DatasetRepository, get_dataset_repository
from ..domain.services.dataset_service import DatasetService, get_dataset_service
from ..domain.services.mongo_indexer import index_datasets_to_mongo
from ..domain.services.llm_dto import (
    LLMQuestion,
    LLMQuestionWithEmbeddings,
    LLMQuestionWithDatasets,
)
from ..domain.services.llm_service import (
    LLMService,
    get_llm_service_dep,
)
from ..infrastructure.config import (
    IS_DOCKER,
    EmbedderModel,
    DEFAULT_EMBEDDER_MODEL,
    SearchMode,
)
from ..infrastructure.logger import get_prefixed_logger
from ..utils.datasets_utils import get_web_services
from ..vector_search.embedder import embed_multi

logger = get_prefixed_logger("API /datasets")

router = APIRouter(prefix="/datasets", tags=["datasets"])


# region Search Datasets
@router.post("/search_datasets", response_model=DatasetSearchResponse)
async def search_datasets(
    request: DatasetSearchRequest,
    service: DatasetService = Depends(get_dataset_service),
    llm_service: LLMService = Depends(get_llm_service_dep),
) -> DatasetSearchResponse:
    """
    Dataset search

    Supported parameters:
    - query: full-text search by name and description
    - embedder_model: embedder model to use for search
    - use_multi_query: LLM generates research questions, searches for each, merges results
    - use_reranker: rerank results using cross-encoder
    - tags, city, state, country, year_from, year_to: filters
    - limit: number of results (1–100)
    - offset: pagination offset
    """
    try:
        original_limit = request.limit

        if request.use_multi_query:
            result = await _search_with_multi_query(request, service, llm_service)
        else:
            if request.use_reranker:
                request.limit = request.reranker_candidates or min(
                    request.limit * 3, 100
                )
            result = await service.search_datasets(
                request,
                embedder_model=request.embedder_model,
                search_mode=request.search_mode,
            )

        if request.use_reranker and result.datasets:
            result.datasets = await service.rerank_results(
                query=request.query,
                datasets=result.datasets,
                top_n=original_limit,
            )
            result.total = len(result.datasets)

        result.limit = original_limit
        result.datasets = result.datasets[:original_limit]
        result.total = len(result.datasets)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _search_with_multi_query(
    request: DatasetSearchRequest,
    service: DatasetService,
    llm_service: LLMService,
) -> DatasetSearchResponse:
    """Generate research questions via LLM, search for each, merge & deduplicate."""
    research_questions = await llm_service.get_research_questions(request.query)
    logger.info(f"Multi-query: generated {len(research_questions)} research questions")

    # Expand limit for merging: fetch more per sub-query so we have enough after dedup
    per_query_limit = (
        request.reranker_candidates if request.use_reranker else request.limit
    )
    per_query_limit = max(per_query_limit, request.limit)

    # Search for each research question
    seen: dict[str, DatasetResponse] = {}  # id → best DatasetResponse
    for rq in research_questions:
        sub_request = request.model_copy(
            update={"query": rq.question, "limit": per_query_limit}
        )
        sub_result = await service.search_datasets(
            sub_request,
            embedder_model=request.embedder_model,
            search_mode=request.search_mode,
        )
        for ds in sub_result.datasets:
            ds_id = ds.metadata.id
            if ds_id not in seen or ds.score > seen[ds_id].score:
                seen[ds_id] = ds

    # Sort by best score, descending
    merged = sorted(seen.values(), key=lambda d: d.score, reverse=True)

    return DatasetSearchResponse(
        datasets=merged,
        total=len(merged),
        limit=request.limit,
        offset=request.offset,
    )


# endregion


# region Bootstrap
@router.post("/bootstrap")
async def bootstrap(
    use_fs_cache: bool = Query(True, description="Use filesystem cache for datasets"),
    clear_store: bool = Query(True, description="Clear MongoDB store"),
    clear_vector_db: bool = Query(True, description="Clear vector database"),
    index_all_embedders: bool = Query(
        False, description="Index datasets with all available embedders after bootstrap"
    ),
    service: DatasetService = Depends(get_dataset_service),
):
    """
    Bootstrap datasets - clear and reload all data

    Parameters:
    - clear_store: Clear MongoDB collections
    - clear_vector_db: Clear vector database
    - use_fs_cache: If True, use existing files from filesystem cache. If False, clear and redownload all data.
    - index_all_embedders: If True, index datasets with all available embedders (not just default)
    """
    try:
        result = await service.bootstrap_datasets(
            use_fs_cache=use_fs_cache,
            clear_store=clear_store,
            clear_vector_db=clear_vector_db,
        )

        # Optionally index with all embedders
        indexing_results = []
        if result and index_all_embedders:
            logger.info("Indexing datasets with all available embedders...")
            for embedder_model in EmbedderModel:
                if embedder_model != DEFAULT_EMBEDDER_MODEL:
                    logger.info(f"Indexing with {embedder_model.value}...")
                    index_result = await service.index_existing_datasets(
                        embedder_model=embedder_model
                    )
                    indexing_results.append(index_result)

        return {
            "ok": result,
            "additional_indexing": indexing_results if index_all_embedders else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index")
async def index_datasets(
    embedder_model: EmbedderModel = Query(
        DEFAULT_EMBEDDER_MODEL, description="Embedder model to use for indexing"
    ),
    service: DatasetService = Depends(get_dataset_service),
):
    """
    Index existing datasets into vector DB with specified embedder model.

    Always stores both dense and sparse vectors (hybrid).
    Use search_mode in /search_datasets to choose the search strategy.

    Parameters:
    - embedder_model: The embedder model to use for generating embeddings and indexing
    """
    try:
        result = await service.index_existing_datasets(
            embedder_model=embedder_model,
        )
        if not result.get("ok"):
            raise HTTPException(status_code=500, detail=result.get("message"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index_mongo")
async def index_mongo(
    clear_before: bool = Query(False, description="Drop all MongoDB collections before indexing"),
    repository: DatasetRepository = Depends(get_dataset_repository),
):
    """
    Index datasets from filesystem into MongoDB (no Qdrant).

    Scans city directories (berlin, chemnitz, leipzig, dresden):
    - Upserts _metadata.json into the 'metadata' collection (by source id)
    - Loads data files (.csv/.json, skipping _dataset_info.json and web_services.json)
      into per-dataset collections
    """
    try:
        result = await index_datasets_to_mongo(repository, clear_before=clear_before)
        if not result["ok"]:
            raise HTTPException(
                status_code=500,
                detail=f"Completed with {result['errors']} errors",
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_all_data(
    clear_store: bool = Query(True, description="Clear MongoDB store"),
    clear_vector_db: bool = Query(True, description="Clear vector database"),
    clear_fs: bool = Query(False, description="Clear filesystem cache"),
    service: DatasetService = Depends(get_dataset_service),
):
    """
    Clear data from MongoDB, vector DB, and/or filesystem

    Parameters:
    - clear_store: Clear MongoDB collections
    - clear_vector_db: Clear vector database
    - clear_fs: Clear filesystem cache (downloaded datasets)
    """
    try:
        result = await service.clear_all_data(
            clear_store=clear_store,
            clear_vector_db=clear_vector_db,
            clear_fs=clear_fs,
        )
        return {"ok": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# endregion


# region QA
async def step_0_llm_questions(
    step: int, question: str, llm_service: LLMService
) -> list[LLMQuestion]:
    logger.info(f"step: {step}. LLM QUESTIONS start")
    start_0 = time.perf_counter()

    research_questions = await llm_service.get_research_questions(question)
    elapsed_0 = time.perf_counter() - start_0
    logger.info(f"step: {step}. LLM QUESTIONS end (elapsed: {elapsed_0:.2f}s)")
    logger.info(research_questions)
    return research_questions


async def step_1_embeddings(
    step: int,
    research_questions: list[LLMQuestion],
    embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
    search_mode: SearchMode = SearchMode.DENSE,
) -> list[LLMQuestionWithEmbeddings]:
    logger.info(
        f"step: {step}. EMBEDDINGS start ({embedder_model.value}, mode={search_mode.value})"
    )
    start_1 = time.perf_counter()

    texts = [q.question for q in research_questions]
    vectors = await embed_multi(texts, embedder_model=embedder_model, mode=search_mode)

    questions_with_embeddings: list[LLMQuestionWithEmbeddings] = [
        LLMQuestionWithEmbeddings(
            question=q.question,
            reason=q.reason,
            embeddings=vec,
        )
        for q, vec in zip(research_questions, vectors)
    ]

    elapsed_1 = time.perf_counter() - start_1
    logger.info(f"step: {step}. EMBEDDINGS end (elapsed: {elapsed_1:.2f}s)")
    return questions_with_embeddings


def _enrich_with_web_services(datasets: list) -> list[dict]:
    """Convert datasets to dicts and attach web_services for geo datasets."""
    result = []
    for ds in datasets:
        ds_dict = ds.to_dict()
        metadata = ds_dict.get("metadata", {})
        if metadata.get("is_geo"):
            ws = get_web_services(metadata.get("id", ""))
            if ws:
                ds_dict["web_services"] = ws
        result.append(ds_dict)
    return result


async def generate_events(
    question: str,
    datasets_service: DatasetService,
    llm_service: LLMService,
    embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
    search_mode: SearchMode = SearchMode.DENSE,
    city: str = None,
    state: str = None,
    country: str = None,
    year_from: int = None,
    year_to: int = None,
    use_multi_query: bool = True,
    use_llm_interpretation: bool = True,
    use_reranker: bool = False,
    reranker_candidates: int | None = None,
    is_geo: bool = False,
):
    step = 0
    try:
        # region 0. LLM QUESTIONS
        research_questions: list[LLMQuestion] | None = None

        if use_multi_query:
            # Multi-query RAG enabled: generate research questions via LLM
            if not IS_DOCKER:
                cached_answer = await check_qa_cache(question, str(step))
                if cached_answer is not None:
                    research_questions = [
                        LLMQuestion(cq["question"], reason=cq["reason"])
                        for cq in cached_answer
                    ]
            if research_questions is None:
                research_questions = await step_0_llm_questions(
                    step, question, llm_service
                )

            yield f"data: {
                json.dumps(
                    {
                        'step': step,
                        'status': 'OK',
                        'data': {
                            'question': question,
                            'research_questions': [
                                q.to_dict() for q in research_questions
                            ],
                        },
                    }
                )
            }\n\n"
            if not IS_DOCKER:
                await set_qa_cache(
                    question, str(step), [q.to_dict() for q in research_questions]
                )
        else:
            # Multi-query RAG disabled: use original question directly
            research_questions = [
                LLMQuestion(question=question, reason="Direct user query")
            ]
            yield f"data: {
                json.dumps(
                    {
                        'step': step,
                        'status': 'OK',
                        'data': {
                            'question': question,
                            'research_questions': [
                                q.to_dict() for q in research_questions
                            ],
                        },
                    }
                )
            }\n\n"

        step += 1
        # endregion

        # region 1. EMBEDDINGS
        embeddings = await step_1_embeddings(
            step, research_questions, embedder_model, search_mode
        )
        yield f"data: {
            json.dumps(
                {
                    'step': step,
                    'status': 'OK',
                    'data': [e.to_dict() for e in embeddings],
                }
            )
        }\n\n"
        step += 1
        # endregion

        # region 2. VECTOR SEARCH
        logger.info(
            f"step: {step}. VECTOR SEARCH start ({embedder_model.value}, mode={search_mode.value})"
        )
        start_2 = time.perf_counter()

        result_questions_with_datasets: list[LLMQuestionWithDatasets] = []
        for i, embedding in enumerate(embeddings):
            default_limit = 25
            search_limit = (
                reranker_candidates
                if use_reranker and reranker_candidates
                else default_limit
            )
            datasets = await datasets_service.search_datasets_with_vector(
                embedding.embeddings,
                search_mode=search_mode,
                embedder_model=embedder_model,
                city_filter=city,
                state_filter=state,
                country_filter=country,
                year_from=year_from,
                year_to=year_to,
                limit=search_limit,
            )
            if use_reranker and datasets:
                datasets = await datasets_service.rerank_results(
                    query=embedding.question,
                    datasets=datasets,
                    top_n=default_limit,
                )
            result_questions_with_datasets.append(
                LLMQuestionWithDatasets(
                    question=embedding.question,
                    reason=embedding.reason,
                    datasets=datasets,
                )
            )
            datasets_payload = (
                _enrich_with_web_services(datasets)
                if is_geo
                else [ds.to_dict() for ds in datasets]
            )
            yield f"data: {
                json.dumps(
                    {
                        'step': step,
                        'sub_step': i,
                        'status': 'OK',
                        'data': {
                            'question_hash': embedding.question_hash,
                            'datasets': datasets_payload,
                        },
                    }
                )
            }\n\n"
        elapsed_2 = time.perf_counter() - start_2
        logger.info(f"step: {step}. VECTOR SEARCH end (elapsed: {elapsed_2:.2f}s)")
        step += 1
        # endregion

        # region 3. INTERPRETATION
        if use_llm_interpretation:
            logger.info(f"step: {step}. INTERPRETATION start")
            start_3 = time.perf_counter()

            for i, q in enumerate(result_questions_with_datasets):
                start_3_i = time.perf_counter()
                logger.info(f"step: {step}.{str(i)} INTERPRETATION STEP start")
                answer: list[str] | None = None
                if not IS_DOCKER:
                    cached_answer = await check_qa_cache(question, f"{step}_{i}")
                    if cached_answer is not None:
                        answer = cached_answer
                if answer is None:
                    answer = await llm_service.answer_research_question(q)

                logger.info(answer)
                yield f"data: {
                    json.dumps(
                        {
                            'step': step,
                            'sub_step': i,
                            'status': 'OK',
                            'data': {
                                'question_hash': q.question_hash,
                                'answer': answer,
                            },
                        }
                    )
                }\n\n"
                if not IS_DOCKER:
                    await set_qa_cache(question, f"{step}_{i}", answer)
                elapsed_3_i = time.perf_counter() - start_3_i
                logger.info(
                    f"step: {step}.{str(i)} INTERPRETATION STEP end (elapsed: {elapsed_3_i:.2f}s)"
                )
            elapsed_3 = time.perf_counter() - start_3
            logger.info(f"step: {step}. INTERPRETATION end (elapsed: {elapsed_3:.2f}s)")
            step += 1
        else:
            logger.info(f"step: {step}. INTERPRETATION skipped (disabled by user)")
        # endregion

        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error("ERROR!", exc_info=True)
        yield f"data: {json.dumps({'step': step, 'status': 'error', 'error': str(e)})}\n\n"


@router.get("/qa")
async def stream(
    question: str = Query(
        "Is there any e-cars in Chemnitz?", description="Ask the system"
    ),
    embedder_model: EmbedderModel = Query(
        DEFAULT_EMBEDDER_MODEL, description="Embedder model to use"
    ),
    search_mode: SearchMode = Query(
        SearchMode.DENSE,
        description="Search mode: dense, sparse, or hybrid (RRF fusion)",
    ),
    city: str = Query(None, description="Filter by city"),
    state: str = Query(None, description="Filter by state/region"),
    country: str = Query(None, description="Filter by country"),
    year_from: int = Query(
        None, description="Filter datasets created from this year (inclusive)"
    ),
    year_to: int = Query(
        None, description="Filter datasets created until this year (inclusive)"
    ),
    use_multi_query: bool = Query(
        True, description="Enable multi-query RAG (generate research questions)"
    ),
    use_llm_interpretation: bool = Query(
        True, description="Enable LLM interpretation of results"
    ),
    use_reranker: bool = Query(
        False, description="Rerank results using cross-encoder reranker"
    ),
    reranker_candidates: int = Query(
        None,
        ge=10,
        le=200,
        description="How many candidates to fetch before reranking",
    ),
    is_geo: bool = Query(
        True, description="Include web_services data for geo datasets in results"
    ),
    datasets_service: DatasetService = Depends(get_dataset_service),
    llm_service: LLMService = Depends(get_llm_service_dep),
):
    return StreamingResponse(
        generate_events(
            question,
            datasets_service,
            llm_service,
            embedder_model,
            search_mode,
            city,
            state,
            country,
            year_from,
            year_to,
            use_multi_query,
            use_llm_interpretation,
            use_reranker,
            reranker_candidates,
            is_geo,
        ),
        media_type="text/event-stream",
    )


# endregion
