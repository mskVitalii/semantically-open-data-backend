import json
import time

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.responses import StreamingResponse

from .datasets_dto import DatasetSearchRequest, DatasetSearchResponse
from .qa_cache.qa_cache import check_qa_cache, set_qa_cache
from ..domain.services.dataset_service import DatasetService, get_dataset_service
from ..domain.services.llm_dto import (
    LLMQuestion,
    LLMQuestionWithEmbeddings,
    LLMQuestionWithDatasets,
)
from ..domain.services.llm_service import (
    LLMService,
    get_llm_service_dep,
)
from ..infrastructure.config import IS_DOCKER
from ..infrastructure.logger import get_prefixed_logger
from ..vector_search.embedder import embed_batch_with_ids

logger = get_prefixed_logger("API /datasets")

router = APIRouter(prefix="/datasets", tags=["datasets"])


# region Search Datasets
@router.post("/search_datasets", response_model=DatasetSearchResponse)
async def search_datasets(
    request: DatasetSearchRequest,
    service: DatasetService = Depends(get_dataset_service),
) -> DatasetSearchResponse:
    """
    Dataset search

    Supported parameters:
    - query: full-text search by name and description
    - tags: filter by tags
    - city: filter by city
    - state: filter by state/region
    - country: filter by country
    - limit: number of results (1–100)
    - offset: pagination offset
    """
    try:
        return await service.search_datasets(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# endregion


# region Bootstrap
@router.post("/bootstrap")
async def bootstrap(
    use_fs_cache: bool = Query(True, description="Use filesystem cache for datasets"),
    clear_store: bool = Query(True, description="Clear MongoDB store"),
    clear_vector_db: bool = Query(True, description="Clear vector database"),
    service: DatasetService = Depends(get_dataset_service),
):
    """
    Bootstrap datasets - clear and reload all data

    Parameters:
    - clear_store: Clear MongoDB collections
    - clear_vector_db: Clear vector database
    - use_fs_cache: If True, use existing files from filesystem cache. If False, clear and redownload all data.
    """
    try:
        result = await service.bootstrap_datasets(
            use_fs_cache=use_fs_cache,
            clear_store=clear_store,
            clear_vector_db=clear_vector_db,
        )
        return {"ok": result}
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
    step: int, research_questions: list[LLMQuestion]
) -> list[LLMQuestionWithEmbeddings]:
    logger.info(f"step: {step}. EMBEDDINGS start")
    start_1 = time.perf_counter()

    questions_list = [
        {"text": q.question, "id": q.question_hash} for q in research_questions
    ]
    embeddings = await embed_batch_with_ids(questions_list)

    embeddings_map: dict[str, np.ndarray] = {
        str(e["id"]): e["embedding"] for e in embeddings
    }

    questions_with_embeddings: list[LLMQuestionWithEmbeddings] = [
        LLMQuestionWithEmbeddings(
            question=q.question,
            reason=q.reason,
            embeddings=embeddings_map.get(q.question_hash),
        )
        for q in research_questions
    ]

    elapsed_1 = time.perf_counter() - start_1
    logger.info(f"step: {step}. EMBEDDINGS end (elapsed: {elapsed_1:.2f}s)")
    return questions_with_embeddings


async def generate_events(
    question: str,
    datasets_service: DatasetService,
    llm_service: LLMService,
    city: str = None,
    state: str = None,
    country: str = None,
    use_multi_query: bool = True,
    use_llm_interpretation: bool = True,
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
                research_questions = await step_0_llm_questions(step, question, llm_service)

            yield f"data: {
                json.dumps(
                    {
                        'step': step,
                        'status': 'OK',
                        'data': {
                            'question': question,
                            'research_questions': [q.to_dict() for q in research_questions],
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
                            'research_questions': [q.to_dict() for q in research_questions],
                        },
                    }
                )
            }\n\n"

        step += 1
        # endregion

        # mb: make class Question & correct types
        # mb, return many mini-steps for each query
        # mb I should provide IDs within the full system to keep the order & do not mix questions embeddings

        # region 1. EMBEDDINGS
        embeddings = await step_1_embeddings(step, research_questions)
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
        logger.info(f"step: {step}. VECTOR SEARCH start")
        start_2 = time.perf_counter()

        result_questions_with_datasets: list[LLMQuestionWithDatasets] = []
        for i, embedding in enumerate(embeddings):
            datasets = await datasets_service.search_datasets_with_embeddings(
                embedding.embeddings,
                city_filter=city,
                state_filter=state,
                country_filter=country,
            )
            result_questions_with_datasets.append(
                LLMQuestionWithDatasets(
                    question=embedding.question,
                    reason=embedding.reason,
                    datasets=datasets,
                )
            )
            yield f"data: {
                json.dumps(
                    {
                        'step': step,
                        'sub_step': i,
                        'status': 'OK',
                        'data': {
                            'question_hash': embedding.question_hash,
                            'datasets': [ds.to_dict() for ds in datasets],
                        },
                    }
                )
            }\n\n"
        # logger.info(result_questions_with_datasets)
        elapsed_2 = time.perf_counter() - start_2
        logger.info(f"step: {step}. VECTOR SEARCH end (elapsed: {elapsed_2:.2f}s)")
        step += 1
        # endregion

        # mb choose fields to reduce context and improve results

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
        "What is the color of grass in Germany?", description="Ask the system"
    ),
    city: str = Query(None, description="Filter by city"),
    state: str = Query(None, description="Filter by state/region"),
    country: str = Query(None, description="Filter by country"),
    use_multi_query: bool = Query(
        True, description="Enable multi-query RAG (generate research questions)"
    ),
    use_llm_interpretation: bool = Query(
        True, description="Enable LLM interpretation of results"
    ),
    datasets_service: DatasetService = Depends(get_dataset_service),
    llm_service: LLMService = Depends(get_llm_service_dep),
):
    return StreamingResponse(
        generate_events(
            question,
            datasets_service,
            llm_service,
            city,
            state,
            country,
            use_multi_query,
            use_llm_interpretation,
        ),
        media_type="text/event-stream",
    )


# endregion
