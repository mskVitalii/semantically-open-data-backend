import asyncio

import httpx

from src.infrastructure.config import RERANKER_URL
from src.infrastructure.logger import get_prefixed_logger

logger = get_prefixed_logger(__name__, "RERANKER")

# Reranker processes requests sequentially — high concurrency only increases queue wait time
_semaphore = asyncio.Semaphore(1)

MAX_RETRIES = 2
RETRY_BASE_DELAY = 1.0


async def rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
    timeout: float = 180.0,
) -> list[dict]:
    """
    Rerank documents by relevance to the query.

    Returns list of {"index": int, "document": str, "relevance_score": float}
    sorted by relevance_score descending.
    """
    if not documents:
        return []

    payload = {"query": query, "documents": documents}
    if top_n is not None:
        payload["top_n"] = top_n

    logger.info(f"reranking {len(documents)} documents (top_n={top_n})")

    async with _semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{RERANKER_URL}/rerank",
                        json=payload,
                        timeout=timeout,
                    )
                response.raise_for_status()

                results = response.json()["results"]
                logger.info(f"reranking done, got {len(results)} results")
                return results
            except (httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        f"reranker attempt {attempt}/{MAX_RETRIES} failed ({type(e).__name__}), "
                        f"retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"reranker failed after {MAX_RETRIES} attempts")
                    raise
