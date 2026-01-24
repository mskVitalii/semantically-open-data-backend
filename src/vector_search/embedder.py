import httpx
import numpy as np

from src.infrastructure.config import (
    EmbedderModel,
    DEFAULT_EMBEDDER_MODEL,
    get_embedder_url,
)
from src.infrastructure.logger import get_prefixed_logger

logger = get_prefixed_logger(__name__, "EMBEDDER")


async def embed(
    text: str, embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL
) -> np.ndarray:
    """Embed single text using specified embedder model"""
    res = await embed_batch([text], embedder_model=embedder_model)
    return res[0]


async def embed_batch(
    texts: list[str], embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL
) -> list[np.ndarray]:
    """Generate embeddings for multiple texts using specified embedder model"""
    embedder_url = get_embedder_url(embedder_model)
    logger.info(
        f"working with texts {len(texts)} using embedder {embedder_model.value}"
    )
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=embedder_url + "/embed",
            json={"texts": texts},
            timeout=len(texts) * 100,
        )
    response.raise_for_status()
    data = response.json()["embeddings"]
    logger.info(f"✅ embeddings done! {len(data)} ({embedder_model.value})")
    return [np.array(vec, dtype=np.float32) for vec in data]


async def embed_batch_with_ids(
    texts: list[dict[str, str]],
    embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
) -> list[dict[str, np.ndarray]]:
    """Embed texts while preserving IDs using specified embedder model"""
    embedder_url = get_embedder_url(embedder_model)
    logger.info(
        f"working with texts {len(texts)} using embedder {embedder_model.value}"
    )
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=embedder_url + "/embed_with_ids",
            json={"texts": texts},
            timeout=len(texts) * 100,
        )
    response.raise_for_status()
    data = response.json()["embeddings"]
    logger.info(f"✅ embeddings done! {len(data)} ({embedder_model.value})")
    return [
        {"id": vec["id"], "embedding": np.array(vec["embedding"], dtype=np.float32)}
        for vec in data
    ]
