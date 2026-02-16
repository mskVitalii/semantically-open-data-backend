from dataclasses import dataclass

import httpx
import numpy as np

from src.infrastructure.config import (
    EmbedderModel,
    DEFAULT_EMBEDDER_MODEL,
    SearchMode,
    DEFAULT_SPARSE_DIM,
    DEFAULT_SPARSE_MODE,
    get_embedder_url,
    get_embedding_dim,
)
from src.infrastructure.logger import get_prefixed_logger

logger = get_prefixed_logger(__name__, "EMBEDDER")


@dataclass
class SparseVector:
    """Sparse vector with indices and values (maps to Qdrant SparseVector)"""
    indices: list[int]
    values: list[float]


@dataclass
class HybridEmbedding:
    """Dense + sparse embedding pair"""
    dense: np.ndarray
    sparse: SparseVector


# ---------------------------------------------------------------------------
# Dense-only helpers (backward-compatible)
# ---------------------------------------------------------------------------

async def embed(
    text: str, embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL
) -> np.ndarray:
    """Embed single text using specified embedder model (dense only)"""
    res = await embed_batch([text], embedder_model=embedder_model)
    return res[0]


async def embed_batch(
    texts: list[str], embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL
) -> list[np.ndarray]:
    """Generate dense embeddings for multiple texts"""
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
    """Embed texts while preserving IDs (dense only)"""
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


# ---------------------------------------------------------------------------
# Sparse helpers
# ---------------------------------------------------------------------------

def _parse_sparse(raw: dict) -> SparseVector:
    return SparseVector(indices=raw["indices"], values=raw["values"])


async def embed_batch_sparse(
    texts: list[str],
    embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
    sparse_dim: int = DEFAULT_SPARSE_DIM,
    sparse_mode: str = DEFAULT_SPARSE_MODE,
) -> list[SparseVector]:
    """Generate sparse embeddings for multiple texts"""
    embedder_url = get_embedder_url(embedder_model)
    logger.info(
        f"sparse embed {len(texts)} texts via {embedder_model.value}"
    )
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=embedder_url + "/embed",
            json={
                "texts": texts,
                "mode": "sparse",
                "sparse_dim": sparse_dim,
                "sparse_mode": sparse_mode,
            },
            timeout=len(texts) * 100,
        )
    response.raise_for_status()
    data = response.json()["embeddings"]
    logger.info(f"✅ sparse embeddings done! {len(data)}")
    return [_parse_sparse(item) for item in data]


# ---------------------------------------------------------------------------
# Hybrid helpers
# ---------------------------------------------------------------------------

async def embed_batch_hybrid(
    texts: list[str],
    embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
    sparse_dim: int = DEFAULT_SPARSE_DIM,
    sparse_mode: str = DEFAULT_SPARSE_MODE,
) -> list[HybridEmbedding]:
    """Generate dense + sparse embeddings in a single call"""
    embedder_url = get_embedder_url(embedder_model)
    logger.info(
        f"hybrid embed {len(texts)} texts via {embedder_model.value}"
    )
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=embedder_url + "/embed",
            json={
                "texts": texts,
                "mode": "hybrid",
                "sparse_dim": sparse_dim,
                "sparse_mode": sparse_mode,
            },
            timeout=len(texts) * 100,
        )
    response.raise_for_status()
    data = response.json()["embeddings"]
    logger.info(f"✅ hybrid embeddings done! {len(data)}")
    return [
        HybridEmbedding(
            dense=np.array(item["dense"], dtype=np.float32),
            sparse=_parse_sparse(item["sparse"]),
        )
        for item in data
    ]


# ---------------------------------------------------------------------------
# Unified entry points (mode-aware)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Token-level embeddings (PCA → 3D / TSV)
# ---------------------------------------------------------------------------

async def embed_tokens(
    text: str,
    embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
) -> dict:
    """Tokenize text, embed each token, PCA → 3D via a specific embedder."""
    embedder_url = get_embedder_url(embedder_model)
    dimension = get_embedding_dim(embedder_model)
    logger.info(f"embed_tokens via {embedder_model.value} (dim={dimension})")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=embedder_url + "/embed_tokens",
            json={"text": text, "dimension": dimension},
            timeout=120.0,
        )
    response.raise_for_status()
    return response.json()


async def embed_tokens_tsv(
    text: str,
    embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
) -> dict:
    """Same as embed_tokens but returns TSV format for TensorFlow Projector."""
    embedder_url = get_embedder_url(embedder_model)
    dimension = get_embedding_dim(embedder_model)
    logger.info(f"embed_tokens_tsv via {embedder_model.value} (dim={dimension})")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url=embedder_url + "/embed_tokens_tsv",
            json={"text": text, "dimension": dimension},
            timeout=120.0,
        )
    response.raise_for_status()
    return response.json()


async def embed_single(
    text: str,
    embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
    mode: SearchMode = SearchMode.DENSE,
    sparse_dim: int = DEFAULT_SPARSE_DIM,
    sparse_mode: str = DEFAULT_SPARSE_MODE,
) -> np.ndarray | SparseVector | HybridEmbedding:
    """Embed a single text in the requested mode"""
    if mode == SearchMode.DENSE:
        return await embed(text, embedder_model)
    elif mode == SearchMode.SPARSE:
        res = await embed_batch_sparse(
            [text], embedder_model, sparse_dim, sparse_mode,
        )
        return res[0]
    else:
        res = await embed_batch_hybrid(
            [text], embedder_model, sparse_dim, sparse_mode,
        )
        return res[0]


async def embed_multi(
    texts: list[str],
    embedder_model: EmbedderModel = DEFAULT_EMBEDDER_MODEL,
    mode: SearchMode = SearchMode.DENSE,
    sparse_dim: int = DEFAULT_SPARSE_DIM,
    sparse_mode: str = DEFAULT_SPARSE_MODE,
) -> list[np.ndarray] | list[SparseVector] | list[HybridEmbedding]:
    """Embed multiple texts in the requested mode"""
    if mode == SearchMode.DENSE:
        return await embed_batch(texts, embedder_model)
    elif mode == SearchMode.SPARSE:
        return await embed_batch_sparse(
            texts, embedder_model, sparse_dim, sparse_mode,
        )
    else:
        return await embed_batch_hybrid(
            texts, embedder_model, sparse_dim, sparse_mode,
        )
