from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..infrastructure.config import EmbedderModel, DEFAULT_EMBEDDER_MODEL
from ..infrastructure.logger import get_prefixed_logger
from ..vector_search.embedder import embed_tokens, embed_tokens_tsv

logger = get_prefixed_logger("API /embeddings")

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


class EmbedTokensRequest(BaseModel):
    text: str
    embedder_model: EmbedderModel = Field(default=DEFAULT_EMBEDDER_MODEL)


@router.post("/embed_tokens")
async def embed_tokens_endpoint(request: EmbedTokensRequest):
    """
    Tokenize text, embed each token, PCA → 3D.

    Proxies the request to the specified embedder instance.
    Dimension is resolved automatically from the embedder model.
    Returns tokens and their 3D coordinates for scatter/word cloud visualization.
    """
    result = await embed_tokens(
        text=request.text,
        embedder_model=request.embedder_model,
    )
    return {
        "embedder": request.embedder_model.value,
        **result,
    }


@router.post("/embed_tokens_tsv")
async def embed_tokens_tsv_endpoint(request: EmbedTokensRequest):
    """
    Tokenize text, embed each token → TSV format.

    Proxies the request to the specified embedder instance.
    Dimension is resolved automatically from the embedder model.
    Returns vectors_tsv and metadata_tsv for TensorFlow Projector.
    """
    result = await embed_tokens_tsv(
        text=request.text,
        embedder_model=request.embedder_model,
    )
    return {
        "embedder": request.embedder_model.value,
        **result,
    }
