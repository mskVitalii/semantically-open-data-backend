import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()
# Get all the env variables & optional validation

ENV = os.getenv("ENV", "development")
IS_DOCKER = os.getenv("IS_DOCKER", "false") == "true"
USE_GRPC = os.getenv("USE_GRPC", "true").lower() == "true"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_HTTP_PORT = int(os.getenv("QDRANT_HTTP_PORT", 6333))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", 6334))
QDRANT_COLLECTION_PREFIX = "datasets_metadata"

EMBEDDING_DIM = 1024  # Deprecated: use get_embedding_dim() instead


class EmbedderModel(str, Enum):
    """Available embedder models"""

    BAAI_BGE_M3 = "baai-bge-m3"
    INTFLOAT_MULTILINGUAL_E5_BASE = "intfloat-multilingual-e5-base"
    JINAAI_JINA_EMBEDDINGS_V3 = "jinaai-jina-embeddings-v3"
    SENTENCE_TRANSFORMERS_LABSE = "sentence-transformers-labse"


class SearchMode(str, Enum):
    """Vectorization / search mode"""

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


# Sparse vectorization defaults
DEFAULT_SPARSE_DIM = 1_048_576
DEFAULT_SPARSE_MODE = "tf"  # "binary" | "tf"


# Embedding dimensions for each model
EMBEDDING_DIMENSIONS = {
    EmbedderModel.BAAI_BGE_M3: 1024,
    EmbedderModel.INTFLOAT_MULTILINGUAL_E5_BASE: 768,
    EmbedderModel.JINAAI_JINA_EMBEDDINGS_V3: 1024,
    EmbedderModel.SENTENCE_TRANSFORMERS_LABSE: 768,
}


# Embedder configuration
EMBEDDER_HOST = os.getenv("EMBEDDER_HOST", "localhost")
EMBEDDER_BASE_PORT = 8080

# Default embedder (can be overridden)
DEFAULT_EMBEDDER_MODEL = EmbedderModel(
    os.getenv("DEFAULT_EMBEDDER_MODEL", EmbedderModel.JINAAI_JINA_EMBEDDINGS_V3.value)
)

# Embedder URLs mapping (each embedder runs on a different port)
EMBEDDER_URLS = {
    EmbedderModel.BAAI_BGE_M3: f"http://{EMBEDDER_HOST}:{EMBEDDER_BASE_PORT}",
    EmbedderModel.INTFLOAT_MULTILINGUAL_E5_BASE: f"http://{EMBEDDER_HOST}:{EMBEDDER_BASE_PORT + 1}",
    EmbedderModel.JINAAI_JINA_EMBEDDINGS_V3: f"http://{EMBEDDER_HOST}:{EMBEDDER_BASE_PORT + 2}",
    EmbedderModel.SENTENCE_TRANSFORMERS_LABSE: f"http://{EMBEDDER_HOST}:{EMBEDDER_BASE_PORT + 3}",
}

# Backwards compatibility
EMBEDDER_PORT = int(os.getenv("EMBEDDER_PORT", EMBEDDER_BASE_PORT + 2))
EMBEDDER_URL = f"http://{EMBEDDER_HOST}:{EMBEDDER_PORT}"


def get_embedder_url(model: EmbedderModel) -> str:
    """Get embedder URL for specific model"""
    return EMBEDDER_URLS[model]


def get_collection_name(embedder_model: EmbedderModel) -> str:
    """Get collection name for specific embedder model.

    All collections are hybrid (named dense + sparse vectors).
    """
    return f"{QDRANT_COLLECTION_PREFIX}_{embedder_model.value}"


def get_embedding_dim(embedder_model: EmbedderModel) -> int:
    """Get embedding dimension for specific embedder model"""
    return EMBEDDING_DIMENSIONS[embedder_model]


MONGO_INITDB_DATABASE = os.getenv("MONGO_INITDB_DATABASE", "db")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))
MONGO_USER = os.getenv("MONGO_USER", "appuser")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "")
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@mongodb:{MONGO_PORT}/{MONGO_INITDB_DATABASE}?authSource={MONGO_INITDB_DATABASE}",
)

RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8084")

LLM_HOST = os.getenv("LLM_HOST", "localhost")
LLM_PORT = int(os.getenv("LLM_PORT", 11434))
LLM_URL = f"http://{LLM_HOST}:{LLM_PORT}"
LLM_OPEN_AI_KEY = os.getenv("LLM_OPEN_AI_KEY", "")

os.environ["GRPC_VERBOSITY"] = "NONE"
