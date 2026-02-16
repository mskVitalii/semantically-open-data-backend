from fastapi import APIRouter

from src.datasets_api.datasets_controller import router as datasets_router
from src.datasets_api.embeddings_controller import router as embeddings_router
from src.testing.testing_controller import router as testing_router

v1_router = APIRouter(prefix="/v1")
v1_router.include_router(datasets_router)
v1_router.include_router(embeddings_router)
v1_router.include_router(testing_router)
