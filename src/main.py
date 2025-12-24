import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette import status
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from src.datasets_api.router import v1_router
from src.domain.services.llm_service import get_llm_service
from src.infrastructure.config import MONGO_INITDB_DATABASE
from src.infrastructure.logger import get_logger
from src.infrastructure.mongo_db import (
    get_mongodb_manager,
    MongoClientDep,
)

logger = get_logger(__name__)

# region FAST_API


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting application...")

    manager = get_mongodb_manager()
    await manager.connect()
    llm_service = get_llm_service()

    yield

    await llm_service.close_llm_session()
    await manager.disconnect()
    logger.info("Application shutdown complete")


app = FastAPI(
    title="Semantic Open Data API",
    description="API to semantically search datasets. Responses to the questions",
    version="1.0.0",
    lifespan=lifespan,
)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://0.0.0.0:5173",
    "http://0.0.0.0:80",
    "http://127.0.0.1:80",
    "http://localhost:80",
    "http://0.0.0.0",
    "http://127.0.0.1",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router)

# endregion

# region DEFAULT ROUTES


@app.get("/")
async def root():
    return {"message": "Semantic Open Data API is running"}


@app.get("/health")
async def health_check():
    try:
        manager = get_mongodb_manager()
        is_healthy = await manager.ping()

        if is_healthy:
            return {
                "status": "healthy",
                "database": "connected",
                "database_name": MONGO_INITDB_DATABASE,
            }
        else:
            raise Exception("MongoDB ping failed")

    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "database": "error", "error": str(e)},
        )


@app.get("/health/detailed")
async def detailed_health_check(client: MongoClientDep):
    """Детальная проверка здоровья с метриками"""
    health_status = {"status": "healthy", "checks": {}}

    try:
        manager = get_mongodb_manager()

        is_connected = await manager.ping()
        if not is_connected:
            raise Exception("MongoDB is not responding")

        if not manager.is_testing:
            server_info = await client.server_info()
            version = server_info.get("version")
        else:
            version = "mock"

        db_stats = await manager.get_database_stats()

        health_status["checks"]["mongodb"] = {
            "status": "up",
            "version": version,
            "database": MONGO_INITDB_DATABASE,
            "collections": db_stats.get("collections"),
            "objects": db_stats.get("objects"),
            "dataSize": db_stats.get("dataSize"),
        }

    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["mongodb"] = {"status": "down", "error": str(e)}
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=health_status
        )

    return health_status


# endregion

# region IGNITE


def main():
    """Main MVP function"""
    run_dev()


def run_dev():
    # Disable noisy third-party library logs
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.DEBUG)

    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


def run_start():
    # Disable noisy third-party library logs
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)

    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
# run_dev()
# logger.info("http://localhost:8000/docs")
# endregion
