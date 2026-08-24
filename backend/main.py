import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

try:
    from config import settings
    from db.health import check_mlflow, check_postgres, check_redis
    from db.migrations import check_and_apply_migrations
    from db.pool import close_pool, init_pool
    from api.ingest import router as ingest_router
except ImportError:
    from backend.config import settings
    from backend.db.health import check_mlflow, check_postgres, check_redis
    from backend.db.migrations import check_and_apply_migrations
    from backend.db.pool import close_pool, init_pool
    from backend.api.ingest import router as ingest_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neuroflow-api")

# Define Prometheus Metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP Requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP Request Duration", ["endpoint"]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan context manager for startup and shutdown event management.
    """
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} ({settings.ENVIRONMENT})...")
    try:
        await init_pool()
        await check_and_apply_migrations()
    except Exception as err:
        logger.error(f"Error during lifespan startup: {err}")
    yield
    logger.info("Shutting down application...")
    await close_pool()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Include Ingestion Router
app.include_router(ingest_router)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check(response: Response):
    """
    Health check endpoint verifying PostgreSQL, Redis, and MLflow connectivity.
    """
    pg_ok = await check_postgres()
    redis_ok = await check_redis()
    mlflow_ok = await check_mlflow()

    all_healthy = pg_ok and redis_ok and mlflow_ok
    if not all_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if all_healthy else "degraded",
        "checks": {
            "postgres": pg_ok,
            "redis": redis_ok,
            "mlflow": mlflow_ok,
        },
    }


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics scraping endpoint.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.APP_NAME} API", "version": settings.APP_VERSION}
