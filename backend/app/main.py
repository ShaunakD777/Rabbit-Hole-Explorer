import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db.session import init_db
from app.api.v1 import router as api_router
from app.api.ws import router as ws_router, run_subscriber as run_ws_subscriber

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initialising DB connection pool...")
    await init_db()
    # Cross-process WebSocket bridge: relays broadcast() calls from any process
    # (Celery worker included) to this process's local WebSocket clients. See ws.py.
    subscriber_task = asyncio.create_task(run_ws_subscriber())
    yield
    logger.info("Shutting down.")
    subscriber_task.cancel()
    try:
        await subscriber_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="AI Rabbit-Hole Explorer API",
    version="1.0.0",
    description="NLP-powered knowledge graph explorer",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")
app.include_router(ws_router, prefix="/ws")


@app.get("/health")
async def health():
    return {"status": "ok", "environment": settings.environment}
