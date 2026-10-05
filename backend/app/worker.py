"""Celery app definition."""
from celery import Celery
from celery.signals import setup_logging

from app.config import get_settings
from app.logging_utils import configure_logging

settings = get_settings()


@setup_logging.connect
def _configure_worker_logging(**_kwargs):
    # Connecting a receiver stops Celery from installing its own root-logger
    # config, so worker output uses the same format (with per-run ids) as the API.
    configure_logging()

celery_app = Celery(
    "rabbit_hole",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
)
