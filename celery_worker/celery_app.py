import os
import logging

from celery import Celery
from celery.signals import task_failure, task_postrun, task_prerun
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

celery_app = Celery(
    "cross_border_ai_suite",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
    include=["celery_worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_default_retry_delay=30,
    task_max_retries=3,
    worker_max_tasks_per_child=500,
    result_expires=60 * 60 * 24,
)


@task_prerun.connect
def task_prerun_handler(task_id: str, task: object, *args: object, **kwargs: object) -> None:
    logger.info("Task %s [%s] started", getattr(task, "name", "<unknown>"), task_id)


@task_postrun.connect
def task_postrun_handler(task_id: str, task: object, *args: object, **kwargs: object) -> None:
    logger.info("Task %s [%s] completed", getattr(task, "name", "<unknown>"), task_id)


@task_failure.connect
def task_failure_handler(task_id: str, exception: Exception, *args: object, **kwargs: object) -> None:
    logger.error("Task %s failed: %s", task_id, exception)
