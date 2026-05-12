from celery import Celery

celery_app = Celery(
    "cross_border_ai_suite",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)
