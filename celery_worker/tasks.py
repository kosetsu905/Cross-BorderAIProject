from celery_worker.celery_app import celery_app


@celery_app.task(name="health_check")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}
