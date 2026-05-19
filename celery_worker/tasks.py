from celery_worker.celery_app import celery_app
from crews.analytics_crew import run_analytics_crew
from crews.bizdev_crew import run_bizdev_crew
from crews.content_crew import run_content_crew
from crews.marketing_crew import run_marketing_crew
from crews.sales_improvement_crew import run_sales_improvement_crew
from crews.scheduler_crew import run_scheduler_crew
from crews.support_crew import run_support_crew
from database import SessionLocal
from job_store import PostgresJobStore
from models import JobStatus
from runtime_config import apply_runtime_environment, load_runtime_config
from utils.retry_policy import is_retryable_exception, retry_countdown_seconds
from utils.usage_tracking import build_usage_summary, monotonic_time, pop_usage_metrics


job_store = PostgresJobStore(SessionLocal)


def _run_with_job_state(
    self: object,
    progress: str,
    crew_function: object,
    inputs: dict,
    config_context: dict | None = None,
) -> dict:
    job_id = self.request.id
    config_context = config_context or load_runtime_config().as_context()
    apply_runtime_environment(config_context)
    self.update_state(state="PROGRESS", meta={"status": progress})
    job_store.update_job(job_id, status=JobStatus.RUNNING, result={"status": progress}, error=None)
    started_at = monotonic_time()
    try:
        result = crew_function(inputs, config_context)
        clean_result, usage_metrics = pop_usage_metrics(result)
        normalized_result = clean_result if isinstance(clean_result, dict) else {"raw": str(clean_result)}
        usage_summary = build_usage_summary(
            usage_metrics,
            monotonic_time() - started_at,
            config_context,
        )
        job_store.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            result=normalized_result,
            **usage_summary,
            error=None,
        )
        return normalized_result
    except Exception as exc:
        request = getattr(self, "request", None)
        retries = int(getattr(request, "retries", 0) or 0)
        max_retries = int(getattr(self, "max_retries", 3) or 3)

        if is_retryable_exception(exc) and retries < max_retries:
            countdown = retry_countdown_seconds(retries)
            retry_meta = {
                "status": "Retrying after transient provider or network error",
                "retry_count": retries + 1,
                "max_retries": max_retries,
                "next_retry_seconds": countdown,
                "error": str(exc),
            }
            job_store.update_job(
                job_id,
                status=JobStatus.RUNNING,
                result=retry_meta,
                error=str(exc),
                duration_seconds=monotonic_time() - started_at,
            )
            raise self.retry(exc=exc, countdown=countdown)

        job_store.update_job(
            job_id,
            status=JobStatus.FAILED,
            result=None,
            error=str(exc),
            duration_seconds=monotonic_time() - started_at,
        )
        raise


@celery_app.task(name="health_check")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@celery_app.task(bind=True, name="workflow.marketing", soft_time_limit=1200, time_limit=1500)
def run_marketing_task(self: object, inputs: dict, config_context: dict | None = None) -> dict:
    return _run_with_job_state(self, "Initializing Marketing Crew...", run_marketing_crew, inputs, config_context)


@celery_app.task(bind=True, name="workflow.content", soft_time_limit=1200, time_limit=1500)
def run_content_task(self: object, inputs: dict, config_context: dict | None = None) -> dict:
    return _run_with_job_state(self, "Generating localized content...", run_content_crew, inputs, config_context)


@celery_app.task(bind=True, name="workflow.support", soft_time_limit=900, time_limit=1200)
def run_support_task(self: object, inputs: dict, config_context: dict | None = None) -> dict:
    return _run_with_job_state(self, "Drafting and QA support response...", run_support_crew, inputs, config_context)


@celery_app.task(bind=True, name="workflow.analytics", soft_time_limit=1500, time_limit=1800)
def run_analytics_task(self: object, inputs: dict, config_context: dict | None = None) -> dict:
    return _run_with_job_state(
        self,
        "Aggregating platform metrics and benchmarking...",
        run_analytics_crew,
        inputs,
        config_context,
    )


@celery_app.task(bind=True, name="workflow.sales_improvement", soft_time_limit=1200, time_limit=1500)
def run_sales_improvement_task(self: object, inputs: dict, config_context: dict | None = None) -> dict:
    return _run_with_job_state(
        self,
        "Analyzing funnel and generating CRO playbook...",
        run_sales_improvement_crew,
        inputs,
        config_context,
    )


@celery_app.task(bind=True, name="workflow.bizdev", soft_time_limit=1200, time_limit=1500)
def run_bizdev_task(self: object, inputs: dict, config_context: dict | None = None) -> dict:
    return _run_with_job_state(
        self,
        "Prospecting leads and drafting outreach...",
        run_bizdev_crew,
        inputs,
        config_context,
    )


@celery_app.task(bind=True, name="workflow.scheduler", soft_time_limit=900, time_limit=1200)
def run_scheduler_task(self: object, inputs: dict, config_context: dict | None = None) -> dict:
    return _run_with_job_state(
        self,
        "Mapping timezones and resolving conflicts...",
        run_scheduler_crew,
        inputs,
        config_context,
    )
