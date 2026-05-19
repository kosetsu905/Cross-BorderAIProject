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
    try:
        result = crew_function(inputs, config_context)
        normalized_result = result if isinstance(result, dict) else {"raw": str(result)}
        job_store.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            result=normalized_result,
            error=None,
        )
        return normalized_result
    except Exception as exc:
        job_store.update_job(job_id, status=JobStatus.FAILED, result=None, error=str(exc))
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
