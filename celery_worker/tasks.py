from celery_worker.celery_app import celery_app
from crews.analytics_crew import run_analytics_crew
from crews.bizdev_crew import run_bizdev_crew
from crews.content_crew import run_content_crew
from crews.marketing_crew import run_marketing_crew
from crews.sales_improvement_crew import run_sales_improvement_crew
from crews.scheduler_crew import run_scheduler_crew
from crews.support_crew import run_support_crew


@celery_app.task(name="health_check")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@celery_app.task(bind=True, name="workflow.marketing", soft_time_limit=1200, time_limit=1500)
def run_marketing_task(self: object, inputs: dict) -> dict:
    self.update_state(state="PROGRESS", meta={"status": "Initializing Marketing Crew..."})
    return run_marketing_crew(inputs)


@celery_app.task(bind=True, name="workflow.content", soft_time_limit=1200, time_limit=1500)
def run_content_task(self: object, inputs: dict) -> dict:
    self.update_state(state="PROGRESS", meta={"status": "Generating localized content..."})
    return run_content_crew(inputs)


@celery_app.task(bind=True, name="workflow.support", soft_time_limit=900, time_limit=1200)
def run_support_task(self: object, inputs: dict) -> dict:
    self.update_state(state="PROGRESS", meta={"status": "Drafting and QA support response..."})
    return run_support_crew(inputs)


@celery_app.task(bind=True, name="workflow.analytics", soft_time_limit=1500, time_limit=1800)
def run_analytics_task(self: object, inputs: dict) -> dict:
    self.update_state(
        state="PROGRESS",
        meta={"status": "Aggregating platform metrics and benchmarking..."},
    )
    return run_analytics_crew(inputs)


@celery_app.task(bind=True, name="workflow.sales_improvement", soft_time_limit=1200, time_limit=1500)
def run_sales_improvement_task(self: object, inputs: dict) -> dict:
    self.update_state(
        state="PROGRESS",
        meta={"status": "Analyzing funnel and generating CRO playbook..."},
    )
    return run_sales_improvement_crew(inputs)


@celery_app.task(bind=True, name="workflow.bizdev", soft_time_limit=1200, time_limit=1500)
def run_bizdev_task(self: object, inputs: dict) -> dict:
    self.update_state(
        state="PROGRESS",
        meta={"status": "Prospecting leads and drafting outreach..."},
    )
    return run_bizdev_crew(inputs)


@celery_app.task(bind=True, name="workflow.scheduler", soft_time_limit=900, time_limit=1200)
def run_scheduler_task(self: object, inputs: dict) -> dict:
    self.update_state(
        state="PROGRESS",
        meta={"status": "Mapping timezones and resolving conflicts..."},
    )
    return run_scheduler_crew(inputs)
