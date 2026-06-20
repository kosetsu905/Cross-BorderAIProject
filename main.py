import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes import create_router
from api.user_routes import create_user_router
from crews.analytics_crew import run_analytics_crew
from crews.bizdev_crew import run_bizdev_crew
from crews.content_crew import run_content_crew
from crews.marketing_crew import run_marketing_crew
from crews.scheduler_crew import run_scheduler_crew
from crews.sales_improvement_crew import run_sales_improvement_crew
from crews.support_crew import run_support_crew
from database import SessionLocal, init_db
from job_store import PostgresJobStore
from models import WorkflowType
from orchestrator import CeleryOrchestrator, MasterOrchestrator
from runtime_config import load_runtime_config
from utils.observability import init_observability

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Cross-Border E-Commerce AI Suite", version="0.1.0")
runtime_config = load_runtime_config()
init_observability("cross-border-fastapi", app=app, config_context=runtime_config.as_context())
init_db()
job_store = PostgresJobStore(SessionLocal)

if os.getenv("WORKFLOW_BACKEND", "local").lower() == "celery":
    orchestrator = CeleryOrchestrator(job_store=job_store, runtime_config=runtime_config)
    logger.info("Using Celery workflow backend.")
else:
    orchestrator = MasterOrchestrator(job_store=job_store, runtime_config=runtime_config)
    orchestrator.register_crew(WorkflowType.ANALYTICS, run_analytics_crew)
    orchestrator.register_crew(WorkflowType.BIZDEV, run_bizdev_crew)
    orchestrator.register_crew(WorkflowType.MARKETING, run_marketing_crew)
    orchestrator.register_crew(WorkflowType.CONTENT, run_content_crew)
    orchestrator.register_crew(WorkflowType.SCHEDULER, run_scheduler_crew)
    orchestrator.register_crew(WorkflowType.SALES_IMPROVEMENT, run_sales_improvement_crew)
    orchestrator.register_crew(WorkflowType.SUPPORT, run_support_crew)
    logger.info("Using local workflow backend with PostgreSQL job store.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(create_router(orchestrator))
app.include_router(create_user_router())


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
