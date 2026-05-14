import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes import create_router
from crews.bizdev_crew import run_bizdev_crew
from crews.content_crew import run_content_crew
from crews.marketing_crew import run_marketing_crew
from models import WorkflowType
from orchestrator import MasterOrchestrator

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Cross-Border E-Commerce AI Suite", version="0.1.0")
orchestrator = MasterOrchestrator()
orchestrator.register_crew(WorkflowType.BIZDEV, run_bizdev_crew)
orchestrator.register_crew(WorkflowType.MARKETING, run_marketing_crew)
orchestrator.register_crew(WorkflowType.CONTENT, run_content_crew)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(create_router(orchestrator))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
