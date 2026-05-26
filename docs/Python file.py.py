#1. Pydantic 结构化模型（匹配 expected_output）修改


from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal

class ListingVariant(BaseModel):
    platform: Literal["Amazon", "Meta", "Google", "TikTok"]
    region: str
    language: str
    title: str = Field(..., max_length=200)
    description_or_bullets: str | List[str]
    cta: str
    backend_keywords: List[str] = []
    visual_notes: Optional[str] = None

class ComplianceStatus(BaseModel):
    status: Literal["APPROVED", "REVIEW_REQUIRED", "REJECTED"]
    notes: List[str]
    required_edits: List[str] = []

class ListingPackageOutput(BaseModel):
    strategy_summary: str
    listing_variants: List[ListingVariant]
    compliance_audit: Dict[str, ComplianceStatus]  # e.g., {"amazon_de": {...}, "google_jp": {...}}
    launch_checklist: List[str]
    data_source: List[str]
    confidence_level: int = Field(..., ge=0, le=100)
    assumptions: List[str]



2. Crew 任务装配片段

# crews/listing_crew.py
import yaml
from crewai import Agent, Task, Crew
from pydantic import BaseModel

with open("config/agents.yaml") as f: agents_cfg = yaml.safe_load(f)
with open("config/tasks.yaml") as f: tasks_cfg = yaml.safe_load(f)

# Agents
strategist = Agent(config=agents_cfg["strategy_channel_listing_planner"], tools=[serper, scraper, platform_specs])
compliance_writer = Agent(config=agents_cfg["creative_compliance_specialist"], tools=[compliance_checker, localizer])

# Tasks
strategy_task = Task(
    config=tasks_cfg["listing_strategy_research"],
    agent=strategist
)
listing_task = Task(
    config=tasks_cfg["platform_listing_compliance_writer"],
    agent=compliance_writer,
    context=[strategy_task],
    output_pydantic=ListingPackageOutput  # 强类型校验 + JSON 序列化
)

listing_crew = Crew(
    agents=[strategist, compliance_writer],
    tasks=[strategy_task, listing_task],
    memory=True,
    verbose=False
)

# Execution
def run_listing_crew(inputs: dict) -> dict:
    result = listing_crew.kickoff(inputs=inputs)
    return result.pydantic.model_dump()