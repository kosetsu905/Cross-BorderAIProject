import os
from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Task
from crewai_tools import SerperDevTool
from pydantic import BaseModel, ConfigDict, Field

from tools.custom.scheduler_tools import (
    ConflictCheckerTool,
    NotificationRouterTool,
    TimezoneHolidayTool,
)

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = BASE_DIR / "config" / "scheduler"
DEFAULT_MODEL = "gpt-4o-mini"


class ScheduledEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_name: str = Field(..., description="Name of the scheduled event or campaign")
    region: str = Field(..., description="Target market or region")
    start_utc: str = Field(..., description="Start time in UTC, ISO 8601 preferred")
    end_utc: str = Field(..., description="End time in UTC, ISO 8601 preferred")
    local_time_note: str = Field(..., description="Optimal local time and timezone context")
    priority: str = Field(..., description="Priority level such as High, Medium, or Low")


class ReminderTrigger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trigger_offset: str = Field(..., description="Time before event, such as T-7d")
    channel: str = Field(..., description="Notification channel")
    message_preview: str = Field(..., description="Preview of reminder content")
    escalation_rule: str = Field(..., description="Action if unacknowledged")


class SchedulerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimized_schedule: list[ScheduledEvent] = Field(
        ..., description="Final conflict-free event schedule"
    )
    regional_calendar_notes: str = Field(
        ..., description="Timezone, holiday, and peak season context"
    )
    reminder_sequence: list[ReminderTrigger] = Field(
        ..., description="Multi-channel reminder and escalation plan"
    )
    conflict_resolution_log: list[str] = Field(
        ..., description="Resolved conflicts and adjustment rationale"
    )
    export_instructions: str = Field(
        ..., description="Steps to export to Google Calendar, Outlook, or ICS"
    )


def _load_yaml_config(file_name: str) -> dict[str, Any]:
    config_path = CONFIG_DIR / file_name
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_campaign_tools() -> list[Any]:
    if os.getenv("SERPER_API_KEY"):
        return [SerperDevTool()]
    return []


def _memory_enabled() -> bool:
    return os.getenv("CREWAI_MEMORY_ENABLED", "false").lower() in {"1", "true", "yes"}


def _serialize_crew_result(result: Any) -> dict[str, Any]:
    pydantic_result = getattr(result, "pydantic", None)
    if pydantic_result is not None:
        if hasattr(pydantic_result, "model_dump"):
            return pydantic_result.model_dump()
        if hasattr(pydantic_result, "dict"):
            return pydantic_result.dict()

    json_dict = getattr(result, "json_dict", None)
    if isinstance(json_dict, dict):
        return json_dict

    raw = getattr(result, "raw", None)
    if raw is not None:
        return {"raw": raw}

    if isinstance(result, dict):
        return result

    return {"raw": str(result)}


def run_scheduler_crew(inputs: dict[str, Any]) -> dict[str, Any]:
    """Callable wrapper for FastAPI orchestration."""
    os.environ.setdefault("OPENAI_MODEL_NAME", DEFAULT_MODEL)

    agents_config = _load_yaml_config("agents.yaml")
    tasks_config = _load_yaml_config("tasks.yaml")

    calendar_manager = Agent(
        config=agents_config["global_calendar_manager"],
        tools=[TimezoneHolidayTool()],
    )
    campaign_planner = Agent(
        config=agents_config["campaign_alignment_planner"],
        tools=_build_campaign_tools(),
    )
    conflict_resolver = Agent(
        config=agents_config["conflict_resolution_optimizer"],
        tools=[ConflictCheckerTool()],
    )
    notification_coordinator = Agent(
        config=agents_config["notification_coordinator"],
        tools=[NotificationRouterTool()],
    )

    timezone_task = Task(
        config=tasks_config["timezone_holiday_mapping"],
        agent=calendar_manager,
    )
    alignment_task = Task(
        config=tasks_config["campaign_event_alignment"],
        agent=campaign_planner,
        context=[timezone_task],
    )
    conflict_task = Task(
        config=tasks_config["conflict_resolution_optimization"],
        agent=conflict_resolver,
        context=[alignment_task],
    )
    notification_task = Task(
        config=tasks_config["notification_reminder_setup"],
        agent=notification_coordinator,
        context=[conflict_task],
        output_pydantic=SchedulerOutput,
    )

    scheduler_crew = Crew(
        agents=[
            calendar_manager,
            campaign_planner,
            conflict_resolver,
            notification_coordinator,
        ],
        tasks=[timezone_task, alignment_task, conflict_task, notification_task],
        verbose=False,
        memory=_memory_enabled(),
    )

    return _serialize_crew_result(scheduler_crew.kickoff(inputs=inputs))
