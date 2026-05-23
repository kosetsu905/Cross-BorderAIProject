import json
import os
import time
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "")

WORKFLOW_EXAMPLES: dict[str, dict[str, Any]] = {
    "marketing": {
        "product_category": "Smart Home Security Cameras",
        "product_usp": "AI-powered motion detection, 4K resolution, privacy-first cloud storage",
        "target_markets": "US, UK, Germany, Japan",
        "target_languages": ["en-US", "en-GB", "de", "ja"],
        "budget": "$15,000 USD",
    },
    "content": {
        "subject": "Sustainable Activewear for Cold Climates",
        "product_category": "Eco-Friendly Winter Sportswear",
        "target_markets": "Germany, Japan, Canada",
        "target_languages": ["de", "ja", "en"],
        "platforms": ["Instagram", "LinkedIn", "X"],
    },
    "support": {
        "customer": "GlobalTech Solutions",
        "person": "Maria Chen",
        "inquiry": (
            "Our bulk order #EU-8842 is delayed. We need it by Friday for a product launch. "
            "What are the expedited shipping options and compensation policy?"
        ),
    },
    "analytics": {
        "product_category": "Smart Home Security Cameras",
        "target_markets": "US, UK, Germany, Japan",
        "date_range": "Last 30 Days",
        "currency": "USD",
    },
    "bizdev": {
        "product_category": "Smart Home Security Cameras",
        "partnership_type": "Regional Distributors & Retail Partners",
        "target_markets": "Germany, Japan, Canada",
        "target_languages": ["de", "ja", "en"],
        "key_decision_maker_roles": "Head of Procurement, Channel Manager",
    },
    "scheduler": {
        "event_type": "Product Launch & Promotional Campaign",
        "target_markets": "US, UK, Germany, Japan",
        "event_list": "Smart Camera Launch, Early Access Sale, Influencer Drop, Post-Launch Retargeting",
        "preferred_launch_window": "2026-05-15 to 2026-06-15",
    },
    "sales_improvement": {
        "product_category": "Smart Home Security Cameras",
        "target_markets": "US, EU, Japan",
        "current_avg_conversion": "2.1%",
        "target_conversion": "3.5%",
        "date_range": "Last 60 Days",
    },
}

PROGRESS_BY_STATUS = {
    "pending": 0.1,
    "running": 0.55,
    "completed": 1.0,
    "failed": 1.0,
}

PROGRESS_BY_EVENT = {
    "submitted": 0.08,
    "queued": 0.18,
    "running": 0.45,
    "task_plan": 0.2,
    "task_started": 0.25,
    "task_completed": 0.4,
    "retrying": 0.5,
    "cache_hit": 1.0,
    "completed": 1.0,
    "failed": 1.0,
}

ACTIVE_STATUSES = {"pending", "running"}


def _headers(token: str) -> dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _request_json(
    method: str,
    base_url: str,
    path: str,
    token: str,
    payload: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | list[dict[str, Any]] | None, str | None]:
    try:
        with httpx.Client(timeout=30) as client:
            response = client.request(
                method,
                _api_url(base_url, path),
                headers=_headers(token),
                json=payload,
            )
        response.raise_for_status()
        return response.json(), None
    except httpx.HTTPStatusError as exc:
        return None, f"{exc.response.status_code}: {exc.response.text}"
    except httpx.HTTPError as exc:
        return None, str(exc)


def _format_json(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _fetch_job(
    base_url: str,
    token: str,
    job_id: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None, str | None]:
    result, error = _request_json("GET", base_url, f"/api/v1/workflow/{job_id}", token)
    if error:
        return None, None, error

    events, events_error = _request_json(
        "GET",
        base_url,
        f"/api/v1/workflow/{job_id}/events",
        token,
    )
    if events_error:
        return result if isinstance(result, dict) else None, None, events_error

    return (
        result if isinstance(result, dict) else None,
        events if isinstance(events, list) else None,
        None,
    )


def _latest_event(events: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not events:
        return None
    return events[-1]


def _progress_value(status: str, events: list[dict[str, Any]] | None) -> float:
    latest = _latest_event(events)
    if latest:
        payload = latest.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("progress"), (int, float)):
            return float(payload["progress"])
        event_progress = PROGRESS_BY_EVENT.get(str(latest.get("event_type")), 0)
        if event_progress:
            return event_progress
    return PROGRESS_BY_STATUS.get(status, 0.25)


def _progress_label(status: str, latest_job: dict[str, Any], events: list[dict[str, Any]] | None) -> str:
    latest = _latest_event(events)
    if latest:
        message = latest.get("message")
        if message:
            payload = latest.get("payload")
            if isinstance(payload, dict) and payload.get("task_index") and payload.get("total_tasks"):
                agent = payload.get("agent_role")
                suffix = f" · {agent}" if agent else ""
                return f"{status}: {message}{suffix}"
            return f"{status}: {message}"

    result = latest_job.get("result")
    if isinstance(result, dict) and result.get("status"):
        return f"{status}: {result['status']}"
    return status


def main() -> None:
    st.set_page_config(page_title="Cross-Border AI Admin", layout="wide")
    st.title("Cross-Border AI Admin")

    with st.sidebar:
        st.header("API")
        api_base_url = st.text_input("Base URL", value=DEFAULT_API_BASE_URL)
        bearer_token = st.text_input("Bearer token", value=DEFAULT_BEARER_TOKEN, type="password")
        if st.button("Check health", width="stretch"):
            health, error = _request_json("GET", api_base_url, "/health", bearer_token)
            if error:
                st.error(error)
            else:
                st.success("API reachable")
                st.json(health)

    workflow_type = st.selectbox("Workflow", list(WORKFLOW_EXAMPLES.keys()))
    selected_example = WORKFLOW_EXAMPLES[workflow_type]

    if (
        "workflow_type" not in st.session_state
        or st.session_state.workflow_type != workflow_type
    ):
        st.session_state.workflow_type = workflow_type
        st.session_state.inputs_json = _format_json(selected_example)

    col_inputs, col_status = st.columns([1, 1])

    with col_inputs:
        st.subheader("Request")
        inputs_json = st.text_area("Inputs JSON", key="inputs_json", height=300)
        provider_credentials_json = st.text_area(
            "Provider credentials JSON",
            value="{}",
            height=120,
            help="Optional request-scoped provider credentials. Leave empty JSON for normal .env behavior.",
        )

        if st.button("Submit workflow", type="primary", width="stretch"):
            try:
                inputs = json.loads(inputs_json)
                provider_credentials = json.loads(provider_credentials_json or "{}")
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
            else:
                payload: dict[str, Any] = {
                    "workflow_type": workflow_type,
                    "inputs": inputs,
                }
                if provider_credentials:
                    payload["provider_credentials"] = provider_credentials

                result, error = _request_json("POST", api_base_url, "/api/v1/workflow", bearer_token, payload)
                if error:
                    st.error(error)
                elif isinstance(result, dict):
                    st.session_state.job_id = result.get("job_id")
                    st.session_state.latest_job = result
                    st.session_state.latest_events = []
                    st.success(f"Submitted job {st.session_state.job_id}")

    with col_status:
        st.subheader("Job")
        job_id = st.text_input("Job ID", value=st.session_state.get("job_id", ""))
        if job_id and job_id != st.session_state.get("job_id"):
            st.session_state.job_id = job_id
            st.session_state.latest_job = None
            st.session_state.latest_events = []
        auto_refresh = st.toggle("Auto refresh active job", value=True)
        refresh_interval = st.slider("Refresh interval seconds", 2, 15, 5)

        if st.button("Refresh now", width="stretch") and job_id:
            result, events, error = _fetch_job(api_base_url, bearer_token, job_id)
            if error:
                st.error(error)
            if result:
                st.session_state.latest_job = result
            if events is not None:
                st.session_state.latest_events = events

        latest_job = st.session_state.get("latest_job")
        if latest_job:
            status = str(latest_job.get("status", "pending"))
            latest_events = st.session_state.get("latest_events")
            st.progress(
                _progress_value(status, latest_events),
                text=_progress_label(status, latest_job, latest_events),
            )
            usage_cols = st.columns(4)
            usage_cols[0].metric("Tokens", latest_job.get("total_tokens") or 0)
            usage_cols[1].metric("Cost USD", latest_job.get("cost_usd") or 0)
            usage_cols[2].metric("Duration", latest_job.get("duration_seconds") or 0)
            usage_cols[3].metric("Status", status)
            if latest_job.get("cache_hit"):
                st.info(f"Served from cache: {latest_job.get('source_job_id')}")
            st.json(latest_job)

    latest_events = st.session_state.get("latest_events")
    if latest_events:
        st.subheader("Execution Events")
        st.dataframe(latest_events, width="stretch", hide_index=True)

    if st.session_state.get("auto_refresh_error"):
        st.warning(st.session_state.auto_refresh_error)

    latest_job = st.session_state.get("latest_job")
    active_job_id = st.session_state.get("job_id")
    if (
        auto_refresh
        and active_job_id
        and latest_job
        and str(latest_job.get("status")) in ACTIVE_STATUSES
    ):
        time.sleep(refresh_interval)
        result, events, error = _fetch_job(api_base_url, bearer_token, active_job_id)
        if error:
            st.session_state.auto_refresh_error = error
        if result:
            st.session_state.latest_job = result
        if events is not None:
            st.session_state.latest_events = events
        st.rerun()


if __name__ == "__main__":
    main()
