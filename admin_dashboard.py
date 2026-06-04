import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "")
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = (PROJECT_ROOT / "artifacts").resolve()

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
        "product_features": "Recycled thermal shell, wind-resistant construction, moisture-wicking base layer, designed for cold outdoor training.",
        "target_markets": "Germany, Japan, Canada",
        "target_languages": ["de", "ja", "en"],
        "platforms": ["Instagram", "LinkedIn", "X"],
        "brand_voice": "Premium, practical, sustainability-minded, and culturally respectful",
        "primary_keywords": ["thermal activewear", "winter training layer", "recycled sportswear"],
        "generate_visual_assets": False,
        "image_generation_count": 1,
        "image_quality": "auto",
        "image_size": "1024x1024",
    },
    "support": {
        "customer": "",
        "person": "",
        "inquiry": "",
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
WORKFLOW_PROVIDER_EXAMPLES: dict[str, dict[str, Any]] = {
    "content": {
        "content_image_model": "gpt-image-2",
        "content_image_scoring_model": "gpt-4o-mini",
        "content_image_artifact_dir": "artifacts/content_creation",
    },
    "support": {
        "llm_profile": "openrouter_gpt4o_mini",
        "gmail_access_token": "",
        "gmail_client_id": "",
        "gmail_client_secret": "",
        "gmail_refresh_token": "",
        "gmail_sender_email": "",
        "gmail_send_enabled": False,
        "gmail_watch_topic_name": "",
        "gmail_watch_label_ids": "INBOX",
        "gmail_sync_enabled": False,
        "whatsapp_access_token": "",
        "whatsapp_phone_number_id": "",
        "whatsapp_business_account_id": "",
        "whatsapp_verify_token": "",
        "whatsapp_app_secret": "",
        "whatsapp_send_enabled": False,
        "whatsapp_provider": "ycloud",
        "ycloud_api_key": "",
        "ycloud_whatsapp_from": "",
        "ycloud_waba_id": "",
        "ycloud_base_url": "https://api.ycloud.com/v2",
        "ycloud_webhook_secret": "",
    }
}
ITEM_CONDITIONS = ["", "unopened", "damaged", "defective", "opened", "used"]
SUPPORT_REGIONS = ["", "US", "EU", "JP", "DE", "FR", "AU"]
SUPPORT_FORM_KEYS = [
    "support_customer",
    "support_person",
    "support_inquiry",
    "support_ticket_id",
    "support_customer_email",
    "support_phone_number",
    "support_order_id",
    "support_item_sku",
    "support_detected_language",
    "support_return_reason",
    "support_lifetime_value",
    "support_order_count",
    "support_days_since_delivery",
    "support_item_condition",
    "support_region",
]


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


def _parse_inputs_json(raw_json: str, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return dict(fallback)
    return parsed if isinstance(parsed, dict) else dict(fallback)


def _support_value(inputs: dict[str, Any], key: str, default: Any = "") -> Any:
    value = inputs.get(key)
    return default if value is None else value


def _select_index(options: list[str], value: Any) -> int:
    normalized = str(value or "")
    return options.index(normalized) if normalized in options else 0


def _reset_support_form() -> None:
    for key in SUPPORT_FORM_KEYS:
        st.session_state.pop(key, None)
    st.session_state.inputs_json = _format_json(WORKFLOW_EXAMPLES["support"])


def _clean_optional_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in fields.items()
        if value is not None and str(value).strip()
    }


def _render_support_builder(selected_example: dict[str, Any]) -> None:
    current_inputs = _parse_inputs_json(st.session_state.get("inputs_json", "{}"), selected_example)
    order_history = current_inputs.get("order_history")
    if not isinstance(order_history, dict):
        order_history = {}

    reset_col, apply_col = st.columns([1, 2])
    if reset_col.button("Clear support form", width="stretch"):
        _reset_support_form()
        st.rerun()

    st.caption("Customer")
    basic_cols = st.columns(2)
    customer = basic_cols[0].text_input(
        "Customer",
        value=str(_support_value(current_inputs, "customer", selected_example["customer"])),
        key="support_customer",
    )
    person = basic_cols[1].text_input(
        "Contact person",
        value=str(_support_value(current_inputs, "person", selected_example["person"])),
        key="support_person",
    )
    inquiry = st.text_area(
        "Inquiry",
        value=str(_support_value(current_inputs, "inquiry", selected_example["inquiry"])),
        height=110,
        key="support_inquiry",
    )

    contact_cols = st.columns(2)
    customer_email = contact_cols[0].text_input(
        "Customer email",
        value=str(_support_value(current_inputs, "customer_email", "")),
        key="support_customer_email",
    )
    phone_number = contact_cols[1].text_input(
        "Phone number",
        value=str(_support_value(current_inputs, "phone_number", "")),
        key="support_phone_number",
    )

    st.caption("Ticket")
    ticket_cols = st.columns(2)
    ticket_id = ticket_cols[0].text_input(
        "Ticket ID",
        value=str(_support_value(current_inputs, "ticket_id", "")),
        key="support_ticket_id",
    )
    detected_language = ticket_cols[1].text_input(
        "Detected language override",
        value=str(_support_value(current_inputs, "detected_language", "")),
        help="Optional ISO language code such as en, ja, zh, es, de, or fr.",
        key="support_detected_language",
    )

    st.caption("Order")
    order_cols = st.columns(2)
    order_id = order_cols[0].text_input(
        "Order ID",
        value=str(_support_value(current_inputs, "order_id", "")),
        key="support_order_id",
    )
    item_sku = order_cols[1].text_input(
        "Item SKU",
        value=str(_support_value(current_inputs, "item_sku", "")),
        key="support_item_sku",
    )

    st.caption("Return / RMA")
    return_reason = st.text_input(
        "Return reason",
        value=str(_support_value(current_inputs, "return_reason", "")),
        key="support_return_reason",
    )

    st.caption("Customer history")
    history_cols = st.columns(5)
    lifetime_value = history_cols[0].number_input(
        "Lifetime value",
        min_value=0.0,
        value=float(order_history.get("lifetime_value") or 0),
        step=100.0,
        key="support_lifetime_value",
    )
    order_count = history_cols[1].number_input(
        "Order count",
        min_value=0,
        value=int(order_history.get("order_count") or 0),
        step=1,
        key="support_order_count",
    )
    days_since_delivery = history_cols[2].number_input(
        "Days since delivery",
        min_value=0,
        value=int(order_history.get("days_since_delivery") or 0),
        step=1,
        key="support_days_since_delivery",
    )
    item_condition = history_cols[3].selectbox(
        "Item condition",
        ITEM_CONDITIONS,
        index=_select_index(ITEM_CONDITIONS, order_history.get("item_condition")),
        key="support_item_condition",
    )
    region = history_cols[4].selectbox(
        "Region",
        SUPPORT_REGIONS,
        index=_select_index(SUPPORT_REGIONS, order_history.get("region")),
        key="support_region",
    )

    if apply_col.button("Apply support fields to JSON", width="stretch"):
        missing_required = [
            label
            for label, value in {
                "Customer": customer,
                "Contact person": person,
                "Inquiry": inquiry,
            }.items()
            if not str(value).strip()
        ]
        if missing_required:
            st.warning(f"Please fill required fields first: {', '.join(missing_required)}")
            return

        support_inputs: dict[str, Any] = {
            "customer": customer,
            "person": person,
            "inquiry": inquiry,
        }

        optional_fields: dict[str, Any] = {
            "ticket_id": ticket_id,
            "customer_email": customer_email,
            "phone_number": phone_number,
            "order_id": order_id,
            "item_sku": item_sku,
            "return_reason": return_reason,
            "detected_language": detected_language,
        }
        support_inputs.update(_clean_optional_fields(optional_fields))

        history_fields = _clean_optional_fields(
            {
                "lifetime_value": lifetime_value if lifetime_value else None,
                "order_count": order_count if order_count else None,
                "days_since_delivery": days_since_delivery if days_since_delivery else None,
                "item_condition": item_condition,
                "region": region,
            }
        )
        if history_fields:
            support_inputs["order_history"] = history_fields

        st.session_state.inputs_json = _format_json(support_inputs)
        st.rerun()


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


def _support_result_payload(latest_job: dict[str, Any]) -> dict[str, Any] | None:
    result = latest_job.get("result")
    return result if isinstance(result, dict) and "sentiment_analysis" in result else None


def _content_result_payload(latest_job: dict[str, Any]) -> dict[str, Any] | None:
    result = latest_job.get("result")
    if not isinstance(result, dict):
        return None
    content_keys = {"visual_assets", "multimodal_outputs", "seo_outputs", "visual_asset_scores"}
    return result if any(key in result for key in content_keys) else None


def _safe_artifact_path(value: Any) -> tuple[Path | None, list[str]]:
    diagnostics: list[str] = []
    for label, path in _artifact_path_candidates(value):
        is_artifact = _is_artifact_path(path)
        exists = path.is_file()
        diagnostics.append(f"{label}: {path} | exists={exists} | under_artifacts={is_artifact}")
        if exists and is_artifact:
            return path, diagnostics
    return None, diagnostics


def _artifact_path_candidates(value: Any) -> list[tuple[str, Path]]:
    raw = str(value or "").strip()
    if not raw:
        return []

    candidates: list[tuple[str, Path]] = []
    seen: set[str] = set()

    def add(label: str, path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except (OSError, RuntimeError):
            return
        key = str(resolved).casefold()
        if key not in seen:
            seen.add(key)
            candidates.append((label, resolved))

    add("original", Path(raw))

    normalized = raw.replace("\\", "/")
    app_artifact_prefix = "/app/artifacts/"
    if normalized.startswith(app_artifact_prefix):
        relative_artifact_path = normalized.removeprefix(app_artifact_prefix)
        add("mapped_from_container", ARTIFACT_ROOT / relative_artifact_path)
    elif normalized.startswith("app/artifacts/"):
        relative_artifact_path = normalized.removeprefix("app/artifacts/")
        add("mapped_from_container", ARTIFACT_ROOT / relative_artifact_path)
    elif normalized.startswith("artifacts/"):
        relative_artifact_path = normalized.removeprefix("artifacts/")
        add("mapped_from_relative_artifacts", ARTIFACT_ROOT / relative_artifact_path)
    elif "/artifacts/" in normalized:
        relative_artifact_path = normalized.split("/artifacts/", 1)[1]
        add("mapped_from_embedded_artifacts", ARTIFACT_ROOT / relative_artifact_path)

    return candidates


def _is_artifact_path(path: Path) -> bool:
    try:
        return path.is_relative_to(ARTIFACT_ROOT)
    except ValueError:
        return False


def _format_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{seconds:.2f}s"


def _score_for_asset(
    scores: list[dict[str, Any]],
    asset: dict[str, Any],
) -> dict[str, Any] | None:
    asset_path = asset.get("asset_path")
    for score in scores:
        if score.get("asset_path") == asset_path:
            return _normalized_visual_score(score)
    return None


def _normalized_visual_score(score: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(score)
    notes_payload = _json_object_from_text(score.get("notes"))
    if not notes_payload:
        return normalized

    for score_key in (
        "prompt_alignment_score",
        "cultural_fit_score",
        "brand_voice_score",
        "publish_readiness_score",
    ):
        if score_key in notes_payload:
            normalized[score_key] = _bounded_score(notes_payload[score_key])
    if notes_payload.get("notes"):
        normalized["notes"] = str(notes_payload["notes"]).strip()
    return normalized


def _json_object_from_text(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value.strip():
        return None

    json_text = _extract_json_object_text(value)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_json_object_text(text: str) -> str:
    stripped = text.strip()
    fenced_match = re.search(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        stripped,
        flags=re.IGNORECASE,
    )
    if fenced_match:
        stripped = fenced_match.group(1).strip()

    object_start = stripped.find("{")
    object_end = stripped.rfind("}")
    if object_start >= 0 and object_end > object_start:
        return stripped[object_start : object_end + 1].strip()
    return stripped


def _bounded_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(score, 100.0))


def _visual_asset_failure_note(asset: dict[str, Any]) -> str | None:
    if not asset.get("error") and str(asset.get("status") or "").lower() != "failed":
        return None

    parts: list[str] = []
    attempts = asset.get("attempts")
    last_status_code = asset.get("last_status_code")
    retryable_status = asset.get("retryable_status")
    if attempts is not None:
        parts.append(f"Attempts: {attempts}")
    if last_status_code:
        parts.append(f"Last HTTP status: {last_status_code}")
    if retryable_status is True:
        parts.append("Retryable upstream status; generation failed after retries.")
    elif retryable_status is False and last_status_code:
        parts.append("Non-retryable final status.")
    if asset.get("error"):
        parts.append(str(asset["error"]))
    return "\n".join(parts) if parts else None


def _render_content_visual_assets(latest_job: dict[str, Any]) -> None:
    content_result = _content_result_payload(latest_job)
    if not content_result:
        return

    visual_assets = [
        asset
        for asset in content_result.get("visual_assets", [])
        if isinstance(asset, dict)
    ]
    scores = [
        score
        for score in content_result.get("visual_asset_scores", [])
        if isinstance(score, dict)
    ]

    st.subheader("Content Visual Assets")
    if not visual_assets:
        st.info("No generated images yet. Set generate_visual_assets=true and provide an OpenAI key to generate visual assets.")
        return

    for index, asset in enumerate(visual_assets, start=1):
        score = _score_for_asset(scores, asset)
        st.markdown(f"**Asset {index}: {asset.get('status', 'unknown')}**")
        media_col, meta_col = st.columns([2, 1])
        safe_path, path_diagnostics = _safe_artifact_path(asset.get("asset_path"))
        if safe_path:
            media_col.image(str(safe_path), caption=safe_path.name, use_container_width=True)
        elif asset.get("asset_url"):
            media_col.image(str(asset["asset_url"]), caption="Remote generated asset", use_container_width=True)
        else:
            media_col.warning(asset.get("error") or "No local image file was returned.")
            failure_note = _visual_asset_failure_note(asset)
            if failure_note:
                media_col.info(failure_note)
            if path_diagnostics:
                media_col.caption("Local image diagnostics")
                media_col.code("\n".join(path_diagnostics))

        meta_col.metric("Generation", _format_seconds(asset.get("duration_seconds")))
        meta_col.metric("Model", asset.get("model") or "n/a")
        if asset.get("attempts") is not None:
            meta_col.metric("Attempts", str(asset.get("attempts")))
        if asset.get("last_status_code"):
            meta_col.metric("HTTP", str(asset.get("last_status_code")))
        if score:
            meta_col.metric("Scoring", _format_seconds(score.get("duration_seconds")))
            meta_col.metric("Readiness", f"{float(score.get('publish_readiness_score') or 0):.0f}/100")
        if asset.get("asset_path"):
            meta_col.caption(str(asset["asset_path"]))
        with st.expander(f"Prompt and scoring details {index}", expanded=False):
            st.write(asset.get("prompt") or "No prompt returned.")
            if score:
                if score.get("notes"):
                    st.caption(f"Scoring notes: {score['notes']}")
                st.json(score)


def _render_support_result_summary(latest_job: dict[str, Any]) -> None:
    support_result = _support_result_payload(latest_job)
    if not support_result:
        return

    st.subheader("Support Summary")
    if support_result.get("escalation_flag"):
        st.warning("Human handoff required for this ticket.")

    summary_cols = st.columns(4)
    summary_cols[0].metric("Ticket", support_result.get("ticket_id") or "n/a")
    summary_cols[1].metric("Email", support_result.get("customer_email") or "n/a")
    summary_cols[2].metric("Phone", support_result.get("phone_number") or "n/a")
    sentiment = support_result.get("sentiment_analysis")
    if isinstance(sentiment, dict):
        summary_cols[3].metric("Intent", sentiment.get("intent_category") or "n/a")
        st.json(
            {
                "sentiment": sentiment.get("sentiment_label"),
                "score": sentiment.get("sentiment_score"),
                "tier": sentiment.get("customer_tier"),
                "urgency": sentiment.get("urgency_level"),
                "language": sentiment.get("language_detected"),
            }
        )

    rma = support_result.get("rma_validation")
    logistics = support_result.get("logistics_output")
    if rma:
        st.markdown("**RMA Validation**")
        st.json(rma)
    if logistics:
        st.markdown("**Logistics Output**")
        st.json(logistics)
    if support_result.get("drafted_response"):
        st.markdown("**Drafted Response**")
        st.write(support_result["drafted_response"])
    email_delivery = support_result.get("email_delivery")
    if email_delivery:
        st.markdown("**Gmail Delivery**")
        st.json(email_delivery)


def _render_support_inbox(api_base_url: str, bearer_token: str) -> None:
    st.subheader("Support Inbox")
    sync_cols = st.columns([1, 1])
    latest_limit = sync_cols[0].number_input("Gmail latest count", min_value=1, max_value=20, value=5)
    latest_query = sync_cols[1].text_input("Gmail query", value="")
    if st.button("Sync latest Gmail", width="stretch"):
        sync_payload: dict[str, Any] = {"max_results": int(latest_limit)}
        if latest_query.strip():
            sync_payload["query"] = latest_query.strip()
        sync_result, sync_error = _request_json(
            "POST",
            api_base_url,
            "/api/v1/channels/gmail/sync-latest",
            bearer_token,
            sync_payload,
        )
        if sync_error:
            st.error(sync_error)
        else:
            st.success("Gmail sync completed")
            st.json(sync_result)

    conversations, error = _request_json(
        "GET",
        api_base_url,
        "/api/v1/support/conversations?limit=25",
        bearer_token,
    )
    if error:
        st.info(f"Support inbox unavailable: {error}")
        return
    if not isinstance(conversations, list) or not conversations:
        st.caption("No channel conversations yet.")
        return

    options = {
        f"{item.get('status')} | {item.get('channel')} | {item.get('customer_handle_masked')} | {item.get('conversation_id')}": item
        for item in conversations
    }
    selected_label = st.selectbox("Conversation", list(options.keys()))
    selected = options[selected_label]
    conversation_id = selected.get("conversation_id")
    if not conversation_id:
        return

    conversation, detail_error = _request_json(
        "GET",
        api_base_url,
        f"/api/v1/support/conversations/{conversation_id}",
        bearer_token,
    )
    if detail_error:
        st.error(detail_error)
        return
    if not isinstance(conversation, dict):
        return

    meta_cols = st.columns(4)
    meta_cols[0].metric("Status", conversation.get("status") or "n/a")
    meta_cols[1].metric("Channel", conversation.get("channel") or "n/a")
    meta_cols[2].metric("Approval", "required" if conversation.get("requires_approval") else "not required")
    meta_cols[3].metric("Escalation", "yes" if conversation.get("escalation_flag") else "no")

    messages = conversation.get("messages")
    if isinstance(messages, list):
        st.markdown("**Messages**")
        for message in messages[-8:]:
            direction = message.get("direction")
            status = message.get("status")
            text = message.get("text") or "[attachment or empty message]"
            st.caption(f"{direction} | {status} | {message.get('created_at')}")
            st.write(text)

    draft = conversation.get("draft_response")
    if draft:
        st.markdown("**Draft response**")
        edited = st.text_area("Approved message", value=str(draft), height=160)
        send_disabled = bool(conversation.get("escalation_flag"))
        send_label = "Approve and send" if conversation.get("requires_approval") else "Send manually"
        if not conversation.get("requires_approval") and not conversation.get("escalation_flag"):
            st.caption("Auto-send eligible. Use manual send only if provider auto-dispatch did not send.")
        if st.button(send_label, disabled=send_disabled, width="stretch"):
            result, send_error = _request_json(
                "POST",
                api_base_url,
                f"/api/v1/support/conversations/{conversation_id}/approve-send",
                bearer_token,
                {"message": edited},
            )
            if send_error:
                st.error(send_error)
            else:
                st.success("Approval submitted")
                st.json(result)
    else:
        st.caption("Draft response will appear after the linked support job completes.")


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
        st.session_state.provider_credentials_json = _format_json(
            WORKFLOW_PROVIDER_EXAMPLES.get(workflow_type, {})
        )

    col_inputs, col_status = st.columns([1, 1])

    with col_inputs:
        st.subheader("Request")
        if workflow_type == "support":
            with st.expander("Customer Service 1.1 fields", expanded=True):
                _render_support_builder(selected_example)
        inputs_json = st.text_area("Inputs JSON", key="inputs_json", height=300)
        provider_credentials_json = st.text_area(
            "Provider credentials JSON",
            key="provider_credentials_json",
            height=120,
            help="Optional request-scoped provider settings. For support, use llm_profile to select a server-side LLM profile.",
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
            _render_content_visual_assets(latest_job)
            _render_support_result_summary(latest_job)
            st.json(latest_job)

    latest_events = st.session_state.get("latest_events")
    if latest_events:
        st.subheader("Execution Events")
        st.dataframe(latest_events, width="stretch", hide_index=True)

    with st.expander("Omni-channel support inbox", expanded=False):
        _render_support_inbox(api_base_url, bearer_token)

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
