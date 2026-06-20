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
        "brand_name": "NorthPeak Layers",
        "product_url": "https://example.com/products/sustainable-activewear",
        "primary_keywords": ["thermal activewear", "winter training layer", "recycled sportswear"],
        "generate_reddit_geo": False,
        "generate_visual_assets": False,
        "image_generation_count": 1,
        "image_quality": "low",
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
    "task_failed": 0.4,
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
CONTENT_FORM_KEYS = [
    "content_subject",
    "content_product_category",
    "content_product_features",
    "content_target_markets",
    "content_target_languages",
    "content_platforms",
    "content_brand_voice",
    "content_brand_name",
    "content_product_url",
    "content_primary_keywords",
    "content_generate_reddit_geo",
    "content_generate_visual_assets",
    "content_image_generation_count",
    "content_image_quality",
    "content_image_size",
    "content_submit_raw_json",
]
CONTENT_IMAGE_QUALITIES = ["auto", "low", "medium", "high"]
CONTENT_IMAGE_SIZES = ["1024x1024", "1024x1536", "1536x1024", "auto"]
USER_OAUTH_PROVIDERS = [
    "google",
    "facebook",
    "twitter",
    "linkedin",
    "apple",
    "github",
    "microsoft",
    "wechat",
    "alipay",
    "weibo",
    "douyin",
    "qq",
]
USER_PAYMENT_METHOD_TYPES = [
    "credit_card",
    "debit_card",
    "paypal",
    "stripe",
    "apple_pay",
    "google_pay",
    "alipay_cn",
    "wechat_pay",
    "union_pay",
    "bank_transfer",
    "crypto",
]
USER_SUBSCRIPTION_PLANS = ["starter", "professional", "enterprise"]

CONTENT_STAGE_LABELS = {
    "workflow_submitted": "Workflow submitted",
    "workflow_queued": "Workflow queued",
    "workflow_running": "Workflow running",
    "task_plan": "Task plan",
    "research_strategy": "Research and strategy",
    "content_generation": "Localized content generation",
    "visual_localization": "Visual localization",
    "seo_metadata": "SEO metadata",
    "cultural_compliance": "Cultural compliance",
    "image_generation": "Image generation",
    "visual_scoring": "Visual scoring",
    "content_assembly": "Final assembly",
    "workflow_completed": "Workflow completed",
    "workflow_failed": "Workflow failed",
    "cache_hit": "Cache hit",
}
CONTENT_STATUS_LABELS = {
    "submitted": "Submitted",
    "queued": "Queued",
    "planned": "Planned",
    "running": "Running",
    "started": "Running",
    "completed": "Completed",
    "retrying": "Retrying",
    "failed": "Failed",
    "skipped": "Skipped",
}


def _headers(token: str) -> dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _user_headers(token: str) -> dict[str, str]:
    return _headers(token)


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


def _json_payload_from_text(raw_json: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_json or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _oauth_payload(provider: str, provider_user_id: str, provider_info_json: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "provider_user_id": provider_user_id.strip(),
        "provider_info": _json_payload_from_text(provider_info_json),
    }


def _payment_method_payload(payment_type: str, payment_data_json: str, is_default: bool) -> dict[str, Any]:
    return {
        "payment_type": payment_type,
        "payment_data": _json_payload_from_text(payment_data_json),
        "is_default": is_default,
    }


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


def _split_csv_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _content_inputs_from_form_values(
    subject: str,
    product_category: str,
    product_features: str,
    target_markets: str,
    target_languages: str,
    platforms: str,
    brand_voice: str,
    brand_name: str,
    product_url: str,
    primary_keywords: str,
    generate_reddit_geo: bool,
    generate_visual_assets: bool,
    image_generation_count: int,
    image_quality: str,
    image_size: str,
) -> dict[str, Any]:
    content_inputs: dict[str, Any] = {
        "subject": str(subject).strip(),
        "product_category": str(product_category).strip(),
        "target_markets": str(target_markets).strip(),
        "target_languages": _split_csv_values(target_languages),
        "platforms": _split_csv_values(platforms),
        "generate_reddit_geo": bool(generate_reddit_geo),
        "generate_visual_assets": bool(generate_visual_assets),
        "image_generation_count": int(image_generation_count),
        "image_quality": str(image_quality).strip() or "low",
        "image_size": str(image_size).strip() or "1024x1024",
    }
    optional_fields = _clean_optional_fields(
        {
            "product_features": product_features,
            "brand_voice": brand_voice,
            "brand_name": brand_name,
            "product_url": product_url,
        }
    )
    content_inputs.update(optional_fields)
    keywords = _split_csv_values(primary_keywords)
    if keywords:
        content_inputs["primary_keywords"] = keywords
    return content_inputs


def _reset_content_form() -> None:
    for key in CONTENT_FORM_KEYS:
        st.session_state.pop(key, None)
    st.session_state.inputs_json = _format_json(WORKFLOW_EXAMPLES["content"])


def _render_content_builder(selected_example: dict[str, Any]) -> dict[str, Any]:
    current_inputs = _parse_inputs_json(st.session_state.get("inputs_json", "{}"), selected_example)
    reset_col, apply_col = st.columns([1, 2])
    if reset_col.button("Reset content form", width="stretch"):
        _reset_content_form()
        st.rerun()

    basic_cols = st.columns(2)
    subject = basic_cols[0].text_input(
        "Subject",
        value=str(current_inputs.get("subject") or selected_example["subject"]),
        key="content_subject",
    )
    product_category = basic_cols[1].text_input(
        "Product category",
        value=str(current_inputs.get("product_category") or selected_example["product_category"]),
        key="content_product_category",
    )
    product_features = st.text_area(
        "Product features",
        value=str(current_inputs.get("product_features") or selected_example.get("product_features") or ""),
        height=90,
        key="content_product_features",
    )

    target_cols = st.columns(3)
    target_markets = target_cols[0].text_input(
        "Target markets",
        value=str(current_inputs.get("target_markets") or selected_example["target_markets"]),
        key="content_target_markets",
    )
    target_languages = target_cols[1].text_input(
        "Target languages",
        value=", ".join(
            _split_csv_values(
                current_inputs.get("target_languages") or selected_example["target_languages"]
            )
        ),
        key="content_target_languages",
    )
    platforms = target_cols[2].text_input(
        "Platforms",
        value=", ".join(_split_csv_values(current_inputs.get("platforms") or selected_example["platforms"])),
        key="content_platforms",
    )

    guidance_cols = st.columns(2)
    brand_voice = guidance_cols[0].text_area(
        "Brand voice",
        value=str(current_inputs.get("brand_voice") or selected_example.get("brand_voice") or ""),
        height=90,
        key="content_brand_voice",
    )
    primary_keywords = guidance_cols[1].text_area(
        "Primary keywords",
        value=", ".join(
            _split_csv_values(
                current_inputs.get("primary_keywords")
                or selected_example.get("primary_keywords")
                or []
            )
        ),
        height=90,
        key="content_primary_keywords",
    )

    entity_cols = st.columns([1, 1, 1])
    brand_name = entity_cols[0].text_input(
        "Brand name",
        value=str(current_inputs.get("brand_name") or selected_example.get("brand_name") or ""),
        key="content_brand_name",
    )
    product_url = entity_cols[1].text_input(
        "Product URL",
        value=str(current_inputs.get("product_url") or selected_example.get("product_url") or ""),
        key="content_product_url",
    )
    generate_reddit_geo = entity_cols[2].toggle(
        "Generate Reddit GEO",
        value=bool(current_inputs.get("generate_reddit_geo", selected_example.get("generate_reddit_geo", False))),
        key="content_generate_reddit_geo",
    )

    visual_cols = st.columns([1, 1, 1, 1])
    generate_visual_assets = visual_cols[0].toggle(
        "Generate visual assets",
        value=bool(current_inputs.get("generate_visual_assets", selected_example.get("generate_visual_assets", False))),
        key="content_generate_visual_assets",
    )
    image_generation_count = visual_cols[1].number_input(
        "Image count",
        min_value=1,
        max_value=4,
        value=int(current_inputs.get("image_generation_count") or selected_example.get("image_generation_count") or 1),
        step=1,
        disabled=not generate_visual_assets,
        key="content_image_generation_count",
    )
    image_quality = visual_cols[2].selectbox(
        "Image quality",
        CONTENT_IMAGE_QUALITIES,
        index=_select_index(
            CONTENT_IMAGE_QUALITIES,
            current_inputs.get("image_quality") or selected_example.get("image_quality"),
        ),
        disabled=not generate_visual_assets,
        key="content_image_quality",
    )
    image_size = visual_cols[3].selectbox(
        "Image size",
        CONTENT_IMAGE_SIZES,
        index=_select_index(
            CONTENT_IMAGE_SIZES,
            current_inputs.get("image_size") or selected_example.get("image_size"),
        ),
        disabled=not generate_visual_assets,
        key="content_image_size",
    )

    content_inputs = _content_inputs_from_form_values(
        subject=subject,
        product_category=product_category,
        product_features=product_features,
        target_markets=target_markets,
        target_languages=target_languages,
        platforms=platforms,
        brand_voice=brand_voice,
        brand_name=brand_name,
        product_url=product_url,
        primary_keywords=primary_keywords,
        generate_reddit_geo=generate_reddit_geo,
        generate_visual_assets=generate_visual_assets,
        image_generation_count=int(image_generation_count),
        image_quality=image_quality,
        image_size=image_size,
    )
    if apply_col.button("Apply content fields to JSON", width="stretch"):
        st.session_state.inputs_json = _format_json(content_inputs)
        st.rerun()
    return content_inputs


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
    progress_values: list[float] = []
    for event in events or []:
        payload = event.get("payload") if isinstance(event, dict) else None
        if isinstance(payload, dict) and isinstance(payload.get("progress"), (int, float)):
            progress_values.append(float(payload["progress"]))
            continue
        if isinstance(event, dict):
            event_progress = PROGRESS_BY_EVENT.get(str(event.get("event_type")), 0)
            if event_progress:
                progress_values.append(event_progress)
    if progress_values:
        return max(progress_values)
    return PROGRESS_BY_STATUS.get(status, 0.25)


def _progress_label(status: str, latest_job: dict[str, Any], events: list[dict[str, Any]] | None) -> str:
    latest = _latest_event(events)
    if latest:
        message = latest.get("message")
        if message:
            payload = latest.get("payload")
            if isinstance(payload, dict) and payload.get("task_index") and payload.get("total_tasks"):
                agent = payload.get("agent_role")
                suffix = f" - {agent}" if agent else ""
                return f"{status}: {message}{suffix}"
            return f"{status}: {message}"

    result = latest_job.get("result")
    if isinstance(result, dict) and result.get("status"):
        return f"{status}: {result['status']}"
    return status


def _event_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def _task_stage(task_name: Any) -> str:
    normalized = str(task_name or "")
    if normalized == "research_and_strategy":
        return "research_strategy"
    if normalized.startswith("content_creation_and_qa"):
        return "content_generation"
    return normalized or "task_plan"


def _timeline_status(event_type: str, payload: dict[str, Any]) -> str:
    status = str(payload.get("status") or "").strip().lower()
    if status:
        return status
    if event_type == "submitted":
        return "submitted"
    if event_type == "queued":
        return "queued"
    if event_type in {"running", "task_started"}:
        return "running"
    if event_type in {"completed", "task_completed", "cache_hit"}:
        return "completed"
    if event_type == "failed":
        return "failed"
    if event_type == "task_plan":
        return "planned"
    return event_type


def _timeline_stage(event_type: str, payload: dict[str, Any]) -> str:
    if payload.get("stage"):
        return str(payload["stage"])
    if event_type == "submitted":
        return "workflow_submitted"
    if event_type == "queued":
        return "workflow_queued"
    if event_type == "running":
        return "workflow_running"
    if event_type == "completed":
        return "workflow_completed"
    if event_type == "failed":
        return "workflow_failed"
    if event_type == "cache_hit":
        return "cache_hit"
    return _task_stage(payload.get("task_name"))


def _is_content_event(event: dict[str, Any]) -> bool:
    payload = _event_payload(event)
    if payload.get("scope") == "content":
        return True
    if payload.get("workflow_type") == "content":
        return True
    task_name = str(payload.get("task_name") or "")
    return task_name == "research_and_strategy" or task_name.startswith("content_creation_and_qa")


def _content_timeline_entries(events: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not events:
        return []

    entries: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict) or not _is_content_event(event):
            continue
        payload = _event_payload(event)
        event_type = str(event.get("event_type") or "")
        if event_type == "content_partial":
            continue
        stage = _timeline_stage(event_type, payload)
        status = _timeline_status(event_type, payload)
        entries.append(
            {
                "event_id": event.get("event_id"),
                "created_at": event.get("created_at"),
                "event_type": event_type,
                "stage": stage,
                "step": CONTENT_STAGE_LABELS.get(stage, stage.replace("_", " ").title()),
                "status": status,
                "status_label": CONTENT_STATUS_LABELS.get(status, status.title()),
                "message": event.get("message") or "",
                "language": payload.get("language"),
                "target_market": payload.get("target_market"),
                "agent_role": payload.get("agent_role"),
                "duration_seconds": payload.get("duration_seconds"),
                "asset_count": payload.get("asset_count"),
                "score_count": payload.get("score_count"),
                "error_summary": payload.get("error_summary"),
                "payload": payload,
            }
        )
    return entries


def _is_content_partial_event(event: dict[str, Any]) -> bool:
    payload = _event_payload(event)
    return (
        str(event.get("event_type") or "") == "content_partial"
        and payload.get("scope") == "content"
        and isinstance(payload.get("content"), dict)
    )


def _preview_group_key(language: Any, target_market: Any) -> str:
    language_text = str(language or "").strip()
    market_text = str(target_market or "").strip()
    return f"{language_text}|{market_text}"


def _content_live_preview_groups(events: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    sorted_events = sorted(
        [event for event in events or [] if isinstance(event, dict)],
        key=lambda event: int(event.get("event_id") or 0),
    )
    for event in sorted_events:
        if not isinstance(event, dict) or not _is_content_partial_event(event):
            continue
        payload = _event_payload(event)
        language = payload.get("language")
        target_market = payload.get("target_market")
        preview_type = str(payload.get("preview_type") or payload.get("stage") or "")
        if not preview_type:
            continue
        key = _preview_group_key(language, target_market)
        group = groups.setdefault(
            key,
            {
                "language": language,
                "target_market": target_market,
                "previews": {},
                "updated_at": None,
            },
        )
        group["previews"][preview_type] = payload["content"]
        group["updated_at"] = payload.get("created_at") or event.get("created_at")

    return sorted(
        groups.values(),
        key=lambda group: (
            str(group.get("language") or ""),
            str(group.get("target_market") or ""),
        ),
    )


def _preview_tab_label(group: dict[str, Any]) -> str:
    language = str(group.get("language") or "").strip()
    target_market = str(group.get("target_market") or "").strip()
    if language and target_market:
        return f"{language} - {target_market}"
    return language or target_market or "Content"


def _render_social_posts(posts: list[Any]) -> None:
    if not posts:
        st.caption("No social posts preview yet.")
        return
    for post in posts:
        if not isinstance(post, dict):
            continue
        with st.container(border=True):
            st.markdown(f"**{post.get('platform') or 'Platform'}**")
            st.write(post.get("content") or "")


def _render_live_article(content_package: dict[str, Any]) -> None:
    status = str(content_package.get("status") or "").strip().lower()
    if status == "failed":
        st.error("Needs retry")
        if content_package.get("error_summary"):
            st.warning(str(content_package["error_summary"]))
        return
    if status == "retrying":
        st.info("Retrying content generation")
        if content_package.get("error_summary"):
            st.caption(str(content_package["error_summary"]))
        return

    article = content_package.get("localized_article")
    if isinstance(article, dict):
        title = str(article.get("title") or "Localized Article")
        st.markdown(f"#### {title}")
        st.markdown(str(article.get("article") or ""))
    else:
        st.caption("Article preview is not available yet.")

    posts = content_package.get("social_media_posts")
    keywords = content_package.get("seo_keywords")
    compliance_notes = str(content_package.get("compliance_notes") or "").strip()
    st.markdown("**Social Posts**")
    _render_social_posts(posts if isinstance(posts, list) else [])
    if isinstance(keywords, list) and keywords:
        st.caption(f"Keywords: {', '.join(str(keyword) for keyword in keywords)}")
    if compliance_notes:
        st.info(compliance_notes)


def _render_live_visual_brief(visual_brief: dict[str, Any]) -> None:
    visual_spec = visual_brief.get("visual_spec")
    if not isinstance(visual_spec, dict):
        st.caption("Visual brief is not available yet.")
        return

    metric_cols = st.columns(3)
    metric_cols[0].metric("Style", visual_spec.get("style_guide") or "n/a")
    metric_cols[1].metric("Palette", visual_spec.get("color_palette") or "n/a")
    metric_cols[2].metric("Scene", visual_spec.get("background_scene") or "n/a")
    if visual_spec.get("model_demographics"):
        st.caption(f"Casting: {visual_spec['model_demographics']}")
    cultural_notes = visual_spec.get("cultural_notes")
    if isinstance(cultural_notes, list) and cultural_notes:
        st.markdown("**Cultural notes**")
        for note in cultural_notes:
            st.write(f"- {note}")
    if visual_spec.get("ai_image_prompt"):
        with st.expander("Image prompt", expanded=False):
            st.write(visual_spec["ai_image_prompt"])
    if visual_brief.get("image_text_consistency_check"):
        st.caption(str(visual_brief["image_text_consistency_check"]))


def _render_live_seo(seo_metadata: dict[str, Any]) -> None:
    if not seo_metadata:
        st.caption("SEO preview is not available yet.")
        return
    if seo_metadata.get("canonical_url_slug"):
        st.caption(f"Canonical slug: {seo_metadata['canonical_url_slug']}")
    strategies = seo_metadata.get("engine_specific_metadata")
    if isinstance(strategies, list):
        for strategy in strategies[:6]:
            if not isinstance(strategy, dict):
                continue
            with st.container(border=True):
                st.markdown(f"**{strategy.get('engine') or 'Search engine'}**")
                st.write(strategy.get("title_template") or "")
                st.caption(strategy.get("meta_description_template") or "")
    alt_texts = seo_metadata.get("alt_text_variants")
    if isinstance(alt_texts, list) and alt_texts:
        with st.expander("Alt text variants", expanded=False):
            st.json(alt_texts)


def _render_live_compliance(compliance: dict[str, Any]) -> None:
    risk_flags = compliance.get("risk_flags")
    if isinstance(risk_flags, list) and risk_flags:
        st.markdown("**Risk flags**")
        for flag in risk_flags:
            st.write(f"- {flag}")
    checklist = compliance.get("compliance_checklist")
    if isinstance(checklist, list) and checklist:
        st.markdown("**Checklist**")
        for item in checklist:
            st.write(f"- {item}")
    actions = compliance.get("recommended_actions")
    if isinstance(actions, list) and actions:
        st.markdown("**Recommended actions**")
        for action in actions:
            st.write(f"- {action}")


def _render_live_images(images: dict[str, Any]) -> None:
    status = str(images.get("status") or "pending")
    st.caption(f"Image generation status: {status}")
    if images.get("error_summary"):
        st.warning(str(images["error_summary"]))
    assets = images.get("assets")
    if not isinstance(assets, list) or not assets:
        return
    for index, asset in enumerate(assets, start=1):
        if not isinstance(asset, dict):
            continue
        with st.container(border=True):
            st.markdown(f"**Image {index}: {asset.get('status') or 'unknown'}**")
            safe_path, _ = _safe_artifact_path(asset.get("asset_path"))
            if safe_path:
                st.image(str(safe_path), caption=safe_path.name, use_container_width=True)
            elif asset.get("asset_url"):
                st.image(str(asset["asset_url"]), caption="Remote generated asset", use_container_width=True)
            elif asset.get("error_summary"):
                st.warning(str(asset["error_summary"]))
            meta = []
            if asset.get("attempts") is not None:
                meta.append(f"attempts={asset['attempts']}")
            if asset.get("last_status_code"):
                meta.append(f"http={asset['last_status_code']}")
            if asset.get("duration_seconds") is not None:
                meta.append(f"duration={_format_seconds(asset['duration_seconds'])}")
            if meta:
                st.caption(" | ".join(meta))


def _render_live_content_group(group: dict[str, Any]) -> None:
    previews = group.get("previews")
    if not isinstance(previews, dict):
        return
    updated_at = group.get("updated_at")
    if updated_at:
        st.caption(f"Last update: {updated_at}")

    content_package = previews.get("content_package")
    if isinstance(content_package, dict):
        st.markdown("### Article and Posts")
        _render_live_article(content_package)

    visual_brief = previews.get("visual_brief")
    if isinstance(visual_brief, dict):
        st.markdown("### Visual Brief")
        _render_live_visual_brief(visual_brief)

    seo_metadata = previews.get("seo_metadata")
    if isinstance(seo_metadata, dict):
        st.markdown("### SEO")
        _render_live_seo(seo_metadata)

    compliance = previews.get("compliance")
    if isinstance(compliance, dict):
        st.markdown("### Compliance")
        _render_live_compliance(compliance)

    images = previews.get("images")
    if isinstance(images, dict):
        st.markdown("### Images")
        _render_live_images(images)


def _render_live_content_preview(events: list[dict[str, Any]] | None) -> None:
    groups = _content_live_preview_groups(events)
    if not groups:
        return

    st.subheader("Live Content Preview")
    tabs = st.tabs([_preview_tab_label(group) for group in groups])
    for tab, group in zip(tabs, groups):
        with tab:
            _render_live_content_group(group)


def _content_visual_assets(latest_job: dict[str, Any]) -> list[dict[str, Any]]:
    content_result = _content_result_payload(latest_job)
    if not content_result:
        return []
    return [
        asset
        for asset in content_result.get("visual_assets", [])
        if isinstance(asset, dict)
    ]


def _content_reddit_geo_review_assets(latest_job: dict[str, Any]) -> list[dict[str, Any]]:
    content_result = _content_result_payload(latest_job)
    if not content_result:
        return []
    reddit_sections = _reddit_geo_display_sections(content_result)
    sections_by_key = {
        _reddit_geo_asset_key(section): section
        for section in reddit_sections
        if _reddit_geo_asset_key(section)
    }
    review_assets: list[dict[str, Any]] = []
    for asset in content_result.get("production_ready_assets", []):
        if not isinstance(asset, dict) or asset.get("asset_type") != "reddit_geo_post":
            continue
        enriched_asset = dict(asset)
        matched_section = sections_by_key.get(_reddit_geo_asset_key(asset))
        if matched_section:
            enriched_asset["data_source"] = matched_section.get("data_source")
            enriched_asset["confidence_level"] = matched_section.get("confidence_level")
            enriched_asset["source_ids"] = matched_section.get("source_ids")
            enriched_asset["title_options"] = matched_section.get("title_options")
        review_assets.append(enriched_asset)
    return review_assets


def _reddit_geo_asset_key(value: dict[str, Any]) -> str:
    parts = [
        str(value.get(key) or "").strip().casefold()
        for key in ("language", "target_market", "recommended_subreddit")
    ]
    return "|".join(parts) if any(parts) else ""


def _content_visual_generation_requested(
    latest_job: dict[str, Any],
    events: list[dict[str, Any]] | None,
    current_inputs: dict[str, Any] | None,
) -> bool:
    if current_inputs and bool(current_inputs.get("generate_visual_assets")):
        return True
    if _content_visual_assets(latest_job):
        return True
    return any(
        _event_payload(event).get("stage") == "image_generation"
        for event in events or []
        if isinstance(event, dict)
    )


def _latest_content_image_event(events: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    for event in reversed(events or []):
        if not isinstance(event, dict):
            continue
        payload = _event_payload(event)
        if payload.get("stage") == "image_generation":
            return event
    return None


def _should_show_content_image_placeholder(
    latest_job: dict[str, Any],
    events: list[dict[str, Any]] | None,
    current_inputs: dict[str, Any] | None,
) -> bool:
    status = str(latest_job.get("status") or "")
    if status not in ACTIVE_STATUSES:
        return False
    if not _content_visual_generation_requested(latest_job, events, current_inputs):
        return False
    if _content_visual_assets(latest_job):
        return False

    image_event = _latest_content_image_event(events)
    if image_event:
        payload = _event_payload(image_event)
        image_status = str(payload.get("status") or "").lower()
        if image_status in {"completed", "failed", "skipped"} and not payload.get("asset_count"):
            return False
    return True


def _support_result_payload(latest_job: dict[str, Any]) -> dict[str, Any] | None:
    result = latest_job.get("result")
    return result if isinstance(result, dict) and "sentiment_analysis" in result else None


def _content_result_payload(latest_job: dict[str, Any]) -> dict[str, Any] | None:
    result = latest_job.get("result")
    if not isinstance(result, dict):
        return None
    content_keys = {
        "visual_assets",
        "multimodal_outputs",
        "seo_outputs",
        "visual_asset_scores",
        "reddit_geo_posts",
        "reddit_geo_sources",
    }
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


def _reddit_geo_display_sections(content_result: dict[str, Any]) -> list[dict[str, Any]]:
    posts = [
        post
        for post in content_result.get("reddit_geo_posts", [])
        if isinstance(post, dict)
    ]
    sources = [
        source
        for source in content_result.get("reddit_geo_sources", [])
        if isinstance(source, dict) and not _hide_reddit_geo_source_for_display(source)
    ]
    sources_by_id = {
        str(source.get("source_id") or ""): source
        for source in sources
        if source.get("source_id")
    }
    sections: list[dict[str, Any]] = []
    for post in posts:
        source_ids = [
            str(source_id)
            for source_id in post.get("source_ids", [])
            if str(source_id).strip()
        ] if isinstance(post.get("source_ids"), list) else []
        sections.append(
            {
                "language": post.get("language"),
                "target_market": post.get("target_market"),
                "recommended_subreddit": post.get("recommended_subreddit"),
                "title_options": post.get("title_options") if isinstance(post.get("title_options"), list) else [],
                "body": str(post.get("body") or ""),
                "body_without_link": str(post.get("body_without_link") or ""),
                "disclosure_note": str(post.get("disclosure_note") or ""),
                "ai_search_entity_signals": (
                    post.get("ai_search_entity_signals")
                    if isinstance(post.get("ai_search_entity_signals"), list)
                    else []
                ),
                "source_ids": source_ids,
                "moderation_notes": (
                    post.get("moderation_notes")
                    if isinstance(post.get("moderation_notes"), list)
                    else []
                ),
                "data_source": post.get("data_source"),
                "confidence_level": post.get("confidence_level"),
                "sources": [
                    sources_by_id[source_id]
                    for source_id in source_ids
                    if source_id in sources_by_id
                ],
            }
        )
    return sections


def _hide_reddit_geo_source_for_display(source: dict[str, Any]) -> bool:
    url = str(source.get("url") or "").lower()
    if "tl=zh" in url:
        return True
    target_language = str(source.get("target_language") or "").strip().lower()
    if target_language.startswith(("zh", "ja", "ko")):
        return False
    snippet = str(source.get("snippet") or "")
    cjk_count = sum(1 for character in snippet if "\u4e00" <= character <= "\u9fff")
    return cjk_count >= 8


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


def _compact_error_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if "<!doctype html" in lowered or "<html" in lowered:
        if "520" in text:
            return "HTTP 520: upstream image API returned an HTML error page."
        return "Upstream image API returned an HTML error page."
    if "moderation_blocked" in lowered or "image_generation_user_error" in lowered:
        request_match = re.search(r"request ID\s+([A-Za-z0-9_-]+)", text)
        request_text = f" Request ID: {request_match.group(1)}." if request_match else ""
        return f"Image generation was blocked by the provider safety system.{request_text}"
    return text[:240]


def _safe_display_payload(value: Any) -> Any:
    if isinstance(value, dict):
        safe_payload: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() == "error":
                safe_payload[key] = _compact_error_text(item)
            else:
                safe_payload[key] = _safe_display_payload(item)
        return safe_payload
    if isinstance(value, list):
        return [_safe_display_payload(item) for item in value]
    if isinstance(value, str):
        return _compact_error_text(value) or ""
    return value


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
        parts.append(_compact_error_text(asset["error"]) or str(asset["error"]))
    return "\n".join(parts) if parts else None


def _render_content_image_placeholder() -> None:
    st.markdown(
        """
        <style>
        .content-image-scanner {
            position: relative;
            min-height: 220px;
            overflow: hidden;
            border: 1px solid #d8dee9;
            border-radius: 8px;
            background:
                repeating-linear-gradient(
                    0deg,
                    #f7f9fb 0,
                    #f7f9fb 10px,
                    #eef3f6 10px,
                    #eef3f6 20px
                );
        }
        .content-image-scanner::before {
            content: "";
            position: absolute;
            inset: -35% 0 auto 0;
            height: 42%;
            background: linear-gradient(
                180deg,
                rgba(255, 255, 255, 0) 0%,
                rgba(86, 141, 133, 0.28) 48%,
                rgba(255, 255, 255, 0) 100%
            );
            animation: contentImageScan 1.8s linear infinite;
        }
        .content-image-scanner__label {
            position: absolute;
            left: 18px;
            bottom: 16px;
            color: #25313a;
            font-size: 0.92rem;
            font-weight: 600;
        }
        @keyframes contentImageScan {
            from { transform: translateY(-20%); }
            to { transform: translateY(360%); }
        }
        </style>
        <div class="content-image-scanner">
            <div class="content-image-scanner__label">Generating visual asset preview...</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_content_timeline(entries: list[dict[str, Any]]) -> None:
    if not entries:
        st.caption("No content execution events have been recorded yet.")
        return

    for entry in entries:
        status = str(entry["status"])
        if status in {"failed"}:
            icon = "[failed]"
        elif status in {"completed", "cache_hit"}:
            icon = "[done]"
        elif status in {"skipped"}:
            icon = "[skipped]"
        elif status in {"retrying"}:
            icon = "[retrying]"
        elif status in {"running", "started"}:
            icon = "[running]"
        else:
            icon = "[queued]"

        with st.container(border=True):
            top_cols = st.columns([1.4, 1, 1, 1])
            top_cols[0].markdown(f"**{icon} {entry['step']}**")
            top_cols[1].caption(f"Status: {entry['status_label']}")
            if entry.get("language"):
                top_cols[2].caption(f"Language: {entry['language']}")
            if entry.get("target_market"):
                top_cols[3].caption(f"Market: {entry['target_market']}")

            detail_cols = st.columns([2, 1, 1])
            detail_cols[0].write(entry["message"])
            if entry.get("agent_role"):
                detail_cols[1].caption(str(entry["agent_role"]))
            if entry.get("duration_seconds") is not None:
                detail_cols[2].caption(f"Duration: {_format_seconds(entry['duration_seconds'])}")
            elif entry.get("created_at"):
                detail_cols[2].caption(f"Time: {entry['created_at']}")

            metrics: dict[str, Any] = {}
            if entry.get("asset_count") is not None:
                metrics["assets"] = entry["asset_count"]
            if entry.get("score_count") is not None:
                metrics["scores"] = entry["score_count"]
            if metrics:
                st.caption(" | ".join(f"{key}: {value}" for key, value in metrics.items()))
            if entry.get("error_summary"):
                st.warning(str(entry["error_summary"]))


def _render_technical_event_logs(events: list[dict[str, Any]]) -> None:
    if not events:
        st.caption("No events returned.")
        return

    rows = [
        {
            "event_id": event.get("event_id"),
            "created_at": event.get("created_at"),
            "event_type": event.get("event_type"),
            "message": event.get("message"),
            "payload": event.get("payload"),
        }
        for event in events
        if isinstance(event, dict)
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
    options = {
        f"#{event.get('event_id')} {event.get('event_type')} - {str(event.get('message') or '')[:80]}": event
        for event in events
        if isinstance(event, dict)
    }
    if options:
        selected = st.selectbox("Inspect raw event", list(options.keys()))
        st.json(options[selected])


def _render_execution_observability(events: list[dict[str, Any]] | None) -> None:
    if not events:
        return

    st.subheader("Execution Events")
    timeline_entries = _content_timeline_entries(events)
    if timeline_entries:
        timeline_tab, technical_tab = st.tabs(["Content timeline", "Technical logs"])
        with timeline_tab:
            _render_content_timeline(timeline_entries)
        with technical_tab:
            _render_technical_event_logs(events)
    else:
        _render_technical_event_logs(events)


def _render_content_visual_assets(latest_job: dict[str, Any]) -> None:
    content_result = _content_result_payload(latest_job)
    if not content_result:
        return

    visual_assets = _content_visual_assets(latest_job)
    scores = [
        score
        for score in content_result.get("visual_asset_scores", [])
        if isinstance(score, dict)
    ]

    st.subheader("Content Visual Assets")
    if not visual_assets:
        st.info(
            "No generated images yet. Set generate_visual_assets=true and provide "
            "an OpenAI key to generate visual assets."
        )
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
            media_col.warning(_compact_error_text(asset.get("error")) or "No local image file was returned.")
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
                st.json(_safe_display_payload(score))


def _render_content_reddit_geo(latest_job: dict[str, Any]) -> None:
    content_result = _content_result_payload(latest_job)
    if not content_result:
        return

    sections = _reddit_geo_display_sections(content_result)
    if not sections:
        return

    st.subheader("Reddit GEO Publishing Package")
    for index, section in enumerate(sections, start=1):
        title = (
            f"{section.get('language') or 'Language'} - "
            f"{section.get('target_market') or 'Market'}"
        )
        with st.container(border=True):
            st.markdown(f"**Post {index}: {title}**")
            st.caption(
                " | ".join(
                    value
                    for value in [
                        f"Subreddit: {section.get('recommended_subreddit') or 'manual review'}",
                        f"Source: {section.get('data_source') or 'n/a'}",
                        f"Confidence: {section.get('confidence_level') or 'n/a'}",
                    ]
                    if value
                )
            )
            title_options = section.get("title_options")
            if isinstance(title_options, list) and title_options:
                st.markdown("**Title options**")
                for option in title_options:
                    st.write(f"- {option}")
            if section.get("disclosure_note"):
                st.info(str(section["disclosure_note"]))
            st.markdown("**Post body**")
            st.markdown(str(section.get("body") or ""))
            if section.get("body_without_link"):
                with st.expander("No-link body", expanded=False):
                    st.markdown(str(section["body_without_link"]))
            entity_signals = section.get("ai_search_entity_signals")
            if isinstance(entity_signals, list) and entity_signals:
                st.caption(
                    "Entity signals: "
                    + ", ".join(str(signal) for signal in entity_signals if str(signal).strip())
                )
            source_ids = section.get("source_ids")
            if isinstance(source_ids, list) and source_ids:
                st.caption("Source IDs: " + ", ".join(str(source_id) for source_id in source_ids))
            moderation_notes = section.get("moderation_notes")
            if isinstance(moderation_notes, list) and moderation_notes:
                st.markdown("**Moderation notes**")
                for note in moderation_notes:
                    st.write(f"- {note}")
            sources = section.get("sources")
            if isinstance(sources, list) and sources:
                with st.expander("Matched Reddit sources", expanded=False):
                    for source in sources:
                        st.markdown(
                            f"**{source.get('source_id')} - r/{source.get('subreddit')}**"
                        )
                        st.write(source.get("title") or "")
                        st.caption(source.get("url") or "")
                        if source.get("snippet"):
                            st.write(source["snippet"])


def _render_content_reddit_geo_review_assets(latest_job: dict[str, Any]) -> None:
    review_assets = _content_reddit_geo_review_assets(latest_job)
    if not review_assets:
        return

    st.subheader("Reddit GEO Review Assets")
    for index, asset in enumerate(review_assets, start=1):
        with st.container(border=True):
            st.markdown(f"**Review asset {index}: {asset.get('status') or 'manual review'}**")
            st.caption(
                " | ".join(
                    value
                    for value in [
                        f"Language: {asset.get('language') or 'n/a'}",
                        f"Market: {asset.get('target_market') or 'n/a'}",
                        f"Subreddit: {asset.get('recommended_subreddit') or 'manual review'}",
                        f"Confidence: {asset.get('confidence_level') or 'n/a'}",
                    ]
                    if value
                )
            )
            title_options = asset.get("title_options")
            if isinstance(title_options, list) and title_options:
                st.write("Suggested titles: " + " | ".join(str(title) for title in title_options[:3]))
            source_ids = asset.get("source_ids")
            if isinstance(source_ids, list) and source_ids:
                st.caption("Source IDs: " + ", ".join(str(source_id) for source_id in source_ids))


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


def _render_user_center(api_base_url: str) -> None:
    user_token = st.session_state.get("user_access_token", "")
    st.subheader("Users")
    token_cols = st.columns([2, 1])
    user_token = token_cols[0].text_input(
        "User access token",
        value=user_token,
        type="password",
        key="user_access_token",
    )
    if token_cols[1].button("Load profile", width="stretch", disabled=not bool(user_token)):
        result, error = _request_json("GET", api_base_url, "/api/v1/users/me", user_token)
        if error:
            st.error(error)
        elif isinstance(result, dict):
            st.session_state.current_user_profile = result
            st.success("Profile loaded")

    register_tab, profile_tab, oauth_tab, payment_tab, subscription_tab = st.tabs(
        ["Register / Login", "Profile", "OAuth", "Payment Methods", "Subscription"]
    )

    with register_tab:
        st.caption("Email account")
        email_cols = st.columns(2)
        email = email_cols[0].text_input("Email", key="user_email")
        password = email_cols[1].text_input("Password", type="password", key="user_password")
        username = st.text_input("Username", key="user_username")
        action_cols = st.columns(2)
        if action_cols[0].button("Register email user", width="stretch"):
            payload = {"email": email, "password": password}
            if username.strip():
                payload["username"] = username.strip()
            result, error = _request_json("POST", api_base_url, "/api/v1/users/register/email", "", payload)
            _handle_user_auth_result(result, error)
        if action_cols[1].button("Login email user", width="stretch"):
            payload = {"email": email, "password": password}
            result, error = _request_json("POST", api_base_url, "/api/v1/users/login/email", "", payload)
            _handle_user_auth_result(result, error)

        st.caption("Phone account")
        phone_cols = st.columns(3)
        country_code = phone_cols[0].text_input("Country code", value="+86", key="user_phone_country_code")
        phone = phone_cols[1].text_input("Phone", key="user_phone")
        phone_password = phone_cols[2].text_input("Phone password", type="password", key="user_phone_password")
        if st.button("Register phone user", width="stretch"):
            payload = {
                "country_code": country_code,
                "phone": phone,
                "password": phone_password,
                "username": username.strip() or None,
            }
            result, error = _request_json("POST", api_base_url, "/api/v1/users/register/phone", "", payload)
            _handle_user_auth_result(result, error)

        st.caption("Simulated OAuth login")
        oauth_cols = st.columns(2)
        provider = oauth_cols[0].selectbox("OAuth provider", USER_OAUTH_PROVIDERS, key="user_login_oauth_provider")
        provider_user_id = oauth_cols[1].text_input("Provider user ID", key="user_login_provider_user_id")
        provider_info_json = st.text_area(
            "Provider info JSON",
            value='{"email": "demo@example.com", "name": "Demo User", "email_verified": true}',
            key="user_login_provider_info",
            height=90,
        )
        if st.button("Login with simulated OAuth", width="stretch"):
            payload = _oauth_payload(provider, provider_user_id, provider_info_json)
            result, error = _request_json("POST", api_base_url, "/api/v1/users/oauth/login", "", payload)
            _handle_user_auth_result(result, error)

    with profile_tab:
        profile = st.session_state.get("current_user_profile")
        if isinstance(profile, dict):
            st.json(profile)
        update_cols = st.columns(2)
        first_name = update_cols[0].text_input("First name", key="user_profile_first_name")
        last_name = update_cols[1].text_input("Last name", key="user_profile_last_name")
        locale_cols = st.columns(3)
        country = locale_cols[0].text_input("Country", key="user_profile_country")
        timezone = locale_cols[1].text_input("Timezone", value="UTC", key="user_profile_timezone")
        language = locale_cols[2].text_input("Language", value="en", key="user_profile_language")
        if st.button("Update profile", width="stretch", disabled=not bool(user_token)):
            payload = _clean_optional_fields(
                {
                    "first_name": first_name,
                    "last_name": last_name,
                    "country": country,
                    "timezone": timezone,
                    "language": language,
                }
            )
            result, error = _request_json("PATCH", api_base_url, "/api/v1/users/me", user_token, payload)
            _handle_user_profile_result(result, error)

        password_cols = st.columns(2)
        old_password = password_cols[0].text_input("Old password", type="password", key="user_old_password")
        new_password = password_cols[1].text_input("New password", type="password", key="user_new_password")
        if st.button("Change password", width="stretch", disabled=not bool(user_token)):
            payload = {"old_password": old_password, "new_password": new_password}
            result, error = _request_json("POST", api_base_url, "/api/v1/users/me/password", user_token, payload)
            _handle_status_result(result, error)

        reset_cols = st.columns(2)
        reset_email = reset_cols[0].text_input("Reset email", key="user_reset_email")
        reset_new_password = reset_cols[1].text_input("Reset new password", type="password", key="user_reset_new_password")
        if st.button("Request reset token", width="stretch"):
            result, error = _request_json(
                "POST",
                api_base_url,
                "/api/v1/users/password-reset/request",
                "",
                {"email": reset_email},
            )
            if error:
                st.error(error)
            elif isinstance(result, dict):
                st.session_state.user_reset_token = result.get("reset_token") or ""
                st.success("Reset request accepted")
                st.json(result)
        reset_token = st.text_input(
            "Reset token",
            value=st.session_state.get("user_reset_token", ""),
            type="password",
            key="user_reset_token",
        )
        if st.button("Confirm password reset", width="stretch"):
            payload = {"token": reset_token, "new_password": reset_new_password}
            result, error = _request_json("POST", api_base_url, "/api/v1/users/password-reset/confirm", "", payload)
            _handle_status_result(result, error)

    with oauth_tab:
        link_cols = st.columns(2)
        link_provider = link_cols[0].selectbox("Provider", USER_OAUTH_PROVIDERS, key="user_link_oauth_provider")
        link_provider_user_id = link_cols[1].text_input("Provider user ID", key="user_link_provider_user_id")
        link_provider_info_json = st.text_area("Provider info JSON", value="{}", key="user_link_provider_info", height=90)
        oauth_action_cols = st.columns(2)
        if oauth_action_cols[0].button("Link OAuth provider", width="stretch", disabled=not bool(user_token)):
            payload = _oauth_payload(link_provider, link_provider_user_id, link_provider_info_json)
            result, error = _request_json("POST", api_base_url, "/api/v1/users/me/oauth", user_token, payload)
            _handle_user_profile_result(result, error)
        if oauth_action_cols[1].button("Unlink OAuth provider", width="stretch", disabled=not bool(user_token)):
            result, error = _request_json(
                "DELETE",
                api_base_url,
                f"/api/v1/users/me/oauth/{link_provider}",
                user_token,
            )
            _handle_status_result(result, error)

    with payment_tab:
        payment_cols = st.columns([1, 2, 1])
        payment_type = payment_cols[0].selectbox("Payment type", USER_PAYMENT_METHOD_TYPES, key="user_payment_type")
        payment_data_json = payment_cols[1].text_area(
            "Tokenized payment data JSON",
            value='{"last_four": "4242", "brand": "visa"}',
            key="user_payment_data",
            height=90,
        )
        is_default = payment_cols[2].checkbox("Default", value=True, key="user_payment_default")
        if st.button("Add payment method", width="stretch", disabled=not bool(user_token)):
            payload = _payment_method_payload(payment_type, payment_data_json, is_default)
            result, error = _request_json("POST", api_base_url, "/api/v1/users/me/payment-methods", user_token, payload)
            _handle_status_result(result, error)
        method_id = st.text_input("Payment method ID", key="user_payment_method_id")
        method_action_cols = st.columns(2)
        if method_action_cols[0].button("Set default", width="stretch", disabled=not bool(user_token)):
            result, error = _request_json(
                "POST",
                api_base_url,
                f"/api/v1/users/me/payment-methods/{method_id}/default",
                user_token,
            )
            _handle_status_result(result, error)
        if method_action_cols[1].button("Remove payment method", width="stretch", disabled=not bool(user_token)):
            result, error = _request_json(
                "DELETE",
                api_base_url,
                f"/api/v1/users/me/payment-methods/{method_id}",
                user_token,
            )
            _handle_status_result(result, error)

    with subscription_tab:
        plan_id = st.selectbox("Plan", USER_SUBSCRIPTION_PLANS, key="user_subscription_plan")
        subscription_cols = st.columns(2)
        if subscription_cols[0].button("Subscribe", width="stretch", disabled=not bool(user_token)):
            result, error = _request_json(
                "POST",
                api_base_url,
                "/api/v1/users/me/subscription",
                user_token,
                {"plan_id": plan_id},
            )
            _handle_user_profile_result(result, error)
        if subscription_cols[1].button("Cancel subscription", width="stretch", disabled=not bool(user_token)):
            result, error = _request_json(
                "POST",
                api_base_url,
                "/api/v1/users/me/subscription/cancel",
                user_token,
            )
            _handle_user_profile_result(result, error)


def _handle_user_auth_result(result: dict[str, Any] | list[dict[str, Any]] | None, error: str | None) -> None:
    if error:
        st.error(error)
        return
    if not isinstance(result, dict):
        st.error("Unexpected user auth response")
        return
    st.session_state.user_access_token = str(result.get("access_token") or "")
    user = result.get("user")
    if isinstance(user, dict):
        st.session_state.current_user_profile = user
    st.success("User session ready")
    st.json(result)


def _handle_user_profile_result(result: dict[str, Any] | list[dict[str, Any]] | None, error: str | None) -> None:
    if error:
        st.error(error)
        return
    if isinstance(result, dict):
        st.session_state.current_user_profile = result
        st.success("User profile updated")
        st.json(result)
    else:
        st.error("Unexpected user profile response")


def _handle_status_result(result: dict[str, Any] | list[dict[str, Any]] | None, error: str | None) -> None:
    if error:
        st.error(error)
        return
    st.success("Request completed")
    if result is not None:
        st.json(result)


def main() -> None:
    st.set_page_config(page_title="Cross-Border AI Admin", layout="wide")
    st.title("Cross-Border AI Admin")

    with st.sidebar:
        st.header("API")
        api_base_url = st.text_input("Base URL", value=DEFAULT_API_BASE_URL)
        bearer_token = st.text_input("Bearer token", value=DEFAULT_BEARER_TOKEN, type="password")
        view = st.selectbox("View", ["Workflow runner", "Users"])
        if st.button("Check health", width="stretch"):
            health, error = _request_json("GET", api_base_url, "/health", bearer_token)
            if error:
                st.error(error)
            else:
                st.success("API reachable")
                st.json(health)

    if view == "Users":
        _render_user_center(api_base_url)
        return

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
    content_form_inputs: dict[str, Any] | None = None
    content_submit_raw_json = False

    with col_inputs:
        st.subheader("Request")
        if workflow_type == "content":
            with st.expander("Content Creation fields", expanded=True):
                content_form_inputs = _render_content_builder(selected_example)
            with st.expander("Advanced request JSON", expanded=False):
                content_submit_raw_json = st.toggle(
                    "Submit raw Inputs JSON instead of form fields",
                    value=False,
                    key="content_submit_raw_json",
                )
                inputs_json = st.text_area("Inputs JSON", key="inputs_json", height=300)
        elif workflow_type == "support":
            with st.expander("Customer Service 1.1 fields", expanded=True):
                _render_support_builder(selected_example)
            inputs_json = st.text_area("Inputs JSON", key="inputs_json", height=300)
        else:
            inputs_json = st.text_area("Inputs JSON", key="inputs_json", height=300)
        provider_credentials_json = st.text_area(
            "Provider credentials JSON",
            key="provider_credentials_json",
            height=120,
            help=(
                "Optional request-scoped provider settings. For support, use llm_profile "
                "to select a server-side LLM profile."
            ),
        )

        if st.button("Submit workflow", type="primary", width="stretch"):
            try:
                if workflow_type == "content" and content_form_inputs is not None and not content_submit_raw_json:
                    inputs = content_form_inputs
                else:
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
            _render_live_content_preview(latest_events)
            if _should_show_content_image_placeholder(
                latest_job,
                latest_events,
                content_form_inputs or _parse_inputs_json(st.session_state.get("inputs_json", "{}"), selected_example),
            ):
                _render_content_image_placeholder()
            _render_content_reddit_geo(latest_job)
            _render_content_reddit_geo_review_assets(latest_job)
            _render_content_visual_assets(latest_job)
            _render_support_result_summary(latest_job)
            with st.expander("Raw job payload", expanded=False):
                st.json(latest_job)

    latest_events = st.session_state.get("latest_events")
    _render_execution_observability(latest_events)

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
