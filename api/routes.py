import asyncio
import time
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query, Request, Response
from pydantic import ValidationError
from sqlalchemy.orm import Session

from api.auth import verify_bearer_token
from database import get_db_session
from db_models import SupportConversationRecord
from models import (
    JobEventResponse,
    JobResponse,
    JobStatus,
    ProviderCredentials,
    WorkflowGroupRequest,
    WorkflowRoutePlan,
    WorkflowRouteRequest,
    WorkflowRequest,
    WorkflowType,
)
from runtime_config import load_runtime_config
from services.whatsapp_provider import get_whatsapp_provider
from services.whatsapp_tmpl_mgr import WhatsAppTemplateManager
from support_inbox import SupportInboxStore
from tools.custom.gmail_tools import (
    get_gmail_message,
    gmail_label_ids,
    list_gmail_messages,
    parse_gmail_message,
    parse_gmail_pubsub_payload,
    resolve_gmail_access_token,
    send_gmail_reply_message,
    watch_gmail_mailbox,
)
from tools.custom.whatsapp_tools import (
    parse_ycloud_webhook,
    parse_whatsapp_webhook,
    verify_ycloud_signature,
    verify_whatsapp_signature,
)

AuthDependency = Annotated[None, Depends(verify_bearer_token)]
DbDependency = Annotated[Session, Depends(get_db_session)]
SERVICE_INQUIRY_CONTROL_FIELDS = {"llm_profile", "provider_credentials"}


def _service_inquiry_provider_credentials(payload: dict[str, object]) -> dict[str, Any] | None:
    credentials: dict[str, Any] = {}
    nested_credentials = payload.get("provider_credentials")
    if nested_credentials not in (None, ""):
        if not isinstance(nested_credentials, dict):
            raise HTTPException(status_code=400, detail="provider_credentials must be an object")
        credentials.update(nested_credentials)

    llm_profile = payload.get("llm_profile")
    if llm_profile not in (None, "") and not isinstance(llm_profile, str):
        raise HTTPException(status_code=400, detail="llm_profile must be a string")
    if isinstance(llm_profile, str):
        normalized_profile = llm_profile.strip()
        if normalized_profile:
            credentials["llm_profile"] = normalized_profile

    if not credentials:
        return None

    try:
        validated = ProviderCredentials.model_validate(credentials)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid provider_credentials: {exc}") from exc
    return validated.model_dump(exclude_none=True) or None


def create_router(orchestrator: object) -> APIRouter:
    router = APIRouter()

    async def submit_channel_message(
        db: Session,
        channel_message: object,
        source: str,
    ) -> dict[str, str | bool]:
        store = SupportInboxStore(db)
        conversation, message, created = store.upsert_inbound_message(channel_message)  # type: ignore[arg-type]
        if not created:
            return {
                "conversation_id": conversation.conversation_id,
                "message_id": message.message_id,
                "job_id": message.job_id or "",
                "created": False,
            }
        support_inputs = store.build_support_inputs(conversation, message)
        job_id = await orchestrator.submit_job(
            WorkflowType.SUPPORT,
            support_inputs,
            metadata={"source": source, "conversation_id": conversation.conversation_id},
        )
        store.attach_job(conversation.conversation_id, message.message_id, job_id)
        return {
            "conversation_id": conversation.conversation_id,
            "message_id": message.message_id,
            "job_id": job_id,
            "created": True,
        }

    async def sync_gmail_message_by_id(
        db: Session,
        *,
        message_id: str,
        source: str,
    ) -> dict[str, object]:
        config = load_runtime_config()
        access_token = _gmail_access_token_from_config(config)
        if not access_token:
            raise HTTPException(status_code=400, detail="Gmail access token or refresh-token credentials are required for Gmail sync")
        gmail_message = get_gmail_message(
            access_token=access_token,
            message_id=message_id,
            message_format="full",
        )
        channel_message = parse_gmail_message(gmail_message, mailbox_email=config.gmail_sender_email)
        return await submit_channel_message(db, channel_message, source)

    async def process_channel_webhook(
        db: Session,
        *,
        inbound_messages: list[object],
        status_updates: list[object],
        source: str,
    ) -> dict[str, object]:
        store = SupportInboxStore(db)
        submitted_jobs: list[dict[str, str]] = []
        duplicate_messages = 0
        updated_statuses = 0

        for status_update in status_updates:
            if store.apply_status_update(status_update):  # type: ignore[arg-type]
                updated_statuses += 1

        for inbound_message in inbound_messages:
            submitted = await submit_channel_message(db, inbound_message, source)
            if not submitted["created"]:
                duplicate_messages += 1
                continue
            submitted_jobs.append(
                {
                    "conversation_id": str(submitted["conversation_id"]),
                    "message_id": str(submitted["message_id"]),
                    "job_id": str(submitted["job_id"]),
                }
            )

        return {
            "status": "accepted",
            "messages_received": len(inbound_messages),
            "status_updates_received": len(status_updates),
            "duplicates": duplicate_messages,
            "statuses_updated": updated_statuses,
            "submitted_jobs": submitted_jobs,
        }

    @router.get("/api/v1/channels/whatsapp/webhook")
    async def verify_whatsapp_webhook(
        response: Response,
        hub_mode: str | None = Query(None, alias="hub.mode"),
        hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
        hub_challenge: str | None = Query(None, alias="hub.challenge"),
    ) -> str:
        config = load_runtime_config()
        if hub_mode == "subscribe" and hub_verify_token and hub_verify_token == config.whatsapp_verify_token:
            response.media_type = "text/plain"
            return hub_challenge or ""
        raise HTTPException(status_code=403, detail="WhatsApp webhook verification failed")

    @router.post("/api/v1/channels/whatsapp/webhook")
    async def receive_whatsapp_webhook(request: Request, db: DbDependency) -> dict[str, object]:
        config = load_runtime_config()
        body = await request.body()
        signature = request.headers.get("x-hub-signature-256")
        if not verify_whatsapp_signature(
            app_secret=config.whatsapp_app_secret,
            body=body,
            signature=signature,
        ):
            raise HTTPException(status_code=403, detail="Invalid WhatsApp webhook signature")

        payload = await request.json()
        inbound_messages, status_updates = parse_whatsapp_webhook(payload)
        return await process_channel_webhook(
            db,
            inbound_messages=inbound_messages,
            status_updates=status_updates,
            source="whatsapp_webhook",
        )

    @router.post("/api/v1/channels/ycloud/webhook")
    async def receive_ycloud_webhook(request: Request, db: DbDependency) -> dict[str, object]:
        config = load_runtime_config()
        body = await request.body()
        signature = request.headers.get("ycloud-signature")
        if not verify_ycloud_signature(
            webhook_secret=config.ycloud_webhook_secret,
            body=body,
            signature=signature,
        ):
            raise HTTPException(status_code=403, detail="Invalid YCloud webhook signature")

        payload = await request.json()
        inbound_messages, status_updates = parse_ycloud_webhook(payload)
        return await process_channel_webhook(
            db,
            inbound_messages=inbound_messages,
            status_updates=status_updates,
            source="ycloud_webhook",
        )

    @router.post("/api/v1/channels/gmail/pubsub")
    async def receive_gmail_pubsub(
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> dict[str, object]:
        payload = await request.json()
        parsed = parse_gmail_pubsub_payload(payload)
        history_id = parsed.get("history_id")
        email_address = parsed.get("email_address")
        message_id = (payload.get("message") or {}).get("attributes", {}).get("message_id")
        if message_id:
            background_tasks.add_task(_sync_gmail_message_background, str(message_id), "gmail_pubsub", orchestrator)
        return {
            "status": "accepted",
            "email_address": email_address,
            "history_id": history_id,
            "message_id": message_id,
            "sync": "queued" if message_id else "history_only",
        }

    @router.post("/api/v1/channels/gmail/sync")
    async def sync_gmail_message(
        _: AuthDependency,
        db: DbDependency,
        payload: dict[str, str] = Body(...),
    ) -> dict[str, object]:
        if not load_runtime_config().gmail_sync_enabled:
            raise HTTPException(status_code=400, detail="GMAIL_SYNC_ENABLED is false")
        message_id = payload.get("message_id")
        if not message_id:
            raise HTTPException(status_code=400, detail="message_id is required")
        return await sync_gmail_message_by_id(db, message_id=message_id, source="gmail_sync")

    @router.post("/api/v1/channels/gmail/sync-latest")
    async def sync_latest_gmail_messages(
        _: AuthDependency,
        db: DbDependency,
        payload: dict[str, object] | None = Body(default=None),
    ) -> dict[str, object]:
        config = load_runtime_config()
        if not config.gmail_sync_enabled:
            raise HTTPException(status_code=400, detail="GMAIL_SYNC_ENABLED is false")
        access_token = _gmail_access_token_from_config(config)
        if not access_token:
            raise HTTPException(status_code=400, detail="Gmail access token or refresh-token credentials are required for Gmail sync")

        payload = payload or {}
        max_results = int(payload.get("max_results") or 5)
        query = payload.get("query")
        label_ids = payload.get("label_ids")
        labels = (
            [str(label) for label in label_ids if str(label).strip()]
            if isinstance(label_ids, list)
            else gmail_label_ids(str(label_ids)) if isinstance(label_ids, str) else ["INBOX"]
        )
        summaries = list_gmail_messages(
            access_token=access_token,
            label_ids=labels,
            max_results=max_results,
            query=str(query) if query else None,
        )
        results: list[dict[str, object]] = []
        created_count = 0
        duplicate_count = 0
        for summary in summaries:
            message_id = summary.get("id")
            if not message_id:
                continue
            synced = await sync_gmail_message_by_id(
                db,
                message_id=str(message_id),
                source="gmail_sync_latest",
            )
            if synced.get("created"):
                created_count += 1
            else:
                duplicate_count += 1
            results.append(synced)
        return {
            "status": "completed",
            "requested": max_results,
            "fetched": len(summaries),
            "created": created_count,
            "duplicates": duplicate_count,
            "results": results,
        }

    @router.post("/api/v1/channels/gmail/watch")
    async def watch_gmail_channel(_: AuthDependency) -> dict[str, object]:
        config = load_runtime_config()
        access_token = _gmail_access_token_from_config(config)
        if not access_token or not config.gmail_watch_topic_name:
            raise HTTPException(status_code=400, detail="Gmail access token/refresh credentials and GMAIL_WATCH_TOPIC_NAME are required")
        try:
            return watch_gmail_mailbox(
                access_token=access_token,
                topic_name=config.gmail_watch_topic_name,
                label_ids=gmail_label_ids(config.gmail_watch_label_ids),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/v1/workflow", response_model=JobResponse)
    async def trigger_workflow(req: WorkflowRequest, _: AuthDependency) -> JobResponse:
        try:
            provider_credentials = (
                req.provider_credentials.model_dump(exclude_none=True)
                if req.provider_credentials
                else None
            )
            job_id = await orchestrator.submit_job(
                req.workflow_type,
                req.inputs,
                provider_credentials=provider_credentials,
                metadata=req.metadata,
            )
            return JobResponse(**orchestrator.get_job_status(job_id))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/v1/workflow-group", response_model=JobResponse)
    async def trigger_workflow_group(req: WorkflowGroupRequest, _: AuthDependency) -> JobResponse:
        try:
            submit_group = getattr(orchestrator, "submit_workflow_group", None)
            if submit_group is None:
                raise ValueError("The configured orchestrator does not support workflow groups.")
            job_id = await submit_group(req)
            return JobResponse(**orchestrator.get_job_status(job_id))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/v1/workflow-route/plan", response_model=WorkflowRoutePlan)
    async def plan_workflow_route(req: WorkflowRouteRequest, _: AuthDependency) -> WorkflowRoutePlan:
        try:
            plan_route = getattr(orchestrator, "plan_workflow_route", None)
            if plan_route is None:
                raise ValueError("The configured orchestrator does not support workflow routing.")
            return plan_route(req)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/api/v1/workflow-route", response_model=JobResponse)
    async def trigger_workflow_route(req: WorkflowRouteRequest, _: AuthDependency) -> JobResponse:
        try:
            submit_route = getattr(orchestrator, "submit_workflow_route", None)
            if submit_route is None:
                raise ValueError("The configured orchestrator does not support workflow routing.")
            job_id = await submit_route(req)
            return JobResponse(**orchestrator.get_job_status(job_id))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/api/v1/workflow/{job_id}", response_model=JobResponse)
    async def get_workflow_status(job_id: str, _: AuthDependency) -> JobResponse:
        job_data = orchestrator.get_job_status(job_id)
        if job_data.get("status") == JobStatus.FAILED and job_data.get("error"):
            raise HTTPException(status_code=404, detail=job_data["error"])
        return JobResponse(**job_data)

    @router.get("/api/v1/workflow/{job_id}/events", response_model=list[JobEventResponse])
    async def get_workflow_events(job_id: str, _: AuthDependency) -> list[JobEventResponse]:
        return [
            JobEventResponse(**event)
            for event in orchestrator.get_job_events(job_id)
        ]

    @router.post("/api/v1/service/inquiry")
    async def submit_customer_service_inquiry(
        _: AuthDependency,
        payload: dict[str, object] = Body(...),
    ) -> dict[str, object]:
        provider_credentials = _service_inquiry_provider_credentials(payload)
        support_payload = {
            key: value
            for key, value in payload.items()
            if key not in SERVICE_INQUIRY_CONTROL_FIELDS
        }
        customer_key = str(
            support_payload.get("customer_email")
            or support_payload.get("customer")
            or support_payload.get("person")
            or "unknown"
        )
        session_id = str(support_payload.get("session_id") or f"sess_{customer_key}_{int(time.time())}")
        inputs = {
            **support_payload,
            "session_id": session_id,
            "customer": support_payload.get("customer") or support_payload.get("person") or customer_key,
            "person": support_payload.get("person") or support_payload.get("customer") or customer_key,
            "inquiry": support_payload.get("inquiry") or support_payload.get("inquiry_text") or "",
            "inquiry_text": support_payload.get("inquiry_text") or support_payload.get("inquiry") or "",
            "conversation_history": support_payload.get("conversation_history") or [],
            "customer_tier": support_payload.get("customer_tier") or "STANDARD",
        }
        try:
            job_id = await orchestrator.submit_job(
                WorkflowType.SUPPORT,
                inputs,
                provider_credentials=provider_credentials,
                metadata={"source": "service_inquiry", "session_id": session_id},
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "status": "queued",
            "job_id": job_id,
            "session_id": session_id,
            "workflow_type": WorkflowType.SUPPORT.value,
        }

    @router.get("/api/v1/support/conversations")
    async def list_support_conversations(
        _: AuthDependency,
        db: DbDependency,
        limit: int = Query(50, ge=1, le=200),
    ) -> list[dict[str, object]]:
        return SupportInboxStore(db).list_conversations(limit=limit)

    @router.get("/api/v1/support/conversations/{conversation_id}")
    async def get_support_conversation(
        conversation_id: str,
        _: AuthDependency,
        db: DbDependency,
    ) -> dict[str, object]:
        store = SupportInboxStore(db)
        conversation = store.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        latest_job_id = conversation.get("latest_job_id")
        if latest_job_id:
            store.sync_job_result(str(conversation_id), orchestrator.get_job_status(str(latest_job_id)))
            conversation = store.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation

    @router.post("/api/v1/support/conversations/{conversation_id}/assign")
    async def assign_support_conversation(
        conversation_id: str,
        _: AuthDependency,
        db: DbDependency,
        payload: dict[str, str | None] | None = Body(default=None),
    ) -> dict[str, object]:
        payload = payload or {}
        assigned = SupportInboxStore(db).assign_conversation(conversation_id, payload.get("assigned_to"))
        if assigned is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return assigned

    @router.post("/api/v1/support/conversations/{conversation_id}/approve-send")
    async def approve_support_conversation_send(
        conversation_id: str,
        _: AuthDependency,
        db: DbDependency,
        payload: dict[str, str | None] | None = Body(default=None),
    ) -> dict[str, object]:
        payload = payload or {}
        store = SupportInboxStore(db)
        conversation = db.get(SupportConversationRecord, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        latest_job_id = conversation.latest_job_id
        if latest_job_id:
            store.sync_job_result(conversation_id, orchestrator.get_job_status(str(latest_job_id)))
            conversation = db.get(SupportConversationRecord, conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if conversation.escalation_flag:
            raise HTTPException(status_code=409, detail="Conversation requires human handoff and cannot be sent from approval API")
        draft_text = payload.get("message") or conversation.draft_response
        if not draft_text:
            raise HTTPException(status_code=409, detail="No draft response is available for this conversation")
        if conversation.channel not in {"whatsapp", "gmail"}:
            raise HTTPException(status_code=400, detail=f"Approval send is not implemented for channel '{conversation.channel}'")

        config = load_runtime_config()
        if conversation.channel == "gmail":
            if not config.gmail_send_enabled:
                return {
                    "status": "disabled",
                    "conversation_id": conversation_id,
                    "message_id": None,
                    "error": "GMAIL_SEND_ENABLED is false; no provider call was made.",
                }
            access_token = _gmail_access_token_from_config(config)
            if not access_token or not config.gmail_sender_email or not conversation.customer_handle:
                return {
                    "status": "missing_credentials",
                    "conversation_id": conversation_id,
                    "message_id": None,
                    "error": "Gmail access token or refresh-token credentials, sender email, and recipient are required.",
                }
            draft_payload = conversation.draft_payload or {}
            raw_headers = _latest_inbound_headers(conversation_id, db)
            delivery = send_gmail_reply_message(
                access_token=access_token,
                sender=config.gmail_sender_email,
                recipient=conversation.customer_handle,
                subject=str(raw_headers.get("subject") or f"Support ticket {conversation.conversation_id}"),
                body=str(draft_text),
                thread_id=conversation.channel_thread_id,
                in_reply_to=raw_headers.get("message-id"),
                references=raw_headers.get("references"),
            )
            outbound = store.record_outbound_message(
                conversation_id=conversation_id,
                text=str(draft_text),
                channel_message_id=delivery.get("message_id"),
                delivery_status=str(delivery.get("status") or "failed"),
                raw_payload={"delivery": delivery, "draft_payload": draft_payload},
            )
            return {
                "status": delivery.get("status"),
                "conversation_id": conversation_id,
                "message_id": outbound.message_id,
                "channel_message_id": outbound.channel_message_id,
                "error": delivery.get("error"),
            }

        if not config.whatsapp_send_enabled:
            return {
                "status": "disabled",
                "conversation_id": conversation_id,
                "message_id": None,
                "error": "WHATSAPP_SEND_ENABLED is false; no provider call was made.",
            }
        if not conversation.customer_handle:
            return {
                "status": "missing_credentials",
                "conversation_id": conversation_id,
                "message_id": None,
                "error": "WhatsApp recipient is required.",
            }

        delivery, raw_payload = await _send_whatsapp_approval_delivery(
            config=config,
            store=store,
            conversation=conversation,
            conversation_id=conversation_id,
            draft_text=str(draft_text),
        )
        if delivery.get("status") == "missing_credentials":
            return {
                "status": "missing_credentials",
                "conversation_id": conversation_id,
                "message_id": None,
                "error": delivery.get("error"),
            }
        outbound = store.record_outbound_message(
            conversation_id=conversation_id,
            text=str(draft_text),
            channel_message_id=delivery.get("message_id"),
            delivery_status=str(delivery.get("status") or "failed"),
            raw_payload=raw_payload,
        )
        return {
            "status": delivery.get("status"),
            "conversation_id": conversation_id,
            "message_id": outbound.message_id,
            "channel_message_id": outbound.channel_message_id,
            "error": delivery.get("error"),
        }

    @router.get("/health")
    async def health_check() -> dict[str, object]:
        health: dict[str, object] = {
            "status": "healthy",
            "workflow_backend": orchestrator.__class__.__name__,
            "registered_workflows": [
                workflow.value for workflow in orchestrator.registered_workflows
            ],
        }
        job_store_name = getattr(orchestrator, "job_store_name", None)
        if job_store_name:
            health["job_store"] = job_store_name
        return health

    return router


def _sync_gmail_message_background(message_id: str, source: str, orchestrator: object) -> None:
    from database import SessionLocal

    config = load_runtime_config()
    access_token = _gmail_access_token_from_config(config)
    if not config.gmail_sync_enabled or not access_token:
        return
    with SessionLocal() as db:
        gmail_message = get_gmail_message(
            access_token=access_token,
            message_id=message_id,
            message_format="full",
        )
        channel_message = parse_gmail_message(gmail_message, mailbox_email=config.gmail_sender_email)
        store = SupportInboxStore(db)
        conversation, message, created = store.upsert_inbound_message(channel_message)
        if not created:
            return
        support_inputs = store.build_support_inputs(conversation, message)
        job_id = asyncio.run(
            orchestrator.submit_job(
                WorkflowType.SUPPORT,
                support_inputs,
                metadata={"source": source, "conversation_id": conversation.conversation_id},
            )
        )
        store.attach_job(conversation.conversation_id, message.message_id, job_id)


def _latest_inbound_headers(conversation_id: str, db: Session) -> dict[str, str]:
    from sqlalchemy import select
    from db_models import SupportMessageRecord

    record = db.execute(
        select(SupportMessageRecord)
        .where(SupportMessageRecord.conversation_id == conversation_id)
        .where(SupportMessageRecord.direction == "inbound")
        .order_by(SupportMessageRecord.created_at.desc())
    ).scalars().first()
    raw_payload = record.raw_payload if record else None
    headers = (raw_payload or {}).get("headers") if isinstance(raw_payload, dict) else None
    return headers if isinstance(headers, dict) else {}


async def _send_whatsapp_approval_delivery(
    *,
    config: object,
    store: SupportInboxStore,
    conversation: SupportConversationRecord,
    conversation_id: str,
    draft_text: str,
) -> tuple[dict[str, object], dict[str, object]]:
    draft_payload = conversation.draft_payload or {}
    session_state = store.session_manager.load_session(conversation_id) or {}
    provider = get_whatsapp_provider(config)
    if store.session_manager.is_window_expired(conversation_id):
        language = _whatsapp_template_language(draft_payload, session_state)
        template = WhatsAppTemplateManager.from_config(config).template_for_language(language)
        delivery = await provider.send_template_message(
            to=conversation.customer_handle,
            template_name=template["name"],
            language_code=template["lang"],
            parameters=[],
        )
        return delivery, {
            "delivery": delivery,
            "draft_payload": draft_payload,
            "session_language": session_state.get("language_preference"),
            "provider": delivery.get("provider") or getattr(provider, "provider_name", "unknown"),
            "send_mode": "template",
        }

    delivery = await provider.send_text_message(
        to=conversation.customer_handle,
        body=draft_text,
    )
    return delivery, {
        "delivery": delivery,
        "draft_payload": draft_payload,
        "provider": delivery.get("provider") or getattr(provider, "provider_name", "unknown"),
        "send_mode": "text",
    }


def _whatsapp_template_language(draft_payload: dict[str, object], session_state: dict[str, object]) -> str:
    sentiment = draft_payload.get("sentiment_analysis")
    if not isinstance(sentiment, dict):
        sentiment = {}
    candidates = [
        draft_payload.get("detected_language"),
        draft_payload.get("language_detected"),
        sentiment.get("language_detected"),
        session_state.get("language_preference"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "en"


def _gmail_access_token_from_config(config: object) -> str | None:
    return resolve_gmail_access_token(
        access_token=getattr(config, "gmail_access_token", None),
        client_id=getattr(config, "gmail_client_id", None),
        client_secret=getattr(config, "gmail_client_secret", None),
        refresh_token=getattr(config, "gmail_refresh_token", None),
    )
