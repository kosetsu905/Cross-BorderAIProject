# Cross-Border E-Commerce AI Suite

CrewAI-based workflow suite for cross-border e-commerce operations. The project exposes FastAPI endpoints for multi-agent workflows, runs long jobs through Celery/Redis, stores durable job state in PostgreSQL, and can emit local observability traces to Phoenix, Langfuse, and MLflow.

## Table Of Contents

- [Overview](#overview)
- [What This Project Does](#what-this-project-does)
- [Architecture At A Glance](#architecture-at-a-glance)
- [Quick Start](#quick-start)
- [Environment Configuration](#environment-configuration)
- [Docker Compose Runtime](#docker-compose-runtime)
- [Observability Harness](#observability-harness)
- [Common Test Commands](#common-test-commands)
- [API Usage](#api-usage)
- [Workflow Examples](#workflow-examples)
- [Admin Dashboard](#admin-dashboard)
- [Support Inbox / Gmail / WhatsApp](#support-inbox--gmail--whatsapp)
- [Advanced Runtime Features](#advanced-runtime-features)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Archived Notes](#archived-notes)

## Overview

The suite currently supports these workflow types:

```text
analytics
bizdev
marketing
content
scheduler
sales_improvement
support
```

Every workflow follows the same broad pattern:

1. FastAPI validates request inputs with Pydantic models.
2. The orchestrator creates a durable job record.
3. A local worker or Celery worker runs the matching CrewAI crew.
4. Progress, usage, result, cache, and error state are written back to PostgreSQL.
5. Optional observability spans are exported to Phoenix/Langfuse/MLflow.

The project is Windows/PowerShell friendly. Commands below assume the workspace is:

```powershell
D:\Cross-BorderAIProject
```

## What This Project Does

The workflow layer is designed for cross-border e-commerce teams:

| Workflow | Purpose |
| --- | --- |
| `marketing` | Multi-market campaign planning, ad channel strategy, compliance notes, creative variants |
| `content` | Multilingual content generation, SEO metadata, Reddit GEO packages, optional image assets |
| `support` | Customer support drafting, QA, RAG policy retrieval, Gmail/WhatsApp inbox workflows |
| `analytics` | Commerce KPI analysis, market research, competitor benchmarking, evidence tracking |
| `bizdev` | Partner prospecting, lead enrichment, outreach planning |
| `scheduler` | Launch/event scheduling across markets, holidays, time zones |
| `sales_improvement` | Funnel analysis, CRO recommendations, pricing and playbook output |

The API also supports:

| Feature | Endpoint |
| --- | --- |
| Single workflow submission | `POST /api/v1/workflow` |
| Workflow group submission | `POST /api/v1/workflow-group` |
| Goal-driven route planning | `POST /api/v1/workflow-route/plan` |
| Goal-driven route execution | `POST /api/v1/workflow-route` |
| Job status | `GET /api/v1/workflow/{job_id}` |
| Job events | `GET /api/v1/workflow/{job_id}/events` |
| Service inquiry shortcut | `POST /api/v1/service/inquiry` |
| Health check | `GET /health` |

## Architecture At A Glance

```text
Client / Admin Dashboard
        |
        v
FastAPI API Layer
        |
        v
Workflow Orchestrator
        |
        +-- Local background execution
        |
        +-- Celery -> Redis -> Celery worker
        |
        v
CrewAI Crews
        |
        +-- YAML agents/tasks
        +-- Custom tools
        +-- LLM profiles/model tiering
        +-- Tool cache
        |
        v
PostgreSQL Job Store
        |
        +-- workflow_jobs
        +-- workflow_job_events
        +-- tool_cache_entries

Optional Observability:

FastAPI/Celery/CrewAI/Tools
        |
        +-- Phoenix / OpenTelemetry
        +-- Langfuse
        +-- MLflow
```

## Quick Start

### 1. Install Python dependencies

```powershell
cd D:\Cross-BorderAIProject
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Optional ML training dependencies are separate:

```powershell
python -m pip install -r requirements-ml.txt
```

### 2. Configure `.env`

The project expects a root-level `.env`.

Do not overwrite an existing `.env` that already contains real keys. Use `.env.example` as the single app and local observability template, and copy values selectively.

Minimum local workflow config:

```env
LLM_PROVIDER=openrouter
LLM_API_KEY=your_openrouter_key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL_NAME=openai/gpt-4o-mini
LLM_DISABLE_REASONING=false

OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key

WORKFLOW_BACKEND=celery
DATABASE_URL=postgresql+psycopg://crossborder:crossborder@localhost:5432/crossborder_ai
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

CREWAI_MEMORY_ENABLED=false
WORKFLOW_ASYNC_EXECUTION_ENABLED=true
WORKFLOW_RESULT_CACHE_ENABLED=true
TOOL_CACHE_ENABLED=true
```

Recommended local observability additions:

```env
OBSERVABILITY_ENABLED=true
OBSERVABILITY_CAPTURE_INPUT_OUTPUT=false
OBSERVABILITY_ENVIRONMENT=local

OTEL_ENABLED=true
OTEL_GLOBAL_AUTO_INSTRUMENTATION_ENABLED=false
OTEL_HTTPX_INSTRUMENTATION_ENABLED=false
OTEL_REDIS_INSTRUMENTATION_ENABLED=false
OTEL_SQLALCHEMY_INSTRUMENTATION_ENABLED=false
OTEL_CELERY_INSTRUMENTATION_ENABLED=false
FASTAPI_OTEL_AUTO_INSTRUMENTATION_ENABLED=false
OPENINFERENCE_CREWAI_ENABLED=false
OPENINFERENCE_LITELLM_ENABLED=true
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://phoenix:6006/v1/traces
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
PHOENIX_PROJECT_NAME=cross-border-ai-dev

LANGFUSE_BASE_URL=http://langfuse-web:3000
LANGFUSE_PUBLIC_KEY=replace_with_langfuse_public_key
LANGFUSE_SECRET_KEY=replace_with_langfuse_secret_key

MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=cross-border-ai
MLFLOW_TRACING_ENABLED=false
MLFLOW_PROMPT_REGISTRY_ENABLED=true
MLFLOW_SUPPORT_PROMPT_ALIAS=production
MLFLOW_AUTOMATIC_EVALUATION_ENABLED=false
MLFLOW_GENAI_JUDGE_DEFAULT_MODEL=openai:/gpt-4o-mini
MLFLOW_GIT_VERSION_TRACKING_ENABLED=true
```

### 3. Start Docker services

Guardrails Hub validators are installed into the FastAPI/Celery Docker image during build. Set a local Hub token before the first build:

```powershell
$env:GUARDRAILS_TOKEN = "<your_guardrails_hub_token>"
docker compose build fastapi celery_worker
```

First app build:

```powershell
docker compose up -d --build
```

Start observability services:

```powershell
docker compose -f docker-compose.monitoring.yml up -d
```

Check service state:

```powershell
docker compose ps
docker compose -f docker-compose.monitoring.yml ps
```

### 4. Verify the app

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health"
```

Local dashboards:

```text
FastAPI:  http://localhost:8000
Flower:   http://localhost:5555
Phoenix:  http://localhost:6006
Langfuse: http://localhost:3000
MLflow:   http://localhost:5000
```

## Environment Configuration

### How To Manage `.env`

Use this rule of thumb:

| File | Purpose |
| --- | --- |
| `.env` | Your real local runtime config. Do not commit it. |
| `.env.example` | Full app, Guardrails, and local observability config template. Copy values selectively. |

If you already have a working `.env`, append missing variables instead of replacing the whole file.

Keep real provider keys local. Rotate any key that was pasted into logs, screenshots, chat, or committed files.

### Required LLM Config

The generic `LLM_*` variables configure OpenAI-compatible providers:

```env
LLM_PROVIDER=openrouter
LLM_API_KEY=your_openrouter_key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL_NAME=openai/gpt-4o-mini
LLM_DISABLE_REASONING=false
```

Legacy OpenAI variables still work:

```env
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL_NAME=gpt-4o-mini
```

OpenRouter reasoning/thinking models such as Qwen3 and DeepSeek R1 variants are run with reasoning output disabled when needed so CrewAI can parse normal structured responses. Set `LLM_DISABLE_REASONING=true` to force that behavior.

### LLM Profiles And Model Tiering

Named profiles let operators switch workflow models without editing code:

```env
LLM_PROFILES_JSON={"openai_gpt4o_mini":{"llm_provider":"openai","llm_model_name":"gpt-4o-mini","llm_api_key_env":"OPENAI_API_KEY"},"openrouter_gpt4o_mini":{"llm_provider":"openrouter","llm_model_name":"openai/gpt-4o-mini","llm_base_url":"https://openrouter.ai/api/v1","llm_api_key_env":"OPENROUTER_API_KEY","llm_disable_reasoning":false},"openrouter_qwen3_14b":{"llm_provider":"openrouter","llm_model_name":"qwen/qwen3-14b","llm_base_url":"https://openrouter.ai/api/v1","llm_api_key_env":"OPENROUTER_API_KEY","llm_disable_reasoning":true}}
SUPPORT_LLM_PROFILE=openai_gpt4o_mini
WORKFLOW_GUARDRAILS_MODEL=openai_gpt4o_mini
WORKFLOW_GUARDRAILS_PROMPT_INJECTION_MODEL=openai_gpt4o_mini
WORKFLOW_GUARDRAILS_PROMPT_INJECTION_TIMEOUT_SECONDS=5
WORKFLOW_GUARDRAILS_PROMPT_INJECTION_CACHE_TTL_SECONDS=86400
WORKFLOW_GUARDRAILS_NATIVE_TRACING_ENABLED=true
SUPPORT_AUTO_SEND_CONFIDENCE_THRESHOLD=0.75
SUPPORT_QA_MODE=full_llm
```

`WORKFLOW_GUARDRAILS_MODEL` selects the Guardrails evaluation profile from `LLM_PROFILES_JSON`. The default `openai_gpt4o_mini` profile is used for asynchronous support provenance evaluation. Guardrails receives the LiteLLM model string only; profile `llm_disable_reasoning` is not passed into Hub validators.

`WORKFLOW_GUARDRAILS_PROMPT_INJECTION_MODEL` independently selects the latency-sensitive prompt-injection evaluator. It receives only the latest customer text, uses the configured hard timeout with no LiteLLM retries, and caches the masked decision in Redis by normalized-text SHA-256. A timeout or provider failure requires human review and cannot auto-dispatch support replies.

`SUPPORT_AUTO_SEND_CONFIDENCE_THRESHOLD` is the canonical support auto-dispatch threshold. A reply must meet this threshold, include a customer-facing response, and have no escalation, business hard blocker, or enforced Guardrails action. PII masking and asynchronous provenance findings are advisory and do not override an otherwise eligible send.

Support provenance is queued after auto-dispatch is evaluated. The Hub `ProvenanceLLM` validator receives only the final customer reply plus authoritative catalog, order, tracking, or policy evidence and runs once with `validation_method="full"`. Its `guardrail_provenance_evaluated` event and trace are advisory; they never change a completed delivery or conversation approval state.

`WORKFLOW_GUARDRAILS_NATIVE_TRACING_ENABLED=true` restores Guardrails AI's native `guard`, `step`, and `validator.validate` spans in Phoenix. Delivery-envelope contacts such as `customer_email`, `customer_handle`, and `phone_number` are masked before validator execution, but free-form customer text can still contain PII in native span input. Set this flag to `false` and recreate FastAPI/Celery when raw validator detail is not acceptable. Project-level `guardrail_*` spans and Langfuse metadata remain masked, and Guardrails Hub metrics collection remains disabled.

Two-tier routing is enabled by default but remains behavior-compatible until tier profiles are configured:

```env
WORKFLOW_MODEL_TIERING_ENABLED=true
WORKFLOW_WORKER_LLM_PROFILE=
WORKFLOW_REVIEWER_LLM_PROFILE=
```

### Memory And Shared Context

CrewAI native memory is opt-in:

```env
CREWAI_MEMORY_ENABLED=false
CREWAI_MEMORY_WORKFLOWS=marketing,content,analytics,bizdev,scheduler,sales_improvement
CREWAI_MEMORY_STORAGE_PATH=artifacts/crewai_memory
CREWAI_MEMORY_EMBEDDER_MODEL=text-embedding-3-small
WORKFLOW_CONTEXT_MAX_CHARS=12000
TASK_CONTEXT_MAX_CHARS=4000
```

Support is excluded from native CrewAI memory by default because customer conversation data can contain PII. Support uses controlled session history, PostgreSQL records, Redis session cache, and local RAG knowledge retrieval.

### Optional Providers

The system can run without most provider credentials. Missing providers usually trigger explicit fallback data or reduced output.

| Area | Variables |
| --- | --- |
| Search | `SERPER_API_KEY`, `SERPER_DEEP_READ_*` |
| Shopify | `SHOPIFY_STORE_DOMAIN`, `SHOPIFY_ADMIN_ACCESS_TOKEN`, `SHOPIFY_API_VERSION` |
| Amazon | `AMAZON_SP_API_ENDPOINT`, `AMAZON_SP_API_ACCESS_TOKEN`, `AMAZON_MARKETPLACE_IDS` |
| BizDev | `CRUNCHBASE_API_KEY`, `APOLLO_API_KEY` |
| Scheduler | `HOLIDAY_API_KEY` |
| Marketing ads | `GOOGLE_ADS_*`, `META_*`, `TIKTOK_*` |
| Gmail | `GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`, `GMAIL_REFRESH_TOKEN`, `GMAIL_SEND_ENABLED`, `GMAIL_SYNC_ENABLED` |
| WhatsApp | `WHATSAPP_*`, `YCLOUD_*` |
| PIM | `PIM_BACKEND`, `PIM_AKENEO_*`, `PIM_PLYTIX_*`, `PIM_CUSTOM_*` |
| Support | `SUPPORT_KNOWLEDGE_DIR`, `SUPPORT_HANDOFF_WEBHOOK_URL`, `SUPPORT_SESSION_*` |

Compatibility aliases accepted by `runtime_config.py`:

```text
WHATSAPP_TOKEN -> WHATSAPP_ACCESS_TOKEN
WHATSAPP_PHONE_ID -> WHATSAPP_PHONE_NUMBER_ID
SLACK_WEBHOOK_URL -> SUPPORT_HANDOFF_WEBHOOK_URL
```

### Request-Scoped Credentials

Provider settings can also be supplied per request with `provider_credentials`. These values are merged into the workflow runtime context and are not stored in job inputs.

```json
{
  "workflow_type": "support",
  "inputs": {
    "customer": "Maria",
    "person": "Maria",
    "inquiry": "I need help with a return."
  },
  "provider_credentials": {
    "llm_profile": "openrouter_gpt4o_mini",
    "serper_api_key": "request_scoped_serper_key"
  }
}
```

## Docker Compose Runtime

### Which Command Should I Run?

| Situation | Command |
| --- | --- |
| First app build | Use the Quick Start first-build command above |
| Start app services after they already exist | `docker compose up -d` |
| Start app plus monitoring | `docker compose up -d && docker compose -f docker-compose.monitoring.yml up -d` |
| Code or `requirements.txt` changed | `docker compose build fastapi celery_worker` then `docker compose up -d fastapi celery_worker` |
| Only `.env` changed | `docker compose up -d --force-recreate fastapi celery_worker flower` |
| Monitoring secrets changed before first use | `docker compose -f docker-compose.monitoring.yml up -d --force-recreate` |
| Monitoring secrets changed after old volumes were initialized | `docker compose -f docker-compose.monitoring.yml down -v` then `docker compose -f docker-compose.monitoring.yml up -d` |
| Stop app containers | `docker compose down` |
| Stop monitoring containers | `docker compose -f docker-compose.monitoring.yml down` |

### Runtime Services

Main stack:

| Service | Purpose | Port |
| --- | --- | --- |
| `fastapi` | API server | `8000` |
| `celery_worker` | CrewAI workflow execution | none |
| `postgres` | Workflow/job/tool-cache state | `5432` |
| `redis` | Celery broker/result backend and hot cache | `6379` |
| `flower` | Celery UI | `5555` |

Monitoring stack:

| Service | Purpose | Port |
| --- | --- | --- |
| `phoenix` | OpenTelemetry trace UI and collector | `6006`, `4317` |
| `langfuse-web` | Langfuse UI/API | `3000` |
| `langfuse-worker` | Langfuse background worker | none |
| `langfuse-postgres` | Langfuse relational store | internal |
| `langfuse-clickhouse` | Langfuse event store | internal |
| `langfuse-redis` | Langfuse queue/cache | internal |
| `langfuse-minio` | Langfuse object store | `9090`, `9091` |
| `mlflow` | MLflow tracking UI/API | `5000` |
| `mlflow-postgres` | MLflow backend store | internal |

`langfuse-minio-create-bucket` is an initialization task. It should normally exit with status `0`; it does not stay running.

## Observability Harness

The local observability harness combines:

| Tool | Tracks |
| --- | --- |
| Phoenix + OpenTelemetry | workflow traces, agent handoffs, tool calls |
| Langfuse | LLM and agent observations, token usage, cost, latency |
| MLflow | optional realtime workflow traces plus offline evaluation records |

### Required `.env` Values

```env
OBSERVABILITY_ENABLED=true
OBSERVABILITY_CAPTURE_INPUT_OUTPUT=false
OBSERVABILITY_ENVIRONMENT=local

OTEL_ENABLED=true
OTEL_GLOBAL_AUTO_INSTRUMENTATION_ENABLED=false
OTEL_HTTPX_INSTRUMENTATION_ENABLED=false
OTEL_REDIS_INSTRUMENTATION_ENABLED=false
OTEL_SQLALCHEMY_INSTRUMENTATION_ENABLED=false
OTEL_CELERY_INSTRUMENTATION_ENABLED=false
FASTAPI_OTEL_AUTO_INSTRUMENTATION_ENABLED=false
OPENINFERENCE_CREWAI_ENABLED=false
OPENINFERENCE_LITELLM_ENABLED=true
OPENINFERENCE_HIDE_INPUTS=true
OPENINFERENCE_HIDE_OUTPUTS=true
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://phoenix:6006/v1/traces
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
PHOENIX_PROJECT_NAME=cross-border-ai-dev

LANGFUSE_BASE_URL=http://langfuse-web:3000
LANGFUSE_PUBLIC_KEY=replace_with_langfuse_public_key
LANGFUSE_SECRET_KEY=replace_with_langfuse_secret_key

MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=cross-border-ai
MLFLOW_TRACING_ENABLED=false
```

Keep `OBSERVABILITY_CAPTURE_INPUT_OUTPUT=false` unless you have reviewed privacy implications. Phoenix is business-first by default: project workflow, agent, stage, LiteLLM, and Guardrails spans remain enabled, while FastAPI route, HTTPX, Redis, SQLAlchemy, Celery, and CrewAI auto-instrumentation stay off unless explicitly enabled. Turn on individual low-level `*_INSTRUMENTATION_ENABLED` flags only while debugging infrastructure noise. Search Phoenix by `workflow_type=support`, `job_id`, `conversation_id`, `guardrail_action`, or `guardrail_severity` to find the useful spans quickly. The observability layer redacts secret-like keys, emails, phone numbers, and raw customer handles before attaching metadata.

Keep the application workflow keys in a dedicated Langfuse project such as `cross-border-workflows`. Codex uses its separate user-level `~/.codex/langfuse.json`; both projects can share the same local Langfuse server at `http://localhost:3000`. The application exports one canonical OTel span for each project-defined observation and filters Langfuse exports to `workflow.*`, `stage.*`, `agent.*`, `guardrail_*`, and essential LiteLLM spans.

Set `MLFLOW_TRACING_ENABLED=true` to mirror the same project-defined workflow, stage, agent, evaluator, and Guardrails spans into MLflow in realtime. The project does not enable MLflow autologging by default, so MLflow stays low-noise: search the `cross-border-ai` experiment by span names such as `workflow.support`, `stage.*`, or guardrail span metadata such as `job_id`, `workflow_type`, `guardrail_action`, and `guardrail_severity`.

## Workflow Guardrails

Project-level guardrails run inside the same Docker image as FastAPI and Celery. The runtime installs these Guardrails Hub validators during image build:

- `hub://guardrails/secrets_present`
- `hub://guardrails/detect_pii`
- `hub://guardrails/regex_match`
- `hub://sainatha/prompt_injection_detector`
- `hub://guardrails/provenance_llm`
- `hub://guardrails/toxic_language`

There is no separate `guardrails-toxic` service. Streamlit submits workflows to FastAPI, FastAPI/Celery load `config/guardrails.yaml`, and all validator results are converted into the project's `GuardrailDecision` policy layer.

Validate the Docker runtime:

```powershell
docker compose exec fastapi guardrails hub list
docker compose exec celery_worker guardrails hub list
docker compose exec fastapi python -c "from services.workflow_guardrails import WorkflowGuardrailService; WorkflowGuardrailService().validate_runtime(smoke_toxic=True)"
```

Custom blocked terms live in `config/guardrails.yaml` under the `forbidden_terms` `regex_match` validators. Update the negative regex and rebuild/recreate FastAPI/Celery before testing through Streamlit.

### Langfuse Local Bootstrap

For local self-hosting, these values are self-generated; you do not need to apply for them:

```env
LANGFUSE_NEXTAUTH_URL=http://localhost:3000
LANGFUSE_NEXTAUTH_SECRET=replace_with_random_value
LANGFUSE_SALT=replace_with_random_value
LANGFUSE_ENCRYPTION_KEY=replace_with_64_hex_characters
LANGFUSE_POSTGRES_PASSWORD=replace_langfuse_postgres_password
LANGFUSE_CLICKHOUSE_PASSWORD=replace_langfuse_clickhouse_password
LANGFUSE_REDIS_PASSWORD=replace_langfuse_redis_password
LANGFUSE_S3_ACCESS_KEY_ID=minio
LANGFUSE_S3_SECRET_ACCESS_KEY=replace_minio_password
LANGFUSE_S3_BUCKET=langfuse
LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT=http://localhost:9090
LANGFUSE_INIT_USER_EMAIL=admin@example.com
LANGFUSE_INIT_USER_PASSWORD=replace_local_admin_password
```

Generate a 64-character encryption key:

```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_hex(32))"
```

Generate other random secrets:

```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

Set these before the first monitoring stack startup. If Langfuse was already initialized with old placeholders, reset the monitoring volumes:

```powershell
docker compose -f docker-compose.monitoring.yml down -v
docker compose -f docker-compose.monitoring.yml up -d
```

### MLflow Config

```env
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=cross-border-ai
MLFLOW_TRACING_ENABLED=true
MLFLOW_PROMPT_REGISTRY_ENABLED=true
MLFLOW_SUPPORT_PROMPT_ALIAS=production
MLFLOW_PROMPT_CACHE_DIR=artifacts/mlflow_prompt_cache
MLFLOW_SUPPORT_EVALUATION_DATASET_NAME=support-governance
MLFLOW_AUTOMATIC_EVALUATION_ENABLED=false
MLFLOW_GENAI_JUDGE_DEFAULT_MODEL=openai:/gpt-4o-mini
MLFLOW_GIT_VERSION_TRACKING_ENABLED=true
MLFLOW_ADMIN_USERNAME=admin
MLFLOW_ADMIN_PASSWORD=replace_mlflow_admin_password
MLFLOW_FLASK_SERVER_SECRET_KEY=replace_with_a_long_random_value
MLFLOW_POSTGRES_PASSWORD=replace_mlflow_postgres_password
```

The local MLflow image is pinned to `3.14.0` and runs with MLflow's official Basic Auth app. Open `http://localhost:5000` and sign in with `MLFLOW_ADMIN_USERNAME` and `MLFLOW_ADMIN_PASSWORD`. New non-admin users inherit read-only access by default.

If `http://localhost:5000` shows `Invalid Host header - possible DNS rebinding attack detected`, MLflow is rejecting the browser Host header. For local development, configure the MLflow service with allowed hosts such as:

```text
localhost:*,127.0.0.1:*,mlflow,mlflow:5000,fastapi,celery_worker
```

Then recreate only MLflow:

```powershell
docker compose -f docker-compose.monitoring.yml up -d --force-recreate mlflow
```

### MLflow Governance

Seed the official Prompt Registry entries, the `support-governance` Evaluation Dataset, and the official built-in scorers:

```powershell
docker compose exec fastapi python scripts/bootstrap_mlflow_governance.py
```

The bootstrap creates one immutable prompt per support agent and task and assigns the `production` alias only when an alias is missing. After bootstrap, edit prompts and move aliases from the MLflow Prompts UI. Support workers load `prompts:/<name>@production`; successful versions are cached under the ignored `artifacts/mlflow_prompt_cache` directory so a temporary MLflow outage does not select an ungoverned new prompt.

Run an official offline evaluation against the MLflow Evaluation Dataset:

```powershell
docker compose exec fastapi python scripts/run_mlflow_support_evaluation.py
```

This calls `mlflow.genai.evaluate()` with MLflow's official `RelevanceToQuery`, `Completeness`, `Safety`, `PIIDetection`, and `Guidelines` scorers. `ConversationalRoleAdherence` and `UserFrustration` are registered for conversation-aware evaluation.

MLflow 3.14 Automatic Evaluation requires an AI Gateway endpoint. This project intentionally keeps `MLFLOW_AUTOMATIC_EVALUATION_ENABLED=false` while the first governance phase does not use AI Gateway. Setting it to `true` without a `gateway:/<endpoint>` judge model fails validation instead of silently running a custom scheduler. Human approve, reject, and override actions are attached to the matching support trace through `mlflow.log_feedback()`; an edited approved response is attached through `mlflow.log_expectation()`.

## Common Test Commands

Compile important runtime files:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  .\runtime_config.py .\main.py .\orchestrator.py `
  .\celery_worker\tasks.py .\celery_worker\celery_app.py `
  .\utils\observability.py .\utils\workflow_progress.py `
  .\utils\tool_cache.py .\scripts\run_observability_evals.py
```

Run focused tests:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_observability.py `
  tests\test_llm_config.py `
  tests\test_tool_cache.py `
  tests\test_workflow_group.py `
  tests\test_workflow_route.py
```

Validate Docker config:

```powershell
docker compose -f docker-compose.yml -f docker-compose.monitoring.yml config --quiet
```

Inspect recent workflow jobs:

```powershell
docker compose exec postgres psql -U crossborder -d crossborder_ai -c "SELECT job_id, workflow_type, status, cache_hit, total_tokens, cost_usd, duration_seconds, updated_at FROM workflow_jobs ORDER BY updated_at DESC LIMIT 20;"
```

Inspect recent job events:

```powershell
docker compose exec postgres psql -U crossborder -d crossborder_ai -c "SELECT event_id, job_id, event_type, message, created_at FROM workflow_job_events ORDER BY event_id DESC LIMIT 30;"
```

## API Usage

### Authentication Headers

If `API_BEARER_TOKEN` is empty or missing, auth is disabled for local development. If it is set, protected endpoints require an authorization header.

PowerShell helper:

```powershell
$headers = @{}
if ($env:API_BEARER_TOKEN) {
  $headers.Authorization = ("{0} {1}" -f "Bearer", $env:API_BEARER_TOKEN)
}
```

### Submit A Workflow

All workflow submit requests may consume LLM/provider tokens unless served from cache. Use `metadata.bypass_cache=true` when you want to force a fresh run for testing.

```powershell
$headers = @{}
if ($env:API_BEARER_TOKEN) {
  $headers.Authorization = ("{0} {1}" -f "Bearer", $env:API_BEARER_TOKEN)
}

$body = @{
  workflow_type = "analytics"
  inputs = @{
    product_category = "Smart Home Security Cameras"
    target_markets = "US"
    date_range = "Last 30 Days"
    currency = "USD"
  }
  metadata = @{
    bypass_cache = $true
    trace_test = "readme_smoke"
  }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/workflow" `
  -Method POST `
  -ContentType "application/json" `
  -Headers $headers `
  -Body $body
```

### Poll A Job

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/workflow/replace-with-real-job-id" `
  -Headers $headers
```

### Read Job Events

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/workflow/replace-with-real-job-id/events" `
  -Headers $headers
```

### Workflow Group

`POST /api/v1/workflow-group` submits 2 to 7 existing workflows concurrently and returns a parent job. Child workflow results are aggregated under the parent.

```json
{
  "workflows": [
    {
      "name": "analytics_us",
      "workflow_type": "analytics",
      "inputs": {
        "product_category": "Smart Home Security Cameras",
        "target_markets": "US",
        "date_range": "Last 30 Days"
      }
    },
    {
      "name": "sales_us",
      "workflow_type": "sales_improvement",
      "inputs": {
        "product_category": "Smart Home Security Cameras",
        "target_markets": "US",
        "current_avg_conversion": "2.1%",
        "target_conversion": "3.5%",
        "date_range": "Last 60 Days"
      }
    }
  ]
}
```

### Workflow Route

`POST /api/v1/workflow-route/plan` returns a validated plan without submitting jobs.

`POST /api/v1/workflow-route` submits the plan as a parent `workflow_route` job. Nodes with no unmet dependencies run concurrently; dependent nodes run in later waves.

```json
{
  "goal": "Launch a smart camera campaign in the US and Germany with content, marketing, and schedule recommendations.",
  "context": {
    "product_category": "Smart Home Security Cameras",
    "product_usp": "Privacy-first 4K AI motion detection",
    "target_markets": "US, Germany",
    "budget": "$15000 USD",
    "target_languages": ["en-US", "de"],
    "preferred_launch_window": "2026-07-01 to 2026-08-15"
  },
  "preferred_workflows": ["marketing", "content", "scheduler"]
}
```

## Workflow Examples

These examples use the generic `POST /api/v1/workflow` endpoint. They assume `$headers` has already been created as shown in [API Usage](#api-usage).

### Content Creation

```powershell
$body = @{
  workflow_type = "content"
  inputs = @{
    subject = "Sustainable Activewear for Cold Climates"
    product_category = "Eco-Friendly Winter Sportswear"
    product_features = "Recycled thermal shell, wind-resistant construction, moisture-wicking base layer."
    target_markets = "Germany, Japan, Canada"
    target_languages = @("de", "ja", "en")
    platforms = @("Instagram", "LinkedIn", "X")
    brand_voice = "Premium, practical, sustainability-minded"
    brand_name = "NorthPeak Layers"
    product_url = "https://example.com/products/sustainable-activewear"
    primary_keywords = @("thermal activewear", "winter training layer", "recycled sportswear")
    generate_reddit_geo = $false
    generate_visual_assets = $false
  }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/workflow" -Method POST -ContentType "application/json" -Headers $headers -Body $body
```

### Marketing Campaign

```powershell
$body = @{
  workflow_type = "marketing"
  inputs = @{
    product_category = "Smart Home Security Cameras"
    product_usp = "AI-powered motion detection, 4K resolution, privacy-first cloud storage"
    target_markets = "US, UK, Germany, Japan"
    target_languages = @("en-US", "en-GB", "de", "ja")
    budget = "$15,000 USD"
  }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/workflow" -Method POST -ContentType "application/json" -Headers $headers -Body $body
```

### Analytics

```powershell
$body = @{
  workflow_type = "analytics"
  inputs = @{
    product_category = "Smart Home Security Cameras"
    target_markets = "US, UK, Germany, Japan"
    date_range = "Last 30 Days"
    currency = "USD"
  }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/workflow" -Method POST -ContentType "application/json" -Headers $headers -Body $body
```

### Sales Improvement

```powershell
$body = @{
  workflow_type = "sales_improvement"
  inputs = @{
    product_category = "Smart Home Security Cameras"
    target_markets = "US, EU, Japan"
    current_avg_conversion = "2.1%"
    target_conversion = "3.5%"
    date_range = "Last 60 Days"
  }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/workflow" -Method POST -ContentType "application/json" -Headers $headers -Body $body
```

### Business Development

```powershell
$body = @{
  workflow_type = "bizdev"
  inputs = @{
    product_category = "Smart Home Security Cameras"
    partnership_type = "Regional Distributors & Retail Partners"
    target_markets = "Germany, Japan, Canada"
    target_languages = @("de", "ja", "en")
    key_decision_maker_roles = "Head of Procurement, Channel Manager"
  }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/workflow" -Method POST -ContentType "application/json" -Headers $headers -Body $body
```

### Scheduler

```powershell
$body = @{
  workflow_type = "scheduler"
  inputs = @{
    event_type = "Product Launch & Promotional Campaign"
    target_markets = "US, UK, Germany, Japan"
    event_list = "Smart Camera Launch, Early Access Sale, Influencer Drop, Post-Launch Retargeting"
    preferred_launch_window = "2026-07-01 to 2026-08-15"
  }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/workflow" -Method POST -ContentType "application/json" -Headers $headers -Body $body
```

### Customer Support

```powershell
$body = @{
  workflow_type = "support"
  inputs = @{
    customer = "GlobalTech Solutions"
    person = "Maria Chen"
    inquiry = "Our bulk order is delayed. We need it by Friday for a product launch. What are the expedited shipping options and compensation policy?"
    ticket_id = "TKT-LOCAL-001"
    order_id = "ORDER-8842"
    item_sku = "CAM-4K-PRO"
    region = "US"
  }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/workflow" -Method POST -ContentType "application/json" -Headers $headers -Body $body
```

## Admin Dashboard

The Streamlit dashboard is a lightweight local UI for submitting workflows, viewing job status, inspecting usage metadata, and browsing job events.

Start it from the virtual environment while the Docker backend is running:

```powershell
streamlit run .\admin_dashboard.py
```

If `API_BEARER_TOKEN` is set, the dashboard reads it from `.env` by default. You can also paste the token into the sidebar.

Opening the dashboard does not run a workflow. Submitting a workflow from the page may consume tokens.

## Support Inbox / Gmail / WhatsApp

### Support RAG Knowledge Base

The Support workflow searches local knowledge before using web tools. Policy documents live in:

```text
docs/knowledge_base/
```

Supported local knowledge formats include `.md`, `.txt`, and `.pdf`. PDF files are parsed with `pdfplumber`.

The local RAG tool uses deterministic local embeddings and does not call OpenAI or an external vector database.

### Support Inbox Endpoints

```text
GET  /api/v1/support/conversations
GET  /api/v1/support/conversations/{conversation_id}
POST /api/v1/support/conversations/{conversation_id}/assign
POST /api/v1/support/conversations/{conversation_id}/approve-send
```

`approve-send` dispatches by conversation channel. It makes no provider call while `WHATSAPP_SEND_ENABLED=false` or `GMAIL_SEND_ENABLED=false`.

Escalated conversations are blocked from automated approval sending and must be handled manually.

### Gmail

Gmail endpoints:

```text
POST /api/v1/channels/gmail/watch
POST /api/v1/channels/gmail/pubsub
POST /api/v1/channels/gmail/sync
POST /api/v1/channels/gmail/sync-latest
```

Recommended Gmail config:

```env
GMAIL_CLIENT_ID=your_google_oauth_client_id
GMAIL_CLIENT_SECRET=your_google_oauth_client_secret
GMAIL_REFRESH_TOKEN=your_google_refresh_token
GMAIL_ACCESS_TOKEN=
GMAIL_SENDER_EMAIL=support@example.com
GMAIL_SEND_ENABLED=false
GMAIL_WATCH_TOPIC_NAME=projects/your-project/topics/gmail-support
GMAIL_WATCH_LABEL_IDS=INBOX
GMAIL_SYNC_ENABLED=false
```

For local testing:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/channels/gmail/sync-latest" `
  -Method POST `
  -ContentType "application/json" `
  -Headers $headers `
  -Body '{"max_results": 5}'
```

Long-running Gmail usage should rely on refresh tokens, not short-lived access tokens. If your Google Cloud OAuth app is `External` and still in `Testing`, Google may expire refresh tokens after 7 days for Gmail scopes.

### WhatsApp

WhatsApp Cloud API webhook verification:

```text
GET /api/v1/channels/whatsapp/webhook?hub.mode=subscribe&hub.verify_token=<WHATSAPP_VERIFY_TOKEN>&hub.challenge=<challenge>
```

Inbound WhatsApp webhooks:

```text
POST /api/v1/channels/whatsapp/webhook
```

YCloud inbound WhatsApp webhooks:

```text
POST /api/v1/channels/ycloud/webhook
```

Recommended short-term local config:

```env
WHATSAPP_PROVIDER=ycloud
WHATSAPP_SEND_ENABLED=false
YCLOUD_API_KEY=optional_ycloud_api_key
YCLOUD_WHATSAPP_FROM=optional_sender_phone_e164
YCLOUD_WABA_ID=optional_whatsapp_business_account_id
YCLOUD_BASE_URL=https://api.ycloud.com/v2
YCLOUD_WEBHOOK_SECRET=optional_ycloud_webhook_secret
```

Meta Cloud API config:

```env
WHATSAPP_PROVIDER=meta
WHATSAPP_ACCESS_TOKEN=optional_whatsapp_cloud_api_access_token
WHATSAPP_PHONE_NUMBER_ID=optional_whatsapp_business_phone_number_id
WHATSAPP_BUSINESS_ACCOUNT_ID=optional_whatsapp_business_account_id
WHATSAPP_VERIFY_TOKEN=choose_a_webhook_verification_token
WHATSAPP_APP_SECRET=optional_meta_app_secret_for_x_hub_signature_validation
WHATSAPP_SEND_ENABLED=false
WHATSAPP_GRAPH_API_VERSION=v23.0
```

When `WHATSAPP_APP_SECRET` is configured, the webhook validates `X-Hub-Signature-256`.

WhatsApp approval sending respects Meta's 24-hour customer service window. Outside the active window, `approve-send` sends a pre-approved template instead of free-form text.

### Support Session State

Redis session state is stored under:

```text
support:session:{session_id}
```

Defaults:

```env
SUPPORT_SESSION_REDIS_URL=redis://localhost:6379/2
SUPPORT_SESSION_TTL_SECONDS=86400
SUPPORT_SESSION_HISTORY_LIMIT=20
```

PostgreSQL `support_conversations` and `support_messages` remain authoritative. Redis failures degrade gracefully to database-backed history.

### Optional Intent Classifier Training

The runtime uses the keyword router by default. Optional offline training is available after installing `requirements-ml.txt`:

```powershell
python .\scripts\train_intent_classifier.py --dry-run
python .\scripts\train_intent_classifier.py --train
```

Generated classifier artifacts are written under:

```text
artifacts/intent_classifier_v1/
```

## Advanced Runtime Features

### Strict Input Validation

Workflow requests are validated before a job is submitted. Each `workflow_type` has its own required input schema in `models.py`. Missing fields, empty strings, wrong list types, or unexpected extra fields return a validation error before any CrewAI workflow can consume tokens.

### Usage Tracking

The orchestrator records:

```text
prompt_tokens
completion_tokens
total_tokens
cost_usd
duration_seconds
usage_metrics
```

`duration_seconds` is always measured by the orchestrator/worker. Token fields depend on whether the current CrewAI result exposes usage metrics.

Cost is calculated from optional rates:

```env
OPENAI_INPUT_COST_PER_1M_TOKENS=0
OPENAI_OUTPUT_COST_PER_1M_TOKENS=0
```

Leave them at `0` to track tokens and duration without estimating dollars.

### Workflow Result Cache

Completed workflow results are reused from PostgreSQL when a new request has the same:

```text
workflow_type
validated inputs
runtime configuration fingerprint
```

A cache hit creates a new completed job with:

```text
cache_hit=true
source_job_id=<original job id>
token/cost fields set to 0
```

Bypass cache for one request:

```json
{
  "metadata": {
    "bypass_cache": true
  }
}
```

### Tool Cache

`TOOL_CACHE_ENABLED=true` caches read-only external I/O such as Serper search, scraping, commerce metrics reads, PIM lookup, support knowledge search, and read-only ad-platform validation calls.

```env
TOOL_CACHE_ENABLED=true
TOOL_CACHE_BACKEND=redis_postgres
TOOL_CACHE_REDIS_URL=
TOOL_CACHE_TTL_SECONDS=86400
TOOL_CACHE_DB_ENABLED=true
TOOL_CACHE_MAX_VALUE_BYTES=1048576
```

Redis is the hot cache. PostgreSQL is the durable mirror/fallback. If `TOOL_CACHE_REDIS_URL` is empty, it falls back to `CELERY_BROKER_URL`.

Workflow result cache and tool cache are separate.

### Async Workflow Execution

```env
WORKFLOW_ASYNC_EXECUTION_ENABLED=true
```

Analytics runs performance analysis and market research concurrently after data collection.

Sales Improvement runs CRO recommendations and pricing optimization concurrently after funnel analysis.

Content and Marketing also include per-language/per-market concurrency:

```env
CONTENT_LANGUAGE_CONCURRENCY=4
MARKETING_MARKET_CONCURRENCY=4
```

### Workflow Router

```env
WORKFLOW_ROUTER_ENABLED=true
WORKFLOW_ROUTER_LLM_FALLBACK_ENABLED=true
WORKFLOW_ROUTER_CONFIDENCE_THRESHOLD=0.75
WORKFLOW_ROUTER_MAX_WORKFLOWS=7
WORKFLOW_ROUTER_LLM_PROFILE=
```

The router uses deterministic rules first and optional LLM JSON fallback for ambiguous goals. Existing crews remain deterministic worker units.

### Content Image Assets

Image generation only runs when request inputs set `generate_visual_assets=true`.

```env
CONTENT_IMAGE_MODEL=gpt-image-2
CONTENT_IMAGE_SCORING_MODEL=gpt-4o-mini
CONTENT_IMAGE_ARTIFACT_DIR=artifacts/content_creation
```

Generated files are stored under the ignored `artifacts/` tree. If no image-capable API key is configured, the workflow still completes and marks image generation as skipped.

### Analytics Deep Read

```env
SERPER_DEEP_READ_ENABLED=false
SERPER_DEEP_READ_MAX_PAGES=3
SERPER_DEEP_READ_CONCURRENCY=5
SERPER_DEEP_READ_TIMEOUT_SECONDS=10
SERPER_DEEP_READ_MAX_CHARS=4000
```

When enabled, Analytics competitive research reads top Serper result pages and passes source excerpts into the market research task.

## Troubleshooting

### FastAPI Does Not See New `.env` Values

Recreate the app containers:

```powershell
docker compose up -d --force-recreate fastapi celery_worker flower
```

### Langfuse Keys Or Passwords Changed After First Startup

Langfuse initializes project/user data into volumes. If local development data can be discarded, reset the monitoring volumes:

```powershell
docker compose -f docker-compose.monitoring.yml down -v
docker compose -f docker-compose.monitoring.yml up -d
```

### `langfuse-minio-create-bucket` Is Not Running

That is normal. It is a one-shot initialization container. `Exited (0)` means it completed successfully.

Check it with:

```powershell
docker compose -f docker-compose.monitoring.yml ps -a
```

### MLflow Shows Invalid Host Header

MLflow 3.5+ protects against DNS rebinding by validating Host headers. For local access at `http://localhost:5000`, the MLflow service must allow localhost with ports:

```text
localhost:*,127.0.0.1:*,mlflow,mlflow:5000,fastapi,celery_worker
```

Recreate MLflow after changing the compose command:

```powershell
docker compose -f docker-compose.monitoring.yml up -d --force-recreate mlflow
```

### Workflow Is Served From Cache But You Need A Fresh Trace

Set request metadata:

```json
{
  "metadata": {
    "bypass_cache": true
  }
}
```

### Celery Jobs Are Not Moving

Check Redis, worker, and Flower:

```powershell
docker compose ps
docker compose logs celery_worker --tail=100
docker compose logs redis --tail=100
```

Flower UI:

```text
http://localhost:5555
```

### Provider Credentials Are Missing

Most workflows degrade to clearly marked fallback data when optional provider credentials are missing. For business decisions, validate outputs with real provider data before acting on them.

### No Observability Data Appears

Check these in order:

1. `OBSERVABILITY_ENABLED=true` is in `.env`.
2. FastAPI and Celery containers were recreated after `.env` changed.
3. Phoenix is running at `http://localhost:6006`.
4. `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://phoenix:6006/v1/traces`.
5. A fresh workflow was submitted with `metadata.bypass_cache=true`.

## Project Structure

```text
Cross-BorderAIProject/
|-- api/
|   `-- routes.py
|-- celery_worker/
|   |-- celery_app.py
|   `-- tasks.py
|-- config/
|   |-- analytics/
|   |-- business_development/
|   |-- content/
|   |-- marketing/
|   |-- sales_improvement/
|   |-- scheduler/
|   `-- support/
|-- crews/
|   |-- analytics_crew.py
|   |-- bizdev_crew.py
|   |-- content_crew.py
|   |-- marketing_crew.py
|   |-- sales_improvement_crew.py
|   |-- scheduler_crew.py
|   `-- support_crew.py
|-- services/
|-- tools/
|   |-- custom/
|   `-- integrations/
|-- utils/
|   |-- observability.py
|   |-- workflow_engine.py
|   |-- workflow_progress.py
|   |-- tool_cache.py
|   |-- model_tiering.py
|   `-- crew_memory.py
|-- tests/
|-- docs/
|-- admin_dashboard.py
|-- database.py
|-- db_models.py
|-- docker-compose.yml
|-- docker-compose.monitoring.yml
|-- Dockerfile
|-- job_store.py
|-- main.py
|-- models.py
|-- orchestrator.py
|-- requirements.txt
|-- runtime_config.py
`-- support_inbox.py
```

## Archived Notes

Original source notes and design references are kept under:

```text
docs/original_code_notes/
docs/high_level_architecture.txt
docs/design_assets/
```

They are implementation references only. The runnable code lives in `api/`, `celery_worker/`, `config/`, `crews/`, `services/`, `tools/`, `utils/`, and the root runtime modules.
