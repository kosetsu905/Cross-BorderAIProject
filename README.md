# Cross-Border E-Commerce AI Suite

CrewAI-based workflow suite for cross-border e-commerce operations. The project structure follows the design document's recommended layout: shared FastAPI entrypoint, master orchestrator, workflow-specific crews, YAML configs, and shared tools.

## Project Structure

```text
Cross-BorderAIProject/
|-- config/
|   |-- business_development/
|   |   |-- agents.yaml
|   |   `-- tasks.yaml
|   |-- marketing/
|   |   |-- agents.yaml
|   |   `-- tasks.yaml
|   |-- content/
|   |   |-- agents.yaml
|   |   `-- tasks.yaml
|   |-- support/
|   |   |-- agents.yaml
|   |   `-- tasks.yaml
|   |-- analytics/
|   |   |-- agents.yaml
|   |   `-- tasks.yaml
|   |-- scheduler/
|   |   |-- agents.yaml
|   |   `-- tasks.yaml
|   `-- sales_improvement/
|       |-- agents.yaml
|       `-- tasks.yaml
|-- tools/
|   |-- base/
|   |-- integrations/
|   |   `-- cross_platform_ads_tools.py
|   `-- custom/
|       |-- analytics_tools.py
|       |-- bizdev_tools.py
|       |-- marketing_tools.py
|       |-- sales_tools.py
|       `-- scheduler_tools.py
|-- crews/
|   |-- analytics_crew.py
|   |-- bizdev_crew.py
|   |-- content_crew.py
|   |-- marketing_crew.py
|   |-- scheduler_crew.py
|   |-- sales_improvement_crew.py
|   `-- support_crew.py
|-- api/
|   `-- routes.py
|-- celery_worker/
|   |-- celery_app.py
|   `-- tasks.py
|-- admin_dashboard.py
|-- docs/
|   |-- design_assets/
|   `-- original_code_notes/
|-- Dockerfile
|-- docker-compose.yml
|-- models.py
|-- orchestrator.py
|-- main.py
|-- requirements.txt
`-- .env
```

## Archived Original Notes

The original per-folder `code.txt` files have been moved into:

```text
docs/original_code_notes/
```

They are kept as source notes and implementation references. The runnable code is now in `config/`, `tools/`, `crews/`, `api/`, `models.py`, `orchestrator.py`, and `main.py`.

The original high-level architecture note and design document are stored in:

```text
docs/high_level_architecture.txt
docs/CrawAI Enterprise Solution high Level Design Document.docx
docs/design_assets/
```

## Registered Workflows

```text
analytics
bizdev
marketing
content
scheduler
sales_improvement
support
```

## Setup

```powershell
cd D:\Cross-BorderAIProject
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Optional ML training dependencies are kept separate from the runtime install:

```powershell
python -m pip install -r requirements-ml.txt
```

The project expects a root-level `.env` file.

Minimum required for running CrewAI workflows:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL_NAME=gpt-4o-mini
CREWAI_MEMORY_ENABLED=false
OPENAI_INPUT_COST_PER_1M_TOKENS=0
OPENAI_OUTPUT_COST_PER_1M_TOKENS=0
```

OpenAI-compatible providers can be configured with the generic `LLM_*` variables. These take precedence over the legacy `OPENAI_*` variables while preserving backwards compatibility:

```env
LLM_PROVIDER=openrouter
LLM_API_KEY=your_openrouter_key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL_NAME=openai/gpt-4o-mini
LLM_DISABLE_REASONING=false
CREWAI_MEMORY_ENABLED=false
```

Customer service can switch models with server-side named LLM profiles, so operators can choose OpenAI or OpenRouter models without editing the active global model variables:

```env
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
LLM_PROFILES_JSON={"openai_gpt4o_mini":{"llm_provider":"openai","llm_model_name":"gpt-4o-mini","llm_api_key_env":"OPENAI_API_KEY"},"openrouter_gpt4o_mini":{"llm_provider":"openrouter","llm_model_name":"openai/gpt-4o-mini","llm_base_url":"https://openrouter.ai/api/v1","llm_api_key_env":"OPENROUTER_API_KEY","llm_disable_reasoning":false}}
SUPPORT_LLM_PROFILE=openai_gpt4o_mini
SUPPORT_QA_MODE=full_llm
```

`SUPPORT_LLM_PROFILE` applies to Gmail, WhatsApp, `/api/v1/service/inquiry`, and normal `support` workflow jobs. A single request can override it by passing only the profile name:

```json
{
  "workflow_type": "support",
  "inputs": {
    "customer": "Maria",
    "person": "Maria",
    "inquiry": "I need help with a return."
  },
  "provider_credentials": {
    "llm_profile": "openrouter_gpt4o_mini"
  }
}
```

For `/api/v1/service/inquiry`, pass the same selector as a top-level field:

```json
{
  "customer": "Maria",
  "inquiry": "The item arrived damaged. Can I return it?",
  "channel": "whatsapp",
  "llm_profile": "openrouter_gpt4o_mini"
}
```

The same endpoint also accepts request-scoped provider credentials without storing them in support inputs:

```json
{
  "customer": "Maria",
  "inquiry": "Which camera works with HomeKit?",
  "channel": "whatsapp",
  "provider_credentials": {
    "serper_api_key": "request_scoped_serper_key",
    "support_serper_pre_sales_enabled": true
  }
}
```

For faster Qwen3-14B post-sales support, set `SUPPORT_QA_MODE=adaptive_fast`. Low-risk post-sales tickets will skip the second LLM QA pass and use deterministic/Pydantic validation plus the existing RMA and compliance guards. High-risk tickets still run the full LLM QA path.

Use `.env.example` as a safe template. Keep real `.env` values local and rotate any key that has been shared, pasted into logs, or committed.

Reasoning/thinking models on OpenRouter, such as Qwen3 and DeepSeek R1 variants, are automatically run with reasoning output disabled so CrewAI can parse normal structured responses. Set `LLM_DISABLE_REASONING=true` to force this compatibility mode for another model.

`CREWAI_MEMORY_ENABLED` defaults to `false`. Turn it on only after your OpenAI account/key can use the embeddings endpoint required by CrewAI memory.
`OPENAI_INPUT_COST_PER_1M_TOKENS` and `OPENAI_OUTPUT_COST_PER_1M_TOKENS` are optional cost-estimation rates. Leave them at `0` to track tokens and duration without estimating dollars.

Optional shared services:

```env
API_BEARER_TOKEN=optional_local_api_token
SERPER_API_KEY=optional_serper_key
SUPPORT_SERPER_PRE_SALES_ENABLED=false
SUPPORT_SERPER_ORDER_FULFILLMENT_ENABLED=false
SUPPORT_SERPER_POST_SALES_ENABLED=false
WORKFLOW_BACKEND=local
DATABASE_URL=postgresql+psycopg://crossborder:crossborder@localhost:5432/crossborder_ai
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
CELERY_RETRY_BASE_DELAY_SECONDS=30
CELERY_RETRY_MAX_DELAY_SECONDS=300
WORKFLOW_RESULT_CACHE_ENABLED=true
WORKFLOW_RESULT_CACHE_TTL_SECONDS=3600
CONTENT_LANGUAGE_CONCURRENCY=4
CONTENT_IMAGE_MODEL=gpt-image-2
CONTENT_IMAGE_SCORING_MODEL=gpt-4o-mini
CONTENT_IMAGE_ARTIFACT_DIR=artifacts/content_creation
MARKETING_MARKET_CONCURRENCY=4
SERPER_DEEP_READ_ENABLED=false
SERPER_DEEP_READ_MAX_PAGES=3
SERPER_DEEP_READ_CONCURRENCY=5
SERPER_DEEP_READ_TIMEOUT_SECONDS=10
SERPER_DEEP_READ_MAX_CHARS=4000
```

`WORKFLOW_BACKEND=local` keeps the current lightweight in-process background execution. Use `WORKFLOW_BACKEND=celery` when Redis and a Celery worker are running and you want workflow jobs to be handled by the message broker.
PostgreSQL is used for persistent local-backend job state. The app creates the `workflow_jobs` table on startup.
Runtime configuration and secrets are centralized in `runtime_config.py`. FastAPI/Celery load `.env` once, pass a `config_context` into the orchestrator and crews, and provider tools receive credentials through constructors instead of reading global environment variables directly.
If `API_BEARER_TOKEN` is set, workflow submit and polling endpoints require `Authorization: Bearer <token>`. `/health` stays public for local and container health checks. If `API_BEARER_TOKEN` is empty or missing, auth is disabled for local development.
Workflow result cache is enabled by default. `WORKFLOW_RESULT_CACHE_TTL_SECONDS` controls how long a completed result can be reused.
Content Creation runs one shared research/strategy task and then generates requested languages in parallel. `CONTENT_LANGUAGE_CONCURRENCY` controls the maximum number of language-generation workers.
Content Creation also produces multimodal localization specs, video storyboard guidance, multi-engine SEO metadata, hreflang tags, JSON-LD, and cultural risk notes. Set request input `generate_visual_assets=true` to call the OpenAI Image API; generated files are stored under `CONTENT_IMAGE_ARTIFACT_DIR`, which defaults to ignored `artifacts/content_creation`. Without an OpenAI API key, the workflow still completes and marks image generation as `skipped_missing_credentials`.
Marketing runs one shared strategy/channel-planning task and then generates market-specific creative/compliance packages in parallel. `MARKETING_MARKET_CONCURRENCY` controls the maximum number of market workers.
Analytics competitive research can optionally deep-read Serper result URLs. When `SERPER_DEEP_READ_ENABLED=true`, `CompetitorBenchmarkTool` reads up to `SERPER_DEEP_READ_MAX_PAGES` pages per market with `SERPER_DEEP_READ_CONCURRENCY` workers and passes source excerpts to CrewAI.
Customer Service keeps external Serper search off by default for faster support responses. Set `SUPPORT_SERPER_PRE_SALES_ENABLED=true`, `SUPPORT_SERPER_ORDER_FULFILLMENT_ENABLED=true`, or `SUPPORT_SERPER_POST_SALES_ENABLED=true` together with `SERPER_API_KEY` to enable live search only for that stage.

Optional workflow data providers:

```env
# Business Development lead enrichment. Without these, BizDev uses development fallback lead data.
CRUNCHBASE_API_KEY=optional_crunchbase_key
APOLLO_API_KEY=optional_apollo_key

# Legacy generic platform token. Prefer the explicit Shopify/Amazon settings below.
ECOM_API_TOKEN=optional_ecommerce_platform_token

# Legacy generic CRM/platform token. Prefer the explicit Shopify/Amazon settings below.
CRM_API_TOKEN=optional_crm_or_platform_token

# Commerce order data for Analytics and Sales Improvement.
# Shopify uses the Admin Orders API. The token needs read_orders scope.
SHOPIFY_STORE_DOMAIN=your-store.myshopify.com
SHOPIFY_ADMIN_ACCESS_TOKEN=optional_shopify_admin_access_token
SHOPIFY_API_VERSION=2025-07

# Amazon uses an SP-API Orders-compatible endpoint.
# Direct SP-API calls normally require AWS SigV4 signing, so this can point to a signed proxy/client endpoint.
AMAZON_SP_API_ENDPOINT=https://sellingpartnerapi-na.amazon.com
AMAZON_SP_API_ACCESS_TOKEN=optional_amazon_sp_api_access_token
AMAZON_MARKETPLACE_IDS=ATVPDKIKX0DER

# Scheduler holiday/timezone provider. Without this, Scheduler uses development fallback calendar data.
HOLIDAY_API_KEY=optional_holiday_provider_key

# Support RAG knowledge base. Defaults to docs/knowledge_base.
SUPPORT_KNOWLEDGE_DIR=docs/knowledge_base

# Gmail delivery for Support replies. GMAIL_SEND_ENABLED defaults to false.
# Sending needs gmail.send. Omni-channel inbound sync/watch also needs gmail.readonly or gmail.modify.
# GMAIL_ACCESS_TOKEN is still supported for short local tests, but the refresh-token settings below
# are preferred because the app can mint fresh access tokens automatically.
GMAIL_ACCESS_TOKEN=optional_short_lived_gmail_oauth_access_token
GMAIL_CLIENT_ID=optional_google_oauth_client_id
GMAIL_CLIENT_SECRET=optional_google_oauth_client_secret
GMAIL_REFRESH_TOKEN=optional_google_oauth_refresh_token
GMAIL_SENDER_EMAIL=support@example.com
GMAIL_SEND_ENABLED=false
GMAIL_WATCH_TOPIC_NAME=projects/your-project/topics/gmail-support
GMAIL_WATCH_LABEL_IDS=INBOX
GMAIL_SYNC_ENABLED=false

# WhatsApp provider for omni-channel Support. WHATSAPP_SEND_ENABLED defaults to false.
# Short-term testing defaults to YCloud; set WHATSAPP_PROVIDER=meta when Meta Cloud API is available.
WHATSAPP_PROVIDER=ycloud
WHATSAPP_ACCESS_TOKEN=optional_whatsapp_cloud_api_access_token
WHATSAPP_PHONE_NUMBER_ID=optional_whatsapp_business_phone_number_id
WHATSAPP_BUSINESS_ACCOUNT_ID=optional_whatsapp_business_account_id
WHATSAPP_VERIFY_TOKEN=choose_a_webhook_verification_token
WHATSAPP_APP_SECRET=optional_meta_app_secret_for_x_hub_signature_validation
WHATSAPP_SEND_ENABLED=false
WHATSAPP_GRAPH_API_VERSION=v23.0

# Optional YCloud WhatsApp provider for validation/gray rollout.
YCLOUD_API_KEY=optional_ycloud_api_key
YCLOUD_WHATSAPP_FROM=optional_sender_phone_e164
YCLOUD_WABA_ID=optional_whatsapp_business_account_id
YCLOUD_BASE_URL=https://api.ycloud.com/v2
YCLOUD_WEBHOOK_SECRET=optional_ycloud_webhook_secret

# Optional PIM connector for pre-sales product knowledge. Missing credentials fall back to mock data.
PIM_BACKEND=akeneo
PIM_AKENEO_BASE_URL=optional_akeneo_base_url
PIM_AKENEO_API_KEY=optional_akeneo_api_key
PIM_PLYTIX_BASE_URL=optional_plytix_base_url
PIM_PLYTIX_API_KEY=optional_plytix_api_key
PIM_CUSTOM_BASE_URL=optional_custom_pim_base_url
PIM_CUSTOM_API_KEY=optional_custom_pim_api_key

SUPPORT_HANDOFF_WEBHOOK_URL=optional_slack_or_support_queue_webhook
SUPPORT_SESSION_REDIS_URL=redis://localhost:6379/2
SUPPORT_SESSION_TTL_SECONDS=86400
SUPPORT_SESSION_HISTORY_LIMIT=20

# Compatibility aliases accepted by runtime_config.py:
# WHATSAPP_TOKEN -> WHATSAPP_ACCESS_TOKEN
# WHATSAPP_PHONE_ID -> WHATSAPP_PHONE_NUMBER_ID
# SLACK_WEBHOOK_URL -> SUPPORT_HANDOFF_WEBHOOK_URL

# Marketing ad platform integrations. Without these, Marketing uses development fallback platform data.
GOOGLE_ADS_DEVELOPER_TOKEN=optional_google_ads_developer_token
GOOGLE_ADS_ACCESS_TOKEN=optional_google_ads_access_token
GOOGLE_ADS_CUSTOMER_ID=optional_google_ads_customer_id
META_ACCESS_TOKEN=optional_meta_access_token
META_AD_ACCOUNT_ID=optional_meta_ad_account_id
META_PAGE_ID=optional_meta_page_id
TIKTOK_ACCESS_TOKEN=optional_tiktok_access_token
TIKTOK_ADVERTISER_ID=optional_tiktok_advertiser_id
```

Analytics and Sales Improvement can fetch order-derived metrics from Shopify Admin API or an Amazon SP-API Orders-compatible endpoint. If these provider settings are missing or a provider call fails, the workflow falls back to clearly marked development sample data.
Current runnable code does not use calendar-provider tokens mentioned in archived notes yet. Those belong to future integration work unless a corresponding tool is implemented under `tools/`.

For future multi-tenant usage, provider credentials can also be supplied per request with `provider_credentials` instead of using the server-wide `.env` values. These request-scoped credentials are merged into the workflow runtime context and are not stored in the job `inputs` history.

Example shape:

```json
{
  "workflow_type": "marketing",
  "inputs": {
    "product_category": "Smart Home Security Cameras",
    "product_usp": "AI-powered motion detection, 4K resolution, privacy-first cloud storage",
    "target_markets": "US, UK, Germany, Japan",
    "budget": "$15,000 USD"
  },
  "provider_credentials": {
    "google_ads_developer_token": "request_scoped_google_developer_token",
    "google_ads_access_token": "request_scoped_google_access_token",
    "google_ads_customer_id": "request_scoped_customer_id",
    "shopify_store_domain": "request-scoped-store.myshopify.com",
    "shopify_admin_access_token": "request_scoped_shopify_token",
    "amazon_sp_api_endpoint": "https://sellingpartnerapi-na.amazon.com",
    "amazon_sp_api_access_token": "request_scoped_amazon_access_token",
    "amazon_marketplace_ids": "ATVPDKIKX0DER",
    "serper_api_key": "request_scoped_serper_key",
    "content_image_model": "gpt-image-2",
    "content_image_scoring_model": "gpt-4o-mini",
    "content_image_artifact_dir": "artifacts/content_creation",
    "support_serper_pre_sales_enabled": false,
    "support_serper_order_fulfillment_enabled": false,
    "support_serper_post_sales_enabled": false,
    "gmail_access_token": "request_scoped_gmail_access_token",
    "gmail_client_id": "request_scoped_google_oauth_client_id",
    "gmail_client_secret": "request_scoped_google_oauth_client_secret",
    "gmail_refresh_token": "request_scoped_google_oauth_refresh_token",
    "gmail_sender_email": "support@example.com",
    "gmail_send_enabled": true,
    "gmail_watch_topic_name": "projects/your-project/topics/gmail-support",
    "gmail_watch_label_ids": "INBOX",
    "gmail_sync_enabled": false,
    "whatsapp_access_token": "request_scoped_whatsapp_access_token",
    "whatsapp_phone_number_id": "request_scoped_phone_number_id",
    "whatsapp_business_account_id": "request_scoped_waba_id",
    "whatsapp_verify_token": "request_scoped_webhook_verify_token",
    "whatsapp_app_secret": "request_scoped_meta_app_secret",
    "whatsapp_send_enabled": false,
    "meta_access_token": "request_scoped_meta_token",
    "meta_ad_account_id": "request_scoped_meta_account_id",
    "tiktok_access_token": "request_scoped_tiktok_token",
    "tiktok_advertiser_id": "request_scoped_tiktok_advertiser_id"
  }
}
```

For a production SaaS implementation, prefer passing a `tenant_id` and loading encrypted provider credentials from a secrets vault. Passing credentials directly in the API request is useful for local development and integration testing, but the Celery broker still receives the task payload.

Support can optionally send the final `drafted_response` through Gmail after the workflow completes. Gmail can use either a short-lived `GMAIL_ACCESS_TOKEN` or the preferred `GMAIL_CLIENT_ID` + `GMAIL_CLIENT_SECRET` + `GMAIL_REFRESH_TOKEN` settings to mint fresh access tokens automatically. Gmail delivery is skipped for escalated support tickets.
Gmail and WhatsApp support can also use the omni-channel inbox flow instead of workflow auto-send. Channel inbound messages are stored, submitted as `support` workflow jobs, and exposed for review. `POST /api/v1/support/conversations/{conversation_id}/approve-send` sends only when the channel's send flag is enabled; otherwise it returns `disabled` and makes no provider API call.

## No-Token Checks

These checks do not run CrewAI jobs and should not consume OpenAI API tokens.

```powershell
python -m py_compile .\main.py .\models.py .\runtime_config.py .\database.py .\db_models.py .\job_store.py .\orchestrator.py .\support_inbox.py .\api\routes.py .\celery_worker\celery_app.py .\celery_worker\tasks.py .\crews\analytics_crew.py .\crews\bizdev_crew.py .\crews\content_crew.py .\crews\marketing_crew.py .\crews\scheduler_crew.py .\crews\sales_improvement_crew.py .\crews\support_crew.py .\tools\custom\analytics_tools.py .\tools\custom\bizdev_tools.py .\tools\custom\content_tools.py .\tools\custom\gmail_tools.py .\tools\custom\marketing_tools.py .\tools\custom\sales_tools.py .\tools\custom\scheduler_tools.py .\tools\custom\support_automation_tools.py .\tools\custom\support_search_tools.py .\tools\custom\whatsapp_tools.py .\tools\integrations\cross_platform_ads_tools.py
python -m pip check
python -c "from main import app, orchestrator; print(app.title); print([w.value for w in orchestrator.registered_workflows])"
```

## Strict Input Validation

Workflow requests are validated before a job is submitted. Each `workflow_type` has its own required input schema in `models.py`; missing fields, empty strings, wrong list types, or unexpected extra fields return a FastAPI validation error before any CrewAI workflow can consume tokens.

Required `inputs` by workflow:

```text
marketing: product_category, product_usp, target_markets, budget
content: subject, product_category, target_markets, target_languages, platforms (optional: product_features, brand_voice, primary_keywords, generate_visual_assets, image_generation_count, image_quality, image_size)
support: customer, person, inquiry (optional: ticket_id, customer_email, phone_number, inquiry_text, order_id, item_sku, return_reason, order_history, detected_language, channel, channel_thread_id, channel_message_id, sender_profile, attachments, conversation_history)
analytics: product_category, target_markets, date_range, currency
bizdev: product_category, partnership_type, target_markets, target_languages, key_decision_maker_roles
scheduler: event_type, target_markets, event_list, preferred_launch_window
sales_improvement: product_category, target_markets, current_avg_conversion, target_conversion, date_range
```

## Usage Tracking

Stage 2A records usage metadata for completed jobs. When CrewAI exposes token usage in its result object, the app stores it in PostgreSQL and includes it in `GET /api/v1/workflow/{job_id}`.

Tracked fields:

```text
usage_metrics
prompt_tokens
completion_tokens
total_tokens
cost_usd
duration_seconds
```

`duration_seconds` is always measured by the orchestrator/worker. Token fields depend on whether the current CrewAI result exposes usage metrics. `cost_usd` is calculated only from the optional per-million-token rates in `.env`; otherwise it remains `0`.

When `WORKFLOW_BACKEND=celery`, each Celery task also returns a structured result to the Redis result backend:

```json
{
  "data": {
    "workflow_output": "..."
  },
  "meta": {
    "prompt_tokens": 1000,
    "completion_tokens": 500,
    "total_tokens": 1500,
    "cost_usd": 0.001,
    "duration_seconds": 12.3
  }
}
```

The HTTP API keeps the same response shape as before: `GET /api/v1/workflow/{job_id}` returns the workflow output in `result` and exposes usage/cost fields at the top level.

You can inspect recent usage directly in PostgreSQL:

```powershell
docker compose exec postgres psql -U crossborder -d crossborder_ai -c "SELECT job_id, workflow_type, status, total_tokens, cost_usd, duration_seconds FROM workflow_jobs ORDER BY created_at DESC LIMIT 20;"
```

## Workflow Result Cache

Completed workflow results are reused from PostgreSQL when a new request has the same `workflow_type`, validated `inputs`, and runtime configuration fingerprint. A cache hit creates a new completed job with `cache_hit=true`, `source_job_id` pointing to the original run, and token/cost fields set to `0` for the cached response.

To bypass cache for one request, include metadata:

```json
{
  "metadata": {
    "bypass_cache": true
  }
}
```

Inspect cache usage directly in PostgreSQL:

```powershell
docker compose exec postgres psql -U crossborder -d crossborder_ai -c "SELECT job_id, workflow_type, status, cache_hit, source_job_id, total_tokens, created_at FROM workflow_jobs ORDER BY created_at DESC LIMIT 20;"
```

## Observability

Stage 2A stores workflow execution events in PostgreSQL for debugging and future admin UI work. The app records submitted, queued, running, retrying, completed, and failed events in `workflow_job_events`.

These events are operational traces, not hidden model reasoning. They are intended to answer questions such as when a job started, which backend handled it, whether it retried, how long it ran, and where it failed.

Poll a job's event timeline:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/workflow/replace-with-real-job-id/events" `
  -Headers @{ Authorization = "Bearer $env:API_BEARER_TOKEN" }
```

Inspect recent events directly in PostgreSQL:

```powershell
docker compose exec postgres psql -U crossborder -d crossborder_ai -c "SELECT event_id, job_id, event_type, message, created_at FROM workflow_job_events ORDER BY event_id DESC LIMIT 30;"
```

## Support RAG Knowledge Base

The Support workflow searches the local knowledge base before using web tools. Policy documents live in:

```text
docs/knowledge_base/
```

The built-in `SupportKnowledgeSearchTool` chunks markdown files, creates deterministic local vector embeddings, and retrieves the closest policy passages for questions about returns, refunds, shipping, compensation, exchanges, and escalation. This local retrieval does not call OpenAI or external vector services.

## Persistent Job State

Stage 2A uses PostgreSQL for local-backend job state. `MasterOrchestrator` writes submitted, running, completed, and failed jobs to the `workflow_jobs` table instead of keeping job history only in a Python dictionary.

For local PowerShell runs, PostgreSQL must be running and reachable through:

```env
DATABASE_URL=postgresql+psycopg://crossborder:crossborder@localhost:5432/crossborder_ai
```

For Docker Compose, the included `postgres` service provides the database and the FastAPI container uses:

```env
DATABASE_URL=postgresql+psycopg://crossborder:crossborder@postgres:5432/crossborder_ai
```

This persists job records across FastAPI restarts. If you use `WORKFLOW_BACKEND=local`, an in-progress job still stops when the FastAPI process stops; use `WORKFLOW_BACKEND=celery` for worker-based execution outside the web process.

## Queue Manager / Message Broker

The queue manager code is the background execution layer for long-running workflows. FastAPI receives the request, Redis acts as the message broker, and Celery workers execute the CrewAI workflow outside the web server process.

The recommended local stack is Docker Compose:

```powershell
docker compose up -d --build
```

For later starts after the image has already been built:

```powershell
docker compose up -d
```

Docker Compose starts FastAPI, Redis, Celery, PostgreSQL, and Flower together.

Use the default local backend only if you intentionally want to run without Celery:

```env
WORKFLOW_BACKEND=local
DATABASE_URL=postgresql+psycopg://crossborder:crossborder@localhost:5432/crossborder_ai
```

Use Celery when you want production-style queueing, retries, worker concurrency, and job results that survive outside a single FastAPI process:

```env
WORKFLOW_BACKEND=celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
```

Celery workers retry only transient provider or network failures, such as rate limits, timeouts, connection errors, and retryable 5xx/429 HTTP responses. Deterministic failures such as validation errors, bad request schemas, missing configuration, authentication failures, and programming errors fail immediately so they do not waste additional API calls.

Flower monitoring is exposed at:

```text
http://localhost:5555
```

The HTTP API endpoints stay the same in both modes:

```text
POST /api/v1/workflow
GET  /api/v1/workflow/{job_id}
```

## Health Check

Health checks do not execute workflows.

```powershell
curl http://localhost:8000/health
```

Expected shape:

```json
{
  "status": "healthy",
  "registered_workflows": ["analytics", "bizdev", "marketing", "content", "scheduler", "sales_improvement", "support"]
}
```

## Run Admin Dashboard

The Stage 2B Streamlit dashboard is a lightweight admin UI for testing workflows without writing raw HTTP requests. It can submit workflow jobs, auto-refresh active job status, display usage metadata, and show execution events. Active jobs also report lightweight CrewAI task-level progress, including planned task count, current task index, task name, and agent role when available.

With the Docker backend running, start the dashboard locally from the already configured virtual environment:

```powershell
streamlit run .\admin_dashboard.py
```

If `API_BEARER_TOKEN` is set, the dashboard reads it from `.env` by default. You can also paste the token into the sidebar. Opening the dashboard does not run a workflow; submitting a workflow from the page may consume tokens.

## Run Business Development

This request starts the CrewAI workflow and may consume OpenAI API tokens.
Without `CRUNCHBASE_API_KEY` or `APOLLO_API_KEY`, the BizDev lead enrichment tool uses development fallback lead data and the output should be treated as illustrative until validated with real B2B provider data.
The API response enforces `data_source`, `confidence_level`, and `assumptions` from configured credentials after the crew finishes, so these fields should not claim live-provider confidence when fallback or placeholder tools were used.

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
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/workflow" `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ Authorization = "Bearer $env:API_BEARER_TOKEN" } `
  -Body $body
```

## Run Sales Performance Improvement

This request starts the Sales Performance Improvement CrewAI workflow and may consume OpenAI API tokens.
Without Shopify or Amazon order API credentials, the sales tools use development fallback sample data and the output should be treated as illustrative until validated with real commerce or CRM analytics.
The API response enforces `data_source`, `confidence_level`, and `assumptions` from configured credentials after the crew finishes, so these fields should not claim live-provider confidence when fallback or placeholder tools were used.

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
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/workflow" `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ Authorization = "Bearer $env:API_BEARER_TOKEN" } `
  -Body $body
```

## Run Event Scheduler

This request starts the Event Scheduler CrewAI workflow and may consume OpenAI API tokens.
Without `HOLIDAY_API_KEY`, the scheduler uses development fallback calendar context and the output should be treated as illustrative until validated with a real holiday/timezone provider.
Scheduler results are validated against `preferred_launch_window`; if the model returns dates outside that window, the job fails instead of returning an invalid completed schedule.
The API response enforces `data_source`, `confidence_level`, and `assumptions` from configured credentials after the crew finishes, so these fields should not claim live-provider confidence when fallback or placeholder tools were used.

```powershell
$body = @{
  workflow_type = "scheduler"
  inputs = @{
    event_type = "Product Launch & Promotional Campaign"
    target_markets = "US, UK, Germany, Japan"
    event_list = "Smart Camera Launch, Early Access Sale, Influencer Drop, Post-Launch Retargeting"
    preferred_launch_window = "2026-05-15 to 2026-06-15"
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/workflow" `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ Authorization = "Bearer $env:API_BEARER_TOKEN" } `
  -Body $body
```

## Run Data Analytics

This request starts the Analytics CrewAI workflow and may consume OpenAI API tokens.
Without real platform credentials such as Shopify or Amazon order API settings, analytics commerce metrics use development fallback sample data and the output should be treated as illustrative. If `SERPER_API_KEY` is configured, competitive research uses live Serper search snippets and source URLs through `CompetitorBenchmarkTool`; with `SERPER_DEEP_READ_ENABLED=true`, it also performs best-effort concurrent reads of the top result pages and passes source excerpts into the market research task. Those search-derived insights still require source-page validation before business decisions.
Analytics output includes `source_evidence`, a claim-level evidence list with market, claim, evidence summary, source URLs, and confidence. It also includes `market_intelligence_by_region`, `evidence_synthesis`, `data_quality_notes`, and `recommended_next_research` so Serper snippets and deep-read source pages are preserved as region-by-region market intelligence instead of being compressed into a short summary.
The API response enforces `data_source`, `confidence_level`, and `assumptions` from configured credentials after the crew finishes, so these fields should not claim live-provider confidence when fallback or placeholder tools were used.

```powershell
$body = @{
  workflow_type = "analytics"
  inputs = @{
    product_category = "Smart Home Security Cameras"
    target_markets = "US, UK, Germany, Japan"
    date_range = "Last 30 Days"
    currency = "USD"
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/workflow" `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ Authorization = "Bearer $env:API_BEARER_TOKEN" } `
  -Body $body
```

## Run Customer Support

This request starts the Customer Support CrewAI workflow and may consume OpenAI API tokens.
The workflow remains backward compatible with the original `customer`, `person`, and `inquiry` fields.

```powershell
$body = @{
  workflow_type = "support"
  inputs = @{
    customer = "GlobalTech Solutions"
    person = "Maria Chen"
    inquiry = "Our bulk order #EU-8842 is delayed. We need it by Friday for a product launch. What are the expedited shipping options and compensation policy?"
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/workflow" `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ Authorization = "Bearer $env:API_BEARER_TOKEN" } `
  -Body $body
```

Customer Service 1.1 also accepts optional ticket, customer, order, and return context. These fields enable
local sentiment/intent triage, VIP handoff detection, RMA policy validation, simulated return-label generation,
and WMS inbound notification context before the CrewAI agents draft and QA the final response.

```powershell
$body = @{
  workflow_type = "support"
  inputs = @{
    customer = "GlobalTech Solutions"
    person = "Tanaka Sora"
    inquiry = "The camera arrived damaged and this is unacceptable. I need a refund immediately."
    ticket_id = "TKT-JP-VIP-001"
    customer_email = "tanaka@enterprise.com"
    phone_number = "+81-3-1234-5678"
    order_id = "JP-2026-8842"
    item_sku = "CAM-4K-PRO"
    return_reason = "Item arrived damaged"
    order_history = @{
      lifetime_value = 8500
      order_count = 12
      days_since_delivery = 10
      item_condition = "damaged"
      region = "JP"
    }
  }
} | ConvertTo-Json -Depth 6

Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/workflow" `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ Authorization = "Bearer $env:API_BEARER_TOKEN" } `
  -Body $body
```

## Omni-Channel Support Inbox

### Deployment checklist

- Configure WhatsApp sending with `WHATSAPP_PROVIDER=ycloud` for short-term testing, or `WHATSAPP_PROVIDER=meta` when Meta Cloud API credentials are available. YCloud uses `YCLOUD_API_KEY`, `YCLOUD_WHATSAPP_FROM`, `YCLOUD_WABA_ID`, optional `YCLOUD_BASE_URL`, and `YCLOUD_WEBHOOK_SECRET` for inbound webhook verification; Meta uses `WHATSAPP_ACCESS_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_VERIFY_TOKEN`, and optionally `WHATSAPP_APP_SECRET`; legacy deployment names `WHATSAPP_TOKEN` and `WHATSAPP_PHONE_ID` are accepted as fallbacks.
- Configure `SUPPORT_HANDOFF_WEBHOOK_URL` for human handoff notifications; `SLACK_WEBHOOK_URL` is accepted as a compatibility fallback.
- Configure Gmail send/sync settings when email replies are needed. `SENDGRID_API_KEY` is not used by the current runnable code; Gmail is the active email provider, and SendGrid can be added later as a separate provider adapter.
- Install the existing async/runtime dependencies from `requirements.txt`; `fastapi`, `uvicorn[standard]`, and `httpx` are already listed. `asyncio` is part of Python's standard library and should not be installed separately.
- Use `WORKFLOW_BACKEND=celery` with Redis and PostgreSQL for production-style queueing and durable job state. The provided `docker-compose.yml` already starts FastAPI, Redis, Celery, PostgreSQL, and Flower.
- Configure `SUPPORT_SESSION_REDIS_URL` when you want Redis-backed session context. If omitted, the app falls back to `CELERY_BROKER_URL`; if Redis is unavailable, PostgreSQL remains the source of truth and support workflows continue with database-backed history.
- Configure `PIM_BACKEND=akeneo|plytix|custom` plus the matching `PIM_*_BASE_URL` and `PIM_*_API_KEY` when pre-sales product answers should use a live PIM. Without credentials, Customer Service uses a deterministic mock PIM fallback for local and CI runs; Postgres/Redis/inbox are not product master-data stores.
- Place product catalogs and support references in `docs/knowledge_base` as `.md`, `.txt`, or `.pdf`. PDF catalogs are parsed with `pdfplumber` and injected into pre-sales context as local catalog knowledge, while support policies remain searchable through the same lightweight RAG tool.
- Train the optional multilingual intent classifier with `scripts/train_intent_classifier.py` only after installing `requirements-ml.txt`. The default runtime still uses the keyword router; trained artifacts are written under `artifacts/intent_classifier_v1` and are ignored by git.
- WhatsApp approval sending respects Meta's 24-hour customer service window. When Redis session state shows the window is still open, `approve-send` uses the normal WhatsApp text API. When the window is expired or no active session window is available, it sends a pre-approved template instead of free-form text.

WhatsApp Cloud API webhook verification uses Meta's challenge flow:

```text
GET /api/v1/channels/whatsapp/webhook?hub.mode=subscribe&hub.verify_token=<WHATSAPP_VERIFY_TOKEN>&hub.challenge=<challenge>
```

Inbound WhatsApp webhooks are accepted at:

```text
POST /api/v1/channels/whatsapp/webhook
```

When `WHATSAPP_APP_SECRET` is configured, the webhook validates `X-Hub-Signature-256`. Inbound messages are stored idempotently by `channel_message_id`, normalized into support inputs, and submitted as `workflow_type=support`. WhatsApp delivery/read status webhooks update outbound message records.

YCloud inbound WhatsApp webhooks are accepted separately at:

```text
POST /api/v1/channels/ycloud/webhook
```

This endpoint requires `YCLOUD_WEBHOOK_SECRET` and validates `YCloud-Signature` before parsing `whatsapp.inbound_message.received` and `whatsapp.message.updated` events.

Gmail omni-channel support uses Gmail API message fetch plus optional Pub/Sub watch:

```text
POST /api/v1/channels/gmail/watch
POST /api/v1/channels/gmail/pubsub
POST /api/v1/channels/gmail/sync
POST /api/v1/channels/gmail/sync-latest
```

`/gmail/watch` calls Gmail `users.watch` with `GMAIL_WATCH_TOPIC_NAME` and `GMAIL_WATCH_LABEL_IDS`. The Pub/Sub topic must grant Gmail publish permission in Google Cloud. Standard Gmail notifications include `emailAddress` and `historyId`; for deterministic local testing, use `/gmail/sync-latest` with `{"max_results": 5}` or `/gmail/sync` with `{"message_id": "gmail-message-id"}` and `GMAIL_SYNC_ENABLED=true`. Gmail attachments are stored as metadata only, not downloaded.

To test without manually looking up Gmail message ids, use the dashboard's "Sync latest Gmail" button in the Omni-channel support inbox, or call:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/channels/gmail/sync-latest" `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ Authorization = "Bearer $env:API_BEARER_TOKEN" } `
  -Body '{"max_results": 5}'
```

Support agents can review drafts in the dashboard's "Omni-channel support inbox" expander or through the protected API:

```text
GET  /api/v1/support/conversations
GET  /api/v1/support/conversations/{conversation_id}
POST /api/v1/support/conversations/{conversation_id}/assign
POST /api/v1/support/conversations/{conversation_id}/approve-send
```

`approve-send` dispatches by conversation channel. It makes no provider call while `WHATSAPP_SEND_ENABLED=false` or `GMAIL_SEND_ENABLED=false`. Escalated conversations are blocked from this endpoint and must be handled manually.

For WhatsApp conversations outside the 24-hour window, `approve-send` resolves the configured WhatsApp provider and sends a pre-approved template. The built-in template map is `en -> support_reengagement_en (en_US)`, `ja -> support_reengagement_ja (ja)`, `es -> support_reengagement_es (es)`, and `default -> support_reengagement (en_US)`. `WHATSAPP_PROVIDER=meta` uses Meta Cloud API directly; `WHATSAPP_PROVIDER=ycloud` uses YCloud's `sendDirectly` and template APIs for validation/gray rollout. Business code calls the provider adapter, so you can switch providers through configuration.

Redis session state is stored under `support:session:{session_id}` with a default 24-hour TTL (`SUPPORT_SESSION_TTL_SECONDS=86400`) and the latest 20 history entries (`SUPPORT_SESSION_HISTORY_LIMIT=20`). It caches channel, customer id, language preference, metadata, window expiry, and rotated inbound/outbound history for fast omni-channel context. PostgreSQL `support_conversations` and `support_messages` remain authoritative; Redis write/read failures degrade gracefully to database history.

### Multilingual intent classifier training

The Customer Service router can be upgraded later with a trained multilingual classifier. The training script is offline by default:

```powershell
python .\scripts\train_intent_classifier.py --dry-run
```

For real training, install the optional ML dependencies first, then run:

```powershell
python -m pip install -r requirements-ml.txt
python .\scripts\train_intent_classifier.py --train
```

The script trains a `bert-base-multilingual-cased` three-class classifier for `pre_sales`, `order_fulfillment`, and `post_sales_support`, exports `label_map.json`, and saves the final model under `artifacts/intent_classifier_v1/final`. The generated model is not used automatically by the production router yet; it is a prepared integration point for a future configured switch.

The offline 8.7.3 expected-outcome validation lives in `tests.test_customer_service_integration`. It verifies PIM fallback and multilingual intent classifier wiring with a fake classifier, so it does not need ML dependencies, credentials, network, or a trained model.

For long-running Gmail usage, do not rely on OAuth Playground's short-lived access token. Use OAuth Playground with "Use your own OAuth credentials", request offline access for Gmail scopes, and save the returned refresh token:

```env
GMAIL_CLIENT_ID=your_google_oauth_client_id
GMAIL_CLIENT_SECRET=your_google_oauth_client_secret
GMAIL_REFRESH_TOKEN=your_google_refresh_token
GMAIL_ACCESS_TOKEN=
```

The app exchanges the refresh token for a new access token before Gmail sync, watch, direct send, and approval send calls. If your Google Cloud OAuth app is `External` and still in `Testing`, Google may expire refresh tokens after 7 days for Gmail scopes; publish the app to production or use a Workspace internal app to avoid weekly re-authorization. After changing these values in `.env` for Docker, recreate the containers:

```powershell
docker compose up -d --force-recreate fastapi celery_worker
```

## Run Content Creation

This request starts the Content Creation CrewAI workflow and may consume OpenAI API tokens.
For multilingual inputs, the workflow runs research/strategy once, then generates each requested language in parallel up to `CONTENT_LANGUAGE_CONCURRENCY`.
Use `product_features` when you want the article to focus on your actual product instead of broad category-level trends.
Use `brand_voice` and `primary_keywords` to steer localization and SEO metadata. `generate_visual_assets` defaults to `false`; when set to `true`, the workflow calls the OpenAI Image API, saves generated files under `artifacts/content_creation`, and scores local assets with a vision-capable model. If no OpenAI API key is configured, the workflow returns visual prompts/specs and marks image generation as skipped instead of failing the content job.

```powershell
$body = @{
  workflow_type = "content"
  inputs = @{
    subject = "Sustainable Activewear for Cold Climates"
    product_category = "Eco-Friendly Winter Sportswear"
    product_features = "Recycled thermal shell, wind-resistant construction, moisture-wicking base layer, designed for cold outdoor training."
    target_markets = "Germany, Japan, Canada"
    target_languages = @("de", "ja", "en")
    platforms = @("Instagram", "LinkedIn", "X")
    brand_voice = "Premium, practical, sustainability-minded, and culturally respectful"
    primary_keywords = @("thermal activewear", "winter training layer", "recycled sportswear")
    generate_visual_assets = $false
    image_generation_count = 1
    image_quality = "auto"
    image_size = "1024x1024"
  }
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/workflow" `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ Authorization = "Bearer $env:API_BEARER_TOKEN" } `
  -Body $body
```

## Run Marketing Campaign

This request starts the Marketing CrewAI workflow and may consume OpenAI API tokens.
The integration tools under `tools/integrations/` connect Marketing to external ad platforms. Without Google Ads, Meta, or TikTok credentials, these tools use development fallback data and the output should be treated as illustrative until validated with live platform APIs. The API response enforces `data_source`, `confidence_level`, and `assumptions` from configured credentials after the crew finishes, so these fields should not claim live-provider confidence when fallback tools were used.

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
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/workflow" `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ Authorization = "Bearer $env:API_BEARER_TOKEN" } `
  -Body $body
```

## Poll A Job

All workflow submit requests return a `job_id`. Poll it with:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/workflow/replace-with-real-job-id" `
  -Headers @{ Authorization = "Bearer $env:API_BEARER_TOKEN" }
```
