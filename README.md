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

The project expects a root-level `.env` file.

Minimum required for running CrewAI workflows:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL_NAME=gpt-4o-mini
CREWAI_MEMORY_ENABLED=false
OPENAI_INPUT_COST_PER_1M_TOKENS=0
OPENAI_OUTPUT_COST_PER_1M_TOKENS=0
```

`CREWAI_MEMORY_ENABLED` defaults to `false`. Turn it on only after your OpenAI account/key can use the embeddings endpoint required by CrewAI memory.
`OPENAI_INPUT_COST_PER_1M_TOKENS` and `OPENAI_OUTPUT_COST_PER_1M_TOKENS` are optional cost-estimation rates. Leave them at `0` to track tokens and duration without estimating dollars.

Optional shared services:

```env
API_BEARER_TOKEN=optional_local_api_token
SERPER_API_KEY=optional_serper_key
WORKFLOW_BACKEND=local
DATABASE_URL=postgresql+psycopg://crossborder:crossborder@localhost:5432/crossborder_ai
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
CELERY_RETRY_BASE_DELAY_SECONDS=30
CELERY_RETRY_MAX_DELAY_SECONDS=300
```

`WORKFLOW_BACKEND=local` keeps the current lightweight in-process background execution. Use `WORKFLOW_BACKEND=celery` when Redis and a Celery worker are running and you want workflow jobs to be handled by the message broker.
PostgreSQL is used for persistent local-backend job state. The app creates the `workflow_jobs` table on startup.
Runtime configuration and secrets are centralized in `runtime_config.py`. FastAPI/Celery load `.env` once, pass a `config_context` into the orchestrator and crews, and provider tools receive credentials through constructors instead of reading global environment variables directly.
If `API_BEARER_TOKEN` is set, workflow submit and polling endpoints require `Authorization: Bearer <token>`. `/health` stays public for local and container health checks. If `API_BEARER_TOKEN` is empty or missing, auth is disabled for local development.

Optional workflow data providers:

```env
# Business Development lead enrichment. Without these, BizDev uses development fallback lead data.
CRUNCHBASE_API_KEY=optional_crunchbase_key
APOLLO_API_KEY=optional_apollo_key

# Analytics platform metrics. Without this, Analytics uses development fallback sample metrics.
ECOM_API_TOKEN=optional_ecommerce_platform_token

# Sales funnel data. Without this, Sales Improvement uses development fallback sample funnel data.
CRM_API_TOKEN=optional_crm_or_platform_token

# Scheduler holiday/timezone provider. Without this, Scheduler uses development fallback calendar data.
HOLIDAY_API_KEY=optional_holiday_provider_key

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

Current runnable code does not use Shopify, Amazon, or calendar-provider tokens mentioned in archived notes yet. Those belong to future integration work unless a corresponding tool is implemented under `tools/`.

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
    "meta_access_token": "request_scoped_meta_token",
    "meta_ad_account_id": "request_scoped_meta_account_id",
    "tiktok_access_token": "request_scoped_tiktok_token",
    "tiktok_advertiser_id": "request_scoped_tiktok_advertiser_id"
  }
}
```

For a production SaaS implementation, prefer passing a `tenant_id` and loading encrypted provider credentials from a secrets vault. Passing credentials directly in the API request is useful for local development and integration testing, but the Celery broker still receives the task payload.

## No-Token Checks

These checks do not run CrewAI jobs and should not consume OpenAI API tokens.

```powershell
python -m py_compile .\main.py .\models.py .\runtime_config.py .\database.py .\db_models.py .\job_store.py .\orchestrator.py .\api\routes.py .\celery_worker\celery_app.py .\celery_worker\tasks.py .\crews\analytics_crew.py .\crews\bizdev_crew.py .\crews\content_crew.py .\crews\marketing_crew.py .\crews\scheduler_crew.py .\crews\sales_improvement_crew.py .\crews\support_crew.py .\tools\custom\analytics_tools.py .\tools\custom\bizdev_tools.py .\tools\custom\marketing_tools.py .\tools\custom\sales_tools.py .\tools\custom\scheduler_tools.py .\tools\integrations\cross_platform_ads_tools.py
python -m pip check
python -c "from main import app, orchestrator; print(app.title); print([w.value for w in orchestrator.registered_workflows])"
```

## Strict Input Validation

Workflow requests are validated before a job is submitted. Each `workflow_type` has its own required input schema in `models.py`; missing fields, empty strings, wrong list types, or unexpected extra fields return a FastAPI validation error before any CrewAI workflow can consume tokens.

Required `inputs` by workflow:

```text
marketing: product_category, product_usp, target_markets, budget
content: subject, product_category, target_markets, target_languages, platforms
support: customer, person, inquiry
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

Use the default local backend for simple development:

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

Start Redis separately, then run these in two terminals:

```powershell
python .\main.py
```

```powershell
celery -A celery_worker.celery_app worker --loglevel=info --pool=solo
```

On Windows, `--pool=solo` is the safest local Celery worker mode. For Docker/Linux workers, the compose file uses normal worker concurrency.

Celery workers retry only transient provider or network failures, such as rate limits, timeouts, connection errors, and retryable 5xx/429 HTTP responses. Deterministic failures such as validation errors, bad request schemas, missing configuration, authentication failures, and programming errors fail immediately so they do not waste additional API calls.

Docker stack:

```powershell
docker compose up -d --build
```

Flower monitoring is exposed at:

```text
http://localhost:5555
```

The HTTP API endpoints stay the same in both modes:

```text
POST /api/v1/workflow
GET  /api/v1/workflow/{job_id}
```

## Run API Server

Starting the API server does not execute a workflow by itself.

```powershell
python .\main.py
```

Health check:

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
Without `CRM_API_TOKEN`, the sales tools use development fallback sample data and the output should be treated as illustrative until validated with real CRM/platform analytics.
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
Without real platform and competitive data provider credentials such as `ECOM_API_TOKEN`, analytics tools use development fallback sample data and the output should be treated as illustrative.
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

## Run Content Creation

This request starts the Content Creation CrewAI workflow and may consume OpenAI API tokens.

```powershell
$body = @{
  workflow_type = "content"
  inputs = @{
    subject = "Sustainable Activewear for Cold Climates"
    product_category = "Eco-Friendly Winter Sportswear"
    target_markets = "Germany, Japan, Canada"
    target_languages = @("de", "ja", "en")
    platforms = @("Instagram", "LinkedIn", "X")
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
