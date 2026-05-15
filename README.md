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
```

`CREWAI_MEMORY_ENABLED` defaults to `false`. Turn it on only after your OpenAI account/key can use the embeddings endpoint required by CrewAI memory.

Optional shared services:

```env
SERPER_API_KEY=optional_serper_key
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
```

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
```

Current runnable code does not use the Google Ads, Meta, TikTok, Shopify, Amazon, or calendar-provider tokens mentioned in archived notes yet. Those belong to future integration work unless a corresponding tool is implemented under `tools/`.

## No-Token Checks

These checks do not run CrewAI jobs and should not consume OpenAI API tokens.

```powershell
python -m py_compile .\main.py .\models.py .\orchestrator.py .\api\routes.py .\celery_worker\celery_app.py .\celery_worker\tasks.py .\crews\analytics_crew.py .\crews\bizdev_crew.py .\crews\content_crew.py .\crews\marketing_crew.py .\crews\scheduler_crew.py .\crews\sales_improvement_crew.py .\crews\support_crew.py .\tools\custom\analytics_tools.py .\tools\custom\bizdev_tools.py .\tools\custom\marketing_tools.py .\tools\custom\sales_tools.py .\tools\custom\scheduler_tools.py
python -m pip check
python -c "from main import app, orchestrator; print(app.title); print([w.value for w in orchestrator.registered_workflows])"
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
  -Body $body
```

## Run Sales Performance Improvement

This request starts the Sales Performance Improvement CrewAI workflow and may consume OpenAI API tokens.
Without `CRM_API_TOKEN`, the sales tools use development fallback sample data and the output should be treated as illustrative until validated with real CRM/platform analytics.

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
  -Body $body
```

## Run Event Scheduler

This request starts the Event Scheduler CrewAI workflow and may consume OpenAI API tokens.
Without `HOLIDAY_API_KEY`, the scheduler uses development fallback calendar context and the output should be treated as illustrative until validated with a real holiday/timezone provider.
Scheduler results are validated against `preferred_launch_window`; if the model returns dates outside that window, the job fails instead of returning an invalid completed schedule.

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
  -Body $body
```

## Run Data Analytics

This request starts the Analytics CrewAI workflow and may consume OpenAI API tokens.
Without real platform and competitive data provider credentials such as `ECOM_API_TOKEN`, analytics tools use development fallback sample data and the output should be treated as illustrative.

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
  -Body $body
```

## Run Marketing Campaign

This request starts the Marketing CrewAI workflow and may consume OpenAI API tokens.

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
  -Body $body
```

## Poll A Job

All workflow submit requests return a `job_id`. Poll it with:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/workflow/replace-with-real-job-id"
```
