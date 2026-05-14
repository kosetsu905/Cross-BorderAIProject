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
|   |-- analytics/
|   |-- scheduler/
|   `-- sales_improvement/
|-- tools/
|   |-- base/
|   |-- integrations/
|   `-- custom/
|       |-- bizdev_tools.py
|       `-- marketing_tools.py
|-- crews/
|   |-- bizdev_crew.py
|   |-- content_crew.py
|   `-- marketing_crew.py
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
bizdev
marketing
content
```

## Setup

```powershell
cd D:\Cross-BorderAIProject
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The project expects a root-level `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL_NAME=gpt-4o-mini
SERPER_API_KEY=optional_serper_key
CREWAI_MEMORY_ENABLED=false
```

`CREWAI_MEMORY_ENABLED` defaults to `false`. Turn it on only after your OpenAI account/key can use the embeddings endpoint required by CrewAI memory.

## No-Token Checks

These checks do not run CrewAI jobs and should not consume OpenAI API tokens.

```powershell
python -m py_compile .\main.py .\models.py .\orchestrator.py .\api\routes.py .\crews\bizdev_crew.py .\crews\content_crew.py .\crews\marketing_crew.py .\tools\custom\bizdev_tools.py .\tools\custom\marketing_tools.py
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
  "registered_workflows": ["bizdev", "marketing", "content"]
}
```

## Run Business Development

This request starts the CrewAI workflow and may consume OpenAI API tokens.

PowerShell recommended:

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

If you want curl syntax in PowerShell, use `curl.exe`, not `curl`:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/workflow `
  -H "Content-Type: application/json" `
  -d '{
    "workflow_type": "bizdev",
    "inputs": {
      "product_category": "Smart Home Security Cameras",
      "partnership_type": "Regional Distributors & Retail Partners",
      "target_markets": "Germany, Japan, Canada",
      "target_languages": ["de", "ja", "en"],
      "key_decision_maker_roles": "Head of Procurement, Channel Manager"
    }
  }'
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
