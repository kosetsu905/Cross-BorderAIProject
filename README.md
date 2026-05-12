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
|   |-- content/
|   |-- support/
|   |-- analytics/
|   |-- scheduler/
|   `-- sales_improvement/
|-- tools/
|   |-- base/
|   |-- integrations/
|   `-- custom/
|       `-- bizdev_tools.py
|-- crews/
|   `-- bizdev_crew.py
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

## About The Marketing Crew Example

The `crews/marketing_crew.py` shown inside the original FastAPI wrapper note was an example of how another workflow would be registered with the master orchestrator. It is not Customer Service code and does not directly depend on the Customer Service workflow.

Right now only the Business Development workflow has been converted into runnable Python code:

```text
workflow_type = "bizdev"
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
```

## No-Token Checks

These checks do not run CrewAI jobs and should not consume OpenAI API tokens.

```powershell
python -m py_compile .\main.py .\models.py .\orchestrator.py .\api\routes.py .\crews\bizdev_crew.py .\tools\custom\bizdev_tools.py
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
  "registered_workflows": ["bizdev"]
}
```

## Run Business Development

This request starts the CrewAI workflow and may consume OpenAI API tokens.

```powershell
curl -X POST http://localhost:8000/api/v1/workflow `
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

Poll the returned job:

```powershell
curl http://localhost:8000/api/v1/workflow/replace-with-real-job-id
```
