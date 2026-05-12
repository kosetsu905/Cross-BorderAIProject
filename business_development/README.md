# Business Development Workflow

CrewAI workflow for cross-border business development, partner research, localized outreach, and CRM handoff.

## Runtime Location

The runnable code now lives in the project package:

```text
business_development/
  config/bizdev_agents.yaml
  config/bizdev_tasks.yaml
  tools/bizdev_tools.py
  crews/bizdev_crew.py
fastapi_wrapper_master_orchestrator/
  main.py
  models.py
  orchestrator.py
main.py
```

`code.txt` is kept as the original design note. The Python files above are the runnable implementation.

## Environment

From the project root:

```powershell
cd D:\Cross-BorderAIProject
.\.venv\Scripts\Activate.ps1
```

Install dependencies if needed:

```powershell
python -m pip install -r requirements.txt
```

The project expects a root-level `.env` file. At minimum:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL_NAME=gpt-4o-mini
```

Optional:

```env
SERPER_API_KEY=your_serper_key
CRUNCHBASE_API_KEY=your_crunchbase_key
APOLLO_API_KEY=your_apollo_key
```

`SERPER_API_KEY` enables web search. If it is not set, the workflow skips `SerperDevTool`.

## No-Token Checks

These commands do not call OpenAI and should not consume API tokens.

Compile-check the Python files:

```powershell
python -m py_compile .\main.py .\fastapi_wrapper_master_orchestrator\main.py .\fastapi_wrapper_master_orchestrator\models.py .\fastapi_wrapper_master_orchestrator\orchestrator.py .\business_development\crews\bizdev_crew.py .\business_development\tools\bizdev_tools.py
```

Check installed dependencies:

```powershell
python -m pip check
```

Validate that the module and YAML configs can load:

```powershell
python -c "from business_development.crews.bizdev_crew import _load_yaml_config; print(list(_load_yaml_config('bizdev_agents.yaml').keys())); print(list(_load_yaml_config('bizdev_tasks.yaml').keys()))"
```

Import the FastAPI app without starting a job:

```powershell
python -c "from fastapi_wrapper_master_orchestrator.main import app; print(app.title)"
```

## Start The API Server

Starting the server does not run the workflow by itself and should not consume OpenAI API tokens.

```powershell
python .\main.py
```

Alternative:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open the health endpoint:

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

## Run Business Development Through FastAPI

Submitting this request starts CrewAI execution and may consume OpenAI API tokens.

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

The response contains a `job_id`:

```json
{
  "job_id": "replace-with-real-job-id",
  "status": "pending",
  "result": null,
  "error": null
}
```

Poll job status:

```powershell
curl http://localhost:8000/api/v1/workflow/replace-with-real-job-id
```

## Expected Input

```json
{
  "product_category": "Smart Home Security Cameras",
  "partnership_type": "Regional Distributors & Retail Partners",
  "target_markets": "Germany, Japan, Canada",
  "target_languages": ["de", "ja", "en"],
  "key_decision_maker_roles": "Head of Procurement, Channel Manager"
}
```

## Expected Output Shape

The final structured output follows `BizDevOutput`:

```json
{
  "target_leads": [],
  "value_proposition": "",
  "outreach_sequences": [],
  "follow_up_cadence": [],
  "crm_payload": {}
}
```

## Notes

- `POST /api/v1/workflow` with `"workflow_type": "bizdev"` starts CrewAI execution and can consume OpenAI API tokens.
- Starting FastAPI, importing the app, loading YAML, running `py_compile`, checking `/health`, or calling the custom local tools directly should not consume OpenAI API tokens.
- `CRUNCHBASE_API_KEY` and `APOLLO_API_KEY` are placeholders for future production data provider integration. Without them, `B2BLeadLookupTool` uses a local development fallback.
