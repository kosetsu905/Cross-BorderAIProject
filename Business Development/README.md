# Business Development Workflow

CrewAI workflow for cross-border business development, partner research, localized outreach, and CRM handoff.

## Files

- `config/bizdev_agents.yaml`: agent role, goal, and backstory configuration
- `config/bizdev_tasks.yaml`: task definitions and expected outputs
- `tools/bizdev_tools.py`: custom BD tools with development fallbacks
- `crews/bizdev_crew.py`: CrewAI assembly and `run_bizdev_crew` entry point

## Expected Inputs

```json
{
  "product_category": "Smart Home Security Cameras",
  "partnership_type": "Regional Distributors & Retail Partners",
  "target_markets": "Germany, Japan, Canada",
  "target_languages": ["de", "ja", "en"],
  "key_decision_maker_roles": "Head of Procurement, Channel Manager"
}
```

## Environment Variables

- `OPENAI_API_KEY`: required by CrewAI/OpenAI runtime
- `OPENAI_MODEL_NAME`: optional, defaults to `gpt-4o-mini`
- `SERPER_API_KEY`: optional, enables Serper web search tools
- `CRUNCHBASE_API_KEY` or `APOLLO_API_KEY`: optional, reserved for production B2B data provider integration
