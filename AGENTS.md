# Project Intelligence Specification

This file defines the role system, coding standards, and architecture preferences for AI agents working on this codebase. All generated code must strictly follow these guidelines.

---

## 🎭 Role System

### Role Assignments

**Development Agent (资深架构师 / Senior Architect)**
- Role: Senior Software Architect with 10+ years experience in distributed systems
- Personality: Methodical, pragmatic, security-conscious
- Responsibilities:
  - Design scalable, maintainable solutions following SOLID principles
  - Enforce type safety with Pydantic models and type hints
  - Implement proper error handling and retry logic (use tenacity)
  - Follow existing patterns in the codebase (crew-based architecture)
  - Write clean, self-documenting code with meaningful variable names
- Communication style: Technical but clear, explain architectural decisions

**Code Review Agent (代码审查员 / Code Reviewer)**
- Role: Senior Code Reviewer specializing in logic and security
- Personality: Detail-oriented, security-focused, thorough
- Responsibilities:
  - Identify logic flaws, race conditions, and edge cases
  - Validate input validation and boundary conditions
  - Check for security vulnerabilities (injection, auth bypass, data leaks)
  - Ensure proper exception handling and logging
  - Verify type consistency and Pydantic model usage
- Review checklist:
  - [ ] All public functions have type hints
  - [ ] Pydantic models use ConfigDict(extra="forbid") for strict validation
  - [ ] Error handling covers all failure paths
  - [ ] No hardcoded secrets or credentials (use .env)
  - [ ] Database operations use proper session management
  - [ ] Async operations use proper await patterns

**Testing Agent (测试工程师 / Test Engineer)**
- Role: QA Engineer focused on integration and edge cases
- Personality: Systematic, boundary-test focused
- Responsibilities:
  - Write pytest-based integration tests
  - Test all error paths and edge cases
  - Validate CrewAI workflow outputs against expected schemas
  - Test multi-channel support scenarios (Gmail, WhatsApp)
  - Mock external services (httpx, CrewAI calls)
- Test patterns:
  - Use fixtures for database sessions and test clients
  - Test idempotency for job submission
  - Verify cache behavior (utils.result_cache)
  - Test retry logic (utils.retry_policy)

---

## 💻 Technology Stack

### Core Framework
- **Language**: Python 3.12+
- **Web Framework**: FastAPI with async support
- **AI Orchestration**: CrewAI for multi-agent workflows
- **Task Queue**: Celery with Redis broker
- **Database**: PostgreSQL with SQLAlchemy ORM
- **HTTP Client**: httpx (async preferred over requests)
- **Validation**: Pydantic v2 with strict mode

### Key Libraries
- pydantic: Data validation with BaseModel, Field, ConfigDict
- 	enacity: Retry logic with exponential backoff
- langdetect: Language detection for multilingual support
- pdfplumber: PDF extraction for document processing
- crewai: Agent and task orchestration
- crewai-tools: Built-in tools for web scraping and search

### Development Tools
- Type checking: mypy (strict mode)
- Linting: ruff
- Testing: pytest with pytest-asyncio
- Formatting: ruff format

---

## 🏗️ Architecture Preferences

### 1. Crew-Based Workflow Pattern

All business workflows (marketing, content, support, analytics, bizdev, scheduler, sales_improvement) follow the crew pattern:

`python
# crews/{workflow}_crew.py

def run_{workflow}_crew(inputs: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    # 1. Load YAML config
    config_dir = BASE_DIR / "config" / "{workflow}"
    agents_config = yaml.safe_load((config_dir / "agents.yaml").read_text())
    tasks_config = yaml.safe_load((config_dir / "tasks.yaml").read_text())
    
    # 2. Define agents with specific roles
    agents = [
        Agent(
            role=agents_config[agent_name]["role"],
            goal=agents_config[agent_name]["goal"],
            backstory=agents_config[agent_name]["backstory"],
            verbose=True,
            tools=[...],
        )
        for agent_name in agents_config
    ]
    
    # 3. Define tasks with Pydantic output schemas
    tasks = [
        Task(
            description=tasks_config[task_name]["description"],
            expected_output=tasks_config[task_name]["expected_output"],
            agent=agent_for_task,
            output_pydantic=OutputSchema,
        )
        for task_name in tasks_config
    ]
    
    # 4. Execute crew and serialize result
    crew = Crew(agents=agents, tasks=tasks, verbose=True)
    result = crew.kickoff(inputs=inputs)
    return serialize_crew_result(result)
`

**Rules:**
- Always use YAML config files (config/{workflow}/agents.yaml, 	asks.yaml)
- Define Pydantic output schemas for all crew outputs
- Serialize results using utils.crew_result.serialize_crew_result
- Track usage with utils.usage_tracking
- Report progress via utils.workflow_progress

### 2. API Layer Pattern

`python
# api/routes.py

@router.post("/workflows/{workflow_type}")
async def submit_workflow(
    workflow_type: WorkflowType,
    request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    db: DbDependency,
    _: AuthDependency,
):
    job_id = await orchestrator.submit_job(
        workflow_type,
        request.inputs,
        metadata=request.metadata or {},
    )
    return JobEventResponse(job_id=job_id, status=JobStatus.PENDING)
`

**Rules:**
- Use FastAPI dependency injection for auth and database
- All endpoints require authentication (AuthDependency)
- Use async/await for all I/O operations
- Return Pydantic response models
- Use BackgroundTasks for long-running operations

### 3. Database Pattern

`python
# database.py

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
`

**Rules:**
- Use SQLAlchemy ORM for all database operations
- Always use session context managers or dependency injection
- Define models in db_models.py with proper indexes
- Use migrations (alembic) for schema changes
- Never commit transactions in business logic (let session manager handle it)

### 4. Tool Pattern

`python
# tools/custom/{tool_name}_tools.py

from crewai import BaseTool
from pydantic import BaseModel, Field

class {ToolName}Input(BaseModel):
    param1: str = Field(..., description="Description of param1")
    param2: int = Field(..., description="Description of param2")

class {ToolName}Tool(BaseTool):
    name: str = "{ToolName} Tool"
    description: str = "Description of what this tool does"
    args_schema: type[BaseModel] = {ToolName}Input
    
    def _run(self, param1: str, param2: int) -> str:
        # Implementation
        return result
`

**Rules:**
- Define input schema with Pydantic
- Use descriptive tool names and descriptions
- Handle errors gracefully and return meaningful messages
- Use httpx for external API calls (async preferred)
- Add retry logic with tenacity for flaky APIs

### 5. Error Handling Pattern

`python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def external_api_call(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=30.0)
        response.raise_for_status()
        return response.json()
`

**Rules:**
- Use tenacity for retry logic (max 3 attempts, exponential backoff)
- Log all errors with appropriate context
- Use HTTPException for API errors with proper status codes
- Validate all inputs with Pydantic before processing
- Handle specific exceptions (ConnectionError, Timeout, ValidationError)

### 6. Configuration Pattern

`python
# runtime_config.py

from pydantic import BaseModel, ConfigDict

class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    
    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        return cls(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
        )
`

**Rules:**
- Use Pydantic for all configuration with strict validation
- Load secrets from environment variables (python-dotenv)
- Never hardcode API keys or credentials
- Provide sensible defaults for non-critical settings
- Validate configuration at startup

---

## 📝 Code Standards

### Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Variables (simple) | snake_case | user_input |
| Variables (context-dependent) | short but meaningful | i, j for loops; e for exceptions; x for transformations |
| Functions | snake_case | un_marketing_crew |
| Classes | PascalCase | MarketingInputs |
| Constants | UPPER_SNAKE_CASE | MAX_RETRIES |
| Files | snake_case | nalytics_crew.py |
| Directories | snake_case | 	ools/custom/ |

### Type Hints

**Always use type hints:**

`python
from typing import Any
from collections.abc import Callable

def process_workflow(
    workflow_type: WorkflowType,
    inputs: dict[str, Any],
    callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    pass
`

### Docstrings

Use Google-style docstrings for complex functions:

`python
def submit_job(
    workflow_type: WorkflowType,
    inputs: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Submit a workflow job to the orchestrator.
    
    Args:
        workflow_type: Type of workflow to execute.
        inputs: Workflow-specific input parameters.
        metadata: Optional metadata to attach to the job.
    
    Returns:
        Unique job identifier (UUID string).
    
    Raises:
        ValueError: If inputs validation fails.
        ConnectionError: If job store is unreachable.
    """
`

### Imports

**Order:**
1. Standard library (alphabetical)
2. Third-party (alphabetical)
3. Local/relative (alphabetical)

`python
import logging
import uuid
from pathlib import Path
from typing import Any

import httpx
from crewai import Agent, Crew
from pydantic import BaseModel

from crews.marketing_crew import run_marketing_crew
from models import WorkflowType
from utils.retry_policy import retry_with_backoff
`

### Pydantic Models

**Always use strict mode:**

`python
from pydantic import BaseModel, ConfigDict, Field

class MyModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    field1: str = Field(..., min_length=1, description="Required field")
    field2: int | None = Field(None, ge=0, description="Optional field")
`

### Logging

`python
import logging

logger = logging.getLogger(__name__)

logger.info("Job submitted", extra={"job_id": job_id, "workflow": workflow_type})
logger.error("Workflow failed", extra={"job_id": job_id, "error": str(e)})
`

**Rules:**
- Use logging.getLogger(__name__) at module level
- Use structured logging with extras for context
- Log at INFO level for normal operations, ERROR for failures
- Never log secrets or sensitive data

---

## 🔐 Security Guidelines

### Authentication

- All API endpoints require bearer token authentication
- Use pi.auth.verify_bearer_token dependency
- Never expose auth tokens in logs or error messages

### Data Validation

- Always validate inputs with Pydantic (strict mode)
- Sanitize user inputs before database insertion
- Use parameterized queries (SQLAlchemy handles this)

### Secrets Management

- Store secrets in .env file (never commit)
- Access via os.getenv() or Pydantic settings
- Never log secrets or include in error messages

### External APIs

- Validate all external API responses
- Use timeouts for all HTTP requests (30s default)
- Implement retry logic with exponential backoff
- Handle rate limiting gracefully

---

## 🧪 Testing Guidelines

### Test Structure

`python
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def test_client(db_session: Session) -> TestClient:
    app.dependency_overrides[get_db_session] = lambda: db_session
    return TestClient(app)

def test_submit_workflow(test_client: TestClient, auth_headers: dict):
    response = test_client.post(
        "/workflows/marketing",
        json={
            "inputs": {
                "product_category": "Electronics",
                "product_usp": "Fast shipping",
                "target_markets": "US, CA",
                "budget": "1000",
            }
        },
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"
`

### Test Coverage

- **Integration tests**: Full workflow execution (crew + orchestrator)
- **Unit tests**: Individual tools and utility functions
- **Edge cases**: Empty inputs, invalid data, timeout scenarios
- **Multi-channel**: Test Gmail, WhatsApp, and other integrations

### Mocking

`python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_external_api_with_retry():
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = [
            httpx.TimeoutException("Timeout"),
            httpx.Response(200, json={"result": "success"}),
        ]
        result = await external_api_call("https://api.example.com")
        assert result == {"result": "success"}
        assert mock_get.call_count == 2
`

---

## 📦 Deployment Guidelines

### Docker

`dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
`

### Environment Variables

Required in .env:

`ash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/crossborder

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=4096

# Redis (Celery broker)
REDIS_URL=redis://localhost:6379/0

# Auth
AUTH_BEARER_TOKEN=your-secret-token

# External APIs (as needed)
SERPER_API_KEY=...
GMAIL_CREDENTIALS_FILE=...
WHATSAPP_ACCESS_TOKEN=...
`

### Health Checks

- /health: Basic service health (database, redis connectivity)
- /ready: Service ready to accept requests (all dependencies initialized)

---

## 🚫 Anti-Patterns (Do NOT Do This)

### 1. No Global State

`python
# ❌ BAD
global_cache = {}

def get_data(key: str):
    return global_cache.get(key)

# ✅ GOOD
class CacheManager:
    def __init__(self):
        self._cache = {}
    
    def get(self, key: str):
        return self._cache.get(key)
`

### 2. No Bare Exceptions

`python
# ❌ BAD
try:
    result = api_call()
except:
    pass

# ✅ GOOD
try:
    result = api_call()
except httpx.TimeoutException as e:
    logger.error("API timeout", extra={"error": str(e)})
    raise
`

### 3. No Hardcoded Values

`python
# ❌ BAD
if len(text) > 1000:
    text = text[:1000]

# ✅ GOOD
MAX_TEXT_LENGTH = 1000
if len(text) > MAX_TEXT_LENGTH:
    text = text[:MAX_TEXT_LENGTH]
`

### 4. No Blocking Calls in Async

`python
# ❌ BAD
async def process():
    response = requests.get(url)  # Blocking!

# ✅ GOOD
async def process():
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
`

### 5. No Unvalidated Inputs

`python
# ❌ BAD
def create_user(name: str, email: str):
    session.add(User(name=name, email=email))

# ✅ GOOD
class CreateUserInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., pattern=r"^[\w.+-]+@[\w-]+\.[\w.]+$")

def create_user(input: CreateUserInput):
    session.add(User(name=input.name, email=input.email))
`

---

## 📚 Documentation Standards

### Code Comments

- Use comments sparingly; write self-documenting code
- Comment "why" not "what"
- Use TODO with issue numbers: # TODO(#123): Optimize this query

### README Updates

When adding new features:
1. Update README.md with usage examples
2. Add to API documentation in docs/
3. Update config examples in config/{workflow}/

### Git Commit Messages

`
<type>(<scope>): <subject>

<body>

<footer>
`

Types: eat, ix, docs, style, efactor, 	est, chore

Example:
`
feat(marketing): add TikTok Ads integration

- Implement TikTokAdsTool with campaign creation
- Add retry logic for rate limiting
- Update marketing crew to include TikTok agent

Closes #456
`

---

## 🎯 Workflow-Specific Guidelines

### Marketing Crew

- Focus on multi-platform ad campaigns (Meta, Google, TikTok)
- Use compliance checking for different regions
- Generate ad variants with proper CTAs
- Output: FinalCampaignOutput schema

### Content Crew

- Multilingual content generation
- SEO optimization with keyword research
- Platform-specific formatting
- Output: ContentOutput schema

### Support Crew

- Multi-channel support (Gmail, WhatsApp)
- Conversation context tracking
- Automated responses with human handoff
- Integration with SupportInboxStore
- Output: SupportResponse schema

### Analytics Crew

- Data aggregation from multiple sources
- KPI tracking and reporting
- Trend analysis and insights
- Output: AnalyticsReport schema

### BizDev Crew

- Partner identification and outreach
- Market opportunity analysis
- Pipeline tracking
- Output: BizDevReport schema

### Scheduler Crew

- Intelligent task scheduling
- Priority-based execution
- Resource optimization
- Output: ScheduleOutput schema

### Sales Improvement Crew

- Sales funnel analysis
- Conversion optimization
- Lead scoring
- Output: SalesReport schema

---

## 🤖 CrewAI Runtime Intelligence Injection

All CrewAI agents automatically receive project intelligence at runtime via utils/project_intelligence.py.

### How It Works

When any crew file initializes agents:

`python
from utils.project_intelligence import augment_agents_config

agents_config = _load_yaml_config('agents.yaml')
agents_config = augment_agents_config(agents_config, workflow='marketing')
`

This function:
1. Loads AGENTS.md content
2. Extracts relevant sections (Security Guidelines, Code Standards, Anti-Patterns)
3. Injects workflow-specific guidelines if workflow parameter is provided
4. Appends the intelligence to each agent's backstory

### What Gets Injected

By default, every agent receives:
- **Security Guidelines**: Authentication, validation, secrets management
- **Code Standards**: Naming conventions, type hints, logging patterns
- **Anti-Patterns**: Common mistakes to avoid

When workflow parameter is specified, agents also receive:
- Workflow-specific crew guidelines

### Customization

`python
# Custom sections
agents_config = augment_agents_config(
    agents_config,
    sections=['Testing Guidelines', 'Deployment Guidelines'],
)

# Workflow-specific
agents_config = augment_agents_config(
    agents_config,
    workflow='content',
)
`

### Benefits

- ✅ Developers: Codex and other coding agents follow AGENTS.md during development
- ✅ Runtime: All 7 CrewAI workflows automatically enforce project standards
- ✅ No manual duplication: Guidelines are centrally managed in AGENTS.md

---
## 🔄 Version Control

### Branch Naming

- Feature: eature/{description}
- Fix: ix/{description}
- Docs: docs/{description}

### Pull Request Process

1. Create branch from main
2. Implement changes with tests
3. Run linters: uff check .
4. Run tests: pytest tests/
5. Update documentation
6. Submit PR with clear description
7. Request review from code review agent

---

## 📞 Support

For questions or clarifications:
- Check existing code patterns in crews/, 	ools/
- Review YAML configs in config/{workflow}/
- Consult utility modules in utils/
- Run tests to understand expected behavior

---

**Last Updated**: 2026-05-30  
**Version**: 1.0.0  
**Maintainers**: Cross-Border AI Project Team