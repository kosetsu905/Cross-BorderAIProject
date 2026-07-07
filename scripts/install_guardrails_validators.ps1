$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($env:GUARDRAILS_TOKEN)) {
    $EnvPath = Join-Path $ProjectRoot ".env"
    if (Test-Path $EnvPath) {
        $GuardrailsTokenLine = Select-String -Path $EnvPath -Pattern '^GUARDRAILS_TOKEN=' | Select-Object -First 1
        if ($GuardrailsTokenLine) {
            $env:GUARDRAILS_TOKEN = ($GuardrailsTokenLine.Line -replace '^GUARDRAILS_TOKEN=', '').Trim().Trim('"').Trim("'")
        }
    }
}

if ([string]::IsNullOrWhiteSpace($env:GUARDRAILS_TOKEN) -or $env:GUARDRAILS_TOKEN -eq "replace_with_guardrails_hub_token_for_docker_build") {
    throw "GUARDRAILS_TOKEN is required in the shell or .env before building the Docker Guardrails runtime."
}

Write-Output "Building FastAPI and Celery images with Guardrails Hub validators installed in-image..."
docker compose build fastapi celery_worker

Write-Output "Validating Guardrails Hub registry in the FastAPI image..."
docker compose run --rm --no-deps fastapi guardrails hub list

Write-Output "Smoke testing project Guardrails runtime in the FastAPI image..."
docker compose run --rm --no-deps fastapi python -c "from services.workflow_guardrails import WorkflowGuardrailService; WorkflowGuardrailService().validate_runtime(smoke_toxic=True); print('Guardrails Hub runtime validation passed.')"

Write-Output "Validating Guardrails Hub registry in the Celery image..."
docker compose run --rm --no-deps celery_worker guardrails hub list

Write-Output "Docker Guardrails runtime is ready. Start/recreate services with:"
Write-Output "docker compose up -d --force-recreate fastapi celery_worker"
