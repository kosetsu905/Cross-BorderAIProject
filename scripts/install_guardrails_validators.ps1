$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$GuardrailsExe = Join-Path $ProjectRoot ".venv\Scripts\guardrails.exe"
$ConfigPath = Join-Path $ProjectRoot "config\guardrails.yaml"
$ModelCache = Join-Path $ProjectRoot ".cache\huggingface"
$SentenceTransformerCache = Join-Path $ProjectRoot ".cache\sentence-transformers"
$env:HF_HOME = $ModelCache
$env:HF_HUB_CACHE = Join-Path $ModelCache "hub"
$env:SENTENCE_TRANSFORMERS_HOME = $SentenceTransformerCache

New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null
New-Item -ItemType Directory -Force -Path $env:HF_HUB_CACHE | Out-Null
New-Item -ItemType Directory -Force -Path $env:SENTENCE_TRANSFORMERS_HOME | Out-Null

if (-not (Test-Path $GuardrailsExe)) {
    throw "guardrails.exe was not found in project .venv. Run: .\.venv\Scripts\python.exe -m pip install guardrails-ai"
}

& $PythonExe -c "import torch; print(torch.__version__)" *> $null
if ($LASTEXITCODE -ne 0) {
    & $PythonExe -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cpu torch
}

& $PythonExe -c "import sentence_transformers; print(sentence_transformers.__version__)" *> $null
if ($LASTEXITCODE -ne 0) {
    & $PythonExe -m pip install sentence-transformers
}

$ValidatorUris = & $PythonExe -c "import pathlib, yaml; data=yaml.safe_load(pathlib.Path(r'$ConfigPath').read_text(encoding='utf-8')) or {}; uris=[]; guards=data.get('guards') or {}; [uris.append(v.get('hub')) for workflow in guards.values() if isinstance(workflow, dict) for stage in ('input','output') for v in ((workflow.get(stage) or {}).get('validators') or []) if isinstance(v, dict) and v.get('hub')]; print('\n'.join(dict.fromkeys(uris)))"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to read Guardrails Hub validators from config\guardrails.yaml"
}

$ToxicValidatorUri = "hub://guardrails/toxic_language"
$LocalValidatorUris = @($ValidatorUris | Where-Object { $_ -ne $ToxicValidatorUri })
if ($ValidatorUris -contains $ToxicValidatorUri) {
    Write-Output "Skipping local install for $ToxicValidatorUri; it runs in the guardrails-toxic Docker service."
    Write-Output 'Build/start it with: $env:GUARDRAILS_TOKEN="<your_guardrails_hub_token>"; docker compose build guardrails-toxic; docker compose up -d guardrails-toxic'
}

$FailedInstalls = @()
foreach ($validator in $LocalValidatorUris) {
    if (-not [string]::IsNullOrWhiteSpace($validator)) {
        & $GuardrailsExe hub install $validator
        if ($LASTEXITCODE -ne 0) {
            $FailedInstalls += $validator
        }
    }
}

if ($FailedInstalls.Count -gt 0) {
    Write-Output "Required Guardrails Hub validator install failed:"
    foreach ($validator in $FailedInstalls) {
        Write-Output "- $validator"
    }
    Write-Output "These validators are required by config\guardrails.yaml. Fix the Guardrails Hub package/index access and rerun:"
    foreach ($validator in $FailedInstalls) {
        Write-Output ".\.venv\Scripts\guardrails.exe hub install $validator"
    }
    throw "Guardrails Hub validator install failed."
}

$InstalledValidators = & $GuardrailsExe hub list
foreach ($validator in $LocalValidatorUris) {
    if (-not [string]::IsNullOrWhiteSpace($validator)) {
        $name = ($validator -split "/")[-1]
        if (-not ($InstalledValidators -match [regex]::Escape($name))) {
            throw "Guardrails Hub validator '$validator' was not listed after install."
        }
    }
}

$SmokeToxic = -not [string]::IsNullOrWhiteSpace($env:WORKFLOW_GUARDRAILS_TOXIC_URL)
if (-not $SmokeToxic) {
    Write-Output "WORKFLOW_GUARDRAILS_TOXIC_URL is not set; skipping toxic_language sidecar smoke validation."
    Write-Output "After Docker is running, set WORKFLOW_GUARDRAILS_TOXIC_URL=http://localhost:8011/validate and rerun this script to smoke test toxic_language."
}
$SmokeToxicLiteral = if ($SmokeToxic) { "True" } else { "False" }
try {
    & $PythonExe -c "from services.workflow_guardrails import WorkflowGuardrailService; WorkflowGuardrailService().validate_runtime(smoke_toxic=$SmokeToxicLiteral); print('Guardrails Hub runtime validation passed.')"
} catch {
    Write-Output "Guardrails Hub runtime validation failed."
    Write-Output "Suggested repair commands:"
    Write-Output '$env:GUARDRAILS_TOKEN="<your_guardrails_hub_token>"'
    Write-Output "docker compose build guardrails-toxic"
    Write-Output "docker compose up -d guardrails-toxic"
    Write-Output '$env:WORKFLOW_GUARDRAILS_TOXIC_URL="http://localhost:8011/validate"'
    Write-Output ".\.venv\Scripts\python.exe -m pip install sentence-transformers"
    throw
}
