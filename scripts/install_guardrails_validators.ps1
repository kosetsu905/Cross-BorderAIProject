$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$GuardrailsExe = Join-Path $ProjectRoot ".venv\Scripts\guardrails.exe"
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

$validators = @(
    "hub://guardrails/secrets_present",
    "hub://guardrails/detect_pii",
    "hub://guardrails/provenance_llm"
)

foreach ($validator in $validators) {
    & $GuardrailsExe hub install $validator
}

if ($env:WORKFLOW_GUARDRAILS_INSTALL_TOXIC_AI -in @("1", "true", "yes")) {
    & $GuardrailsExe hub install --no-install-local-models hub://guardrails/toxic_language
} else {
    Write-Output "Skipping hub://guardrails/toxic_language AI runtime. Deterministic toxic policy remains active."
}
