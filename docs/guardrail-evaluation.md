# Guardrail Evaluation Plan

## Scope and ground truth

This harness tests the project workflow guardrails configured in `config/guardrails.yaml` for the `support` and `content` workflows. It does not test the Codex desktop policy hook.

The source-controlled golden set is `config/mlflow/guardrail_evaluation_dataset.json`:

- exactly 200 ordered cases, `G001` through `G200`;
- 140 English and 60 Chinese cases;
- input, output, action, and provenance stages;
- positive, negative, hard-negative, cross-policy, validator-fault, cache, and known-gap cases;
- a positive and a negative case for every configured validator and governed action type;
- synthetic identities, credentials, and business data only.

Each case has two independent labels:

- `policy_expectation`: desired safety semantics and the PR gate target;
- `current_expectation`: the current YAML/runtime contract, used to expose unintentional regressions separately from known safety gaps.

The authoritative full case text is the JSON file. Expectation profiles at the top keep repeated labels consistent and make reviewer changes auditable.

## Execution flow

1. Strict Pydantic validation checks the dataset schema, contiguous IDs, exact count, stage contracts, dual-label invariants, and judge URI.
2. Coverage validation compares the cases with `config/guardrails.yaml` and requires positive/negative coverage for every configured validator and action type.
3. All selected cases execute through `WorkflowGuardrailService`: raw Secrets/PII detection runs in memory, every string is redacted recursively, and only the redacted semantic view may reach Qwen or observability.
4. Code scorers compare detection, policy IDs, action, severity, masking, skipped validators, cache behavior, and runtime-error handling against both labels in-process.
5. MLflow logs the redacted per-case score table and aggregate metrics in one deterministic run, avoiding thousands of redundant per-row trace jobs.
6. Four bounded MLflow `Guidelines` judges audit prompt injection, toxicity, provenance, and restricted-claim decisions. The default model is `openrouter:/qwen/qwen3.7-plus`, temperature is zero, and total calls are capped at 200. These judges supplement but never replace fixed-label metrics.
7. The balanced gate returns exit code 0, 1, or 2 for pass, quality failure, or infrastructure failure.

## Metrics and default PR thresholds

| Metric | Threshold |
| --- | ---: |
| Macro F1 across six validators | >= 0.90 |
| Overall false-positive rate | <= 0.05 |
| Per-policy false-positive rate | <= 0.10 |
| Minimum high-risk policy recall | >= 0.90 |
| Secret recall | 1.00 |
| PII detection plus masking recall | >= 0.95 |
| Desired action accuracy | >= 0.90 |
| Desired severity accuracy | >= 0.95 |
| Privacy leakage rate | 0.00 |
| Unexpected validator error rate | <= 0.01 |
| Configured-target coverage | 1.00 |
| Qwen judge pass rate | >= 0.85 |
| Qwen judge error rate | <= 0.10 |
| End-to-end suite duration | <= 1800 seconds |

Per-validator and per-language/per-stage precision, recall, F1, false-positive rates, and confusion matrices are also logged. PII detection recall, conditional masking recall, joint detection-plus-masking recall, toxicity recall, Qwen degraded rate, and detector versions are first-class MLflow metrics.

## MLflow settings

Required non-secret settings:

```env
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_GUARDRAIL_EXPERIMENT_NAME=cross-border-ai-guardrails
MLFLOW_GUARDRAIL_EVALUATION_DATASET_NAME=guardrail-regression-v1
MLFLOW_GUARDRAIL_MAX_CASES=200
MLFLOW_GUARDRAIL_MAX_JUDGE_CALLS=200
MLFLOW_GUARDRAIL_SUITE_TIMEOUT_SECONDS=1800
MLFLOW_GUARDRAIL_JUDGE_MODEL=openrouter:/qwen/qwen3.7-plus
MLFLOW_AUTOMATIC_EVALUATION_ENABLED=false
MLFLOW_GENAI_JUDGE_DEFAULT_MODEL=openrouter:/qwen/qwen3.7-plus
```

The tracking server must use PostgreSQL, MySQL, SQLite, or MSSQL; the project's monitoring stack already uses PostgreSQL. Run `scripts/bootstrap_mlflow_guardrail_evaluation.py` once after startup and again after dataset/judge changes; it is idempotent. MLflow OSS intentionally does not register arbitrary custom code scorers, so deterministic scorers execute from this repository during offline evaluation; the built-in LLM judges are registered in MLflow.

Production semantic review additionally requires an `openrouter_qwen37` entry in `LLM_PROFILES_JSON` whose key environment is `OPENROUTER_GUARDRAIL_API_KEY`, plus `WORKFLOW_GUARDRAILS_SEMANTIC_MODEL=openrouter_qwen37` and `WORKFLOW_GUARDRAILS_SEMANTIC_TIMEOUT_SECONDS=8`. Qwen receives redacted text only. A failed Qwen request falls back to local detectors for ordinary content and fails closed to review for Gmail, WhatsApp, and publish actions.

Store the judge credential `OPENROUTER_API_KEY` only in `.env.guardrail-eval` on the evaluator host or in the GitHub Actions secret with the same name. The evaluator deliberately ignores an `OPENROUTER_API_KEY` loaded only from the shared `.env`. Production FastAPI/Celery may receive the separate `OPENROUTER_GUARDRAIL_API_KEY`; never inject either key into the MLflow server, Streamlit, frontend builds, or client-side code. The live runtime validators currently also need `OPENAI_API_KEY`; CI stores it independently from the Qwen judge key.

Automatic Evaluation stays disabled. Judges are called only by `scripts/evaluate_guardrails.py`, which makes cost, timing, and dataset selection explicit.

Run the evaluator in a dedicated container so local ML models do not compete with the live FastAPI process for the same container memory:

```powershell
docker compose run --rm --no-deps fastapi python scripts/evaluate_guardrails.py
```

## Privacy and trace contents

MLflow Evaluation Dataset inputs contain only `case_id`. Expectations contain the dual labels and dataset digest. Evaluation runs may contain sanitized candidate text and masked guardrail findings, but never the source payload object. The harness sets `raw_payloads_uploaded=false` in dataset/run metadata.

The evaluator report contains case IDs, normalized decisions, scores, metrics, and gate failures. It does not include source payloads or provider credentials.

## Calibration corpus

`config/mlflow/guardrail_calibration_dataset.json` contains 600 cases: 100 for each of the six policies, evenly split between English/Chinese and positive/hard-negative samples. It is generated reproducibly and remains separate from the frozen 200-case release suite:

```powershell
.\.venv\Scripts\python.exe scripts\generate_guardrail_calibration_dataset.py
.\.venv\Scripts\python.exe scripts\calibrate_guardrails.py
```

Tune thresholds only against this corpus. The 200 release cases may change labels only through reviewed adjudication.

## CI

`.github/workflows/guardrail-regression.yml` runs dataset/unit validation for every pull request. For same-repository pull requests and manual dispatches, it builds the validator image, starts a SQLite-backed MLflow server, runs all 200 cases and up to 200 Qwen judges, and uploads the redacted report.

Configure these repository secrets:

- `GUARDRAILS_TOKEN` for image-time Hub validator installation;
- `OPENROUTER_API_KEY` for Qwen judges;
- `OPENAI_API_KEY` for the currently configured live prompt-injection/provenance validators.

Fork pull requests do not receive secrets and therefore run only the deterministic dataset/unit validation job.
