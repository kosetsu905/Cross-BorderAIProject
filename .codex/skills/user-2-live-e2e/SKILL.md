---
name: user-2-live-e2e
description: "Run the project's live User 2.0 regression workflow. Use when Codex needs to mimic the user's real Streamlit user-management behavior: register/login by email and phone, use simulated OAuth, manage profile/security/connected accounts/billing/subscription through the Users UI, and verify the resulting /api/v1/users state without exposing tokens or raw JSON in the UI."
---

# User 2.0 Live E2E

## Overview

Use this skill to reproduce the user's real User 2.0 workflow from the Streamlit dashboard, then verify the resulting user state through the FastAPI `/api/v1/users/*` endpoints.

Keep the workflow live at the UI/API boundary. Do not mock `UserService`, user routes, database records, session validation, OAuth provider records, payment method records, or subscription state when performing the E2E check.

## Safety Rules

- Do not ask for or store real passwords, OAuth credentials, payment card numbers, CVV/CVC values, bearer tokens, cookies, or reset tokens.
- Do not use Codex's in-app browser for this regression. Follow the same style as `support-inbox-live-e2e`: prefer the user's local Chrome/dashboard session when it is available; otherwise print the exact UI steps and ask the user to complete them manually.
- Use only generated demo credentials from `references/scenarios.json`.
- Treat any visible `access_token`, `token_hash`, `reset_token`, password, raw JSON response, or stack trace in the Users UI as a regression failure.
- Payment methods are simulated only. Use tokenized or masked metadata; never enter a full card number or real payment secret.

## Workflow

1. Read `references/scenarios.md` for the human-readable UI checklist and `references/scenarios.json` for the machine-readable scenario data.
2. Generate a run id and exact UI inputs:

   ```powershell
   .\.venv\Scripts\python.exe .\.codex\skills\user-2-live-e2e\scripts\run_user_2_live_e2e.py --suite full --print-ui-scenario
   ```

3. Open the Streamlit admin dashboard in the user's local Chrome session, normally:

   ```powershell
   streamlit run .\admin_dashboard.py
   ```

   Use `http://localhost:8501` unless the local run prints a different URL. If local Chrome automation cannot reuse the user's session, show the printed UI scenario and wait for the user to complete it.

4. In the dashboard:
   - Select `View` -> `Users`.
   - Complete the printed email, phone, simulated OAuth, profile, security, connected account, billing, and subscription steps.
   - Confirm the UI never displays tokens, reset tokens, token hashes, raw JSON responses, or JSON-shaped error text.

5. Verify the UI-created state:

   ```powershell
   .\.venv\Scripts\python.exe .\.codex\skills\user-2-live-e2e\scripts\run_user_2_live_e2e.py --suite full --run-id <RUN_ID> --verify-ui-run
   ```

6. Use API fallback only when the user explicitly accepts bypassing the frontend:

   ```powershell
   .\.venv\Scripts\python.exe .\.codex\skills\user-2-live-e2e\scripts\run_user_2_live_e2e.py --suite full --run-api-fallback
   ```

## Environment

The script reads:

- `API_BASE_URL`, default `http://localhost:8000`
- Optional `.env` values when `python-dotenv` is installed

The report is saved under `artifacts/user_2_e2e/<run_id>.json`.

## Acceptance Criteria

- The default run is performed from the Streamlit `Users` UI unless `--run-api-fallback` was explicitly requested.
- Email registration/login, phone registration/login, and simulated OAuth login are exercised with generated demo data.
- Logged-out UI shows only login/register entry points; account tabs are available only after login.
- Profile, password change, password reset, OAuth link/unlink, payment method add/default/remove, subscription, and cancellation are exercised.
- User-facing UI does not reveal `access_token`, `token_hash`, `reset_token`, passwords, raw JSON responses, or JSON validation bodies.
- API verification confirms final profile, auth providers, payment methods, subscription state, phone login, and email password state.
- The JSON report contains no plaintext passwords, session tokens, reset tokens, token hashes, cookies, or bearer tokens.
