---
name: support-inbox-live-e2e
description: "Run the project's live QQ Mail to Gmail omni-channel support inbox regression workflow. Use when the user wants Codex to mimic their real customer-service testing behavior: write QQ emails to the Gmail inbox configured by GMAIL_SENDER_EMAIL, sync latest Gmail in the Streamlit Omni-channel support inbox, and verify the balanced 12-message pre-sales, order, post-sales, multilingual, and high-risk support scenarios without mocking the support workflow internals."
---

# Support Inbox Live E2E

## Overview

Use this skill to reproduce the user's real support workflow test: send live emails from QQ Mail to the Gmail inbox, sync them into the Omni-channel support inbox, and verify the resulting support drafts and structured payloads.

Keep the workflow live at the email/channel boundary. Do not mock `SupportInboxStore`, `run_support_crew`, Gmail sync, conversation detail retrieval, or support draft validation when performing the E2E check.

## Safety Rules

- Never ask for or store QQ/Gmail passwords, 2FA codes, Gmail refresh tokens, bearer tokens, or cookies.
- Do not click approve-send during this regression. Verify drafts only.
- If the system auto-dispatches a response, do not block it in this skill; record `auto_dispatch_observed` and outbound message evidence in the report.
- If browser automation cannot reuse the user's logged-in QQ Mail session, print the prepared messages and ask the user to send them manually.
- Treat any raw JSON/code fence in `draft_response` as a regression failure.

## Workflow

1. Read `references/scenarios.md` for the human-readable scenario guide and `references/scenarios.json` for the machine-readable checks.
2. Generate a run id and message set:

   ```powershell
   python .\.codex\skills\support-inbox-live-e2e\scripts\run_support_inbox_live_e2e.py --suite balanced --print-messages
   ```

   If local `python` is unavailable, use the FastAPI container:

   ```powershell
   docker compose exec -T fastapi python /app/.codex/skills/support-inbox-live-e2e/scripts/run_support_inbox_live_e2e.py --suite balanced --print-messages
   ```

3. Send all selected messages from the user's logged-in QQ Mail account to the Gmail address configured by `GMAIL_SENDER_EMAIL`.
   - Prefer browser automation only if the logged-in QQ Mail session is available.
   - Otherwise show the subjects and bodies to the user and wait for confirmation that they sent the messages.
4. Sync and verify through the backend/API path used by the Streamlit support inbox:

   ```powershell
   python .\.codex\skills\support-inbox-live-e2e\scripts\run_support_inbox_live_e2e.py --suite balanced --run-id <RUN_ID> --sync-and-verify
   ```

   Docker fallback:

   ```powershell
   docker compose exec -T fastapi python /app/.codex/skills/support-inbox-live-e2e/scripts/run_support_inbox_live_e2e.py --suite balanced --run-id <RUN_ID> --sync-and-verify
   ```

5. If the user specifically wants a UI check, open the Streamlit admin dashboard, expand "Omni-channel support inbox", click "Sync latest Gmail", select each conversation by the run marker, and compare the UI against the script report.

## Environment

The script reads:

- `API_BASE_URL`, default `http://localhost:8000`
- `API_BEARER_TOKEN`
- `GMAIL_SENDER_EMAIL`, loaded from the project `.env` when not already set in the shell; this is the live E2E recipient.

Backend Gmail sync must have `GMAIL_SYNC_ENABLED=true` and usable Gmail access/refresh-token credentials. The Streamlit dashboard is started separately with:

```powershell
streamlit run .\admin_dashboard.py
```

## Scenarios

The standard suites are:

- `balanced`: 12 messages covering catalog hits, catalog uncertainty, unknown product, correct/wrong/missing tracking, marker pollution, hygiene-sensitive return, defective item, expired return window, and Chinese escalation.
- `smoke`: 4 core messages covering headset catalog hit, correct tracking `C88943021`, wrong tracking `C99943021`, and worn bra return.
- Custom: `--scenarios pre_sales_catalog_headset,order_tracking_found`.

Each subject must include `CB-SUPPORT-E2E-{scenario}-{run_id}`. The marker must also appear in the body so Gmail search and conversation matching can find it.
The marker must be on its own line and must not appear in customer-facing drafts.

## Acceptance Criteria

- All selected marker conversations are found after Gmail sync.
- Each conversation has a non-empty customer-facing `draft_response`.
- No draft contains fenced JSON, raw JSON object output, `response_type`, or `labels.example.local`.
- Intents match the expected scenario.
- The wrong tracking number scenario does not expose correct local tracking facts.
- Marker/run-id values are not treated as tracking or order identifiers.
- The JSON report records `conversation_status`, `requires_approval`, outbound message ids, and whether auto-dispatch was observed.
- A JSON report is saved under `artifacts/support_inbox_e2e/<run_id>.json`.
