---
name: content-generation-live-e2e
description: "Run the project's live Content Generation frontend E2E regression. Use when Codex needs to mimic the user's real dashboard behavior: submit a content workflow through Streamlit Content Creation fields, enable Reddit GEO and visual asset generation, poll the FastAPI job, and verify multilingual localized entities, SEO, image prompts/assets, Reddit GEO drafts, and production-ready assets without mocking the content workflow."
---

# Content Generation Live E2E

## Overview

Use this skill to reproduce the user's real Content Generation test: submit a content workflow from the Streamlit dashboard, enable Reddit GEO and image generation, then verify the resulting job payload and artifacts through the same FastAPI endpoints the dashboard uses.

Keep the workflow live at the UI/workflow boundary. Do not mock `run_content_crew`, image generation, Reddit GEO, job polling, or result validation when performing the E2E check.

## Safety Rules

- Do not publish Reddit posts; verify `reddit_geo_posts` drafts only.
- Do not paste or print API keys, bearer tokens, cookies, or other secrets.
- Treat `OPENAI_API_KEY` as required for the live image-generation E2E. If it is missing, fail the E2E with an environment readiness issue.
- Treat `SERPER_API_KEY` as optional for Reddit source enrichment. If missing, accept low-confidence fallback posts but record a warning.
- Keep image cost controlled: use `image_generation_count=1`, `image_quality=low`, and `image_size=1024x1024` unless the user explicitly asks for a heavier run.
- If browser automation cannot submit through the dashboard, print the prepared fields and ask the user to submit manually, then verify the copied `job_id`.

## Workflow

1. Read `references/scenarios.md` for the human-readable checklist and `references/scenarios.json` for the machine-readable scenarios.
2. Generate the exact UI field values:

   ```powershell
   .\.venv\Scripts\python.exe .\.codex\skills\content-generation-live-e2e\scripts\run_content_generation_live_e2e.py --suite multilingual_geo_visual --print-ui-fields
   ```

3. Open the Streamlit admin dashboard, normally:

   ```powershell
   streamlit run .\admin_dashboard.py
   ```

   Use `http://localhost:8501` unless the local run prints a different URL.

4. In the dashboard:
   - Confirm the sidebar API base URL points to `http://localhost:8000` or the requested `API_BASE_URL`.
   - Select workflow `content`.
   - Fill the Content Creation fields from the printed scenario.
   - Enable `Generate Reddit GEO`.
   - Enable `Generate visual assets`.
   - Leave image count `1`, quality `low`, and size `1024x1024`.
   - Click `Submit workflow`.
   - Copy the displayed `job_id`.

5. Verify the job with:

   ```powershell
   .\.venv\Scripts\python.exe .\.codex\skills\content-generation-live-e2e\scripts\run_content_generation_live_e2e.py --suite multilingual_geo_visual --run-id <RUN_ID> --job-id <JOB_ID> --verify-job
   ```

6. Use `--submit-api` only as an explicit fallback when the user accepts that the submission will bypass the frontend:

   ```powershell
   .\.venv\Scripts\python.exe .\.codex\skills\content-generation-live-e2e\scripts\run_content_generation_live_e2e.py --suite multilingual_geo_visual --submit-api --verify-job
   ```

## Environment

The script reads:

- `API_BASE_URL`, default `http://localhost:8000`
- `API_BEARER_TOKEN`, optional local auth token
- `OPENAI_API_KEY`, required for live visual asset generation
- `SERPER_API_KEY`, optional Reddit GEO source enrichment

The report is saved under `artifacts/content_generation_e2e/<run_id>.json`.

## Acceptance Criteria

- The workflow is submitted from the Streamlit frontend unless `--submit-api` was explicitly requested.
- The final job status is `completed`.
- The result includes localized articles, localized entities, SEO outputs, multimodal outputs, visual assets, Reddit GEO posts, and production-ready assets.
- `de`, `ja`, and `en` are all represented in localized articles and localized entities.
- SEO metadata, visual prompts, and Reddit GEO drafts do not leak the Chinese source phrases listed in the scenario checks.
- At least one visual asset has an existing local artifact path under `artifacts/content_creation` or a remote `asset_url`.
- Reddit GEO posts are drafts for manual review and are never posted automatically.
