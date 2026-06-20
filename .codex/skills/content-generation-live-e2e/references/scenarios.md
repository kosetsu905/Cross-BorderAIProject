# Content Generation E2E Scenarios

## multilingual_geo_visual

This is the default live regression suite. It mirrors the user's real Content Creation form behavior and stresses the localization leakage paths that previously affected German and Japanese runs.

### Frontend Submission

- Open the Streamlit dashboard.
- Select workflow `content`.
- Fill the Content Creation fields from `scenarios.json`.
- Toggle `Generate Reddit GEO` on.
- Toggle `Generate visual assets` on.
- Submit from the dashboard and copy the displayed `job_id`.

### Human-Visible Checks

- The dashboard shows `Submitted job <job_id>`.
- The job eventually reaches `completed`.
- Live previews show content generation progress for `de`, `ja`, and `en`.
- Reddit GEO Publishing Package appears with draft posts only.
- Content Visual Assets appears with at least one generated image or remote image URL.

### Machine Checks

- `localized_articles` covers `de`, `ja`, and `en`.
- `localized_entities` covers `de`, `ja`, and `en`, with non-empty subject/category/brand/voice/keywords.
- `seo_outputs`, `multimodal_outputs`, `visual_assets`, `reddit_geo_posts`, and `production_ready_assets` are present.
- `production_ready_assets` contains `localized_article`, `seo_metadata`, `visual_asset`, and `reddit_geo_post`.
- German and English user-facing SEO, visual, and Reddit fields do not contain `寒带登山套装`, `登山套装`, `寒带地区登山`, or `登山`.
- Japanese user-facing SEO, visual, and Reddit fields do not contain the full Chinese source phrases `寒带登山套装`, `登山套装`, or `寒带地区登山`; the shorter `登山` is allowed because it can be natural Japanese.
