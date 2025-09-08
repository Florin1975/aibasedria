# regimpact-ai

Regulatory Impact Assessment (RIA) demo monorepo with:

- web/ (Astro + Tailwind) — Landing, Wizard, Playground
- api/ (FastAPI) — Local-only AI analysis pipeline
- docs/ (MkDocs Material) — Project overview
- .devcontainer/ — Node 18 + Python 3.11 + pnpm + uvicorn
- GitHub Actions — CI and Pages deploy

Never commit secrets. Pages is static; AI only via local API.

## Open in Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/OWNER/regimpact-ai?quickstart=1)

Replace `OWNER` above with your GitHub username when this repo is pushed.

## 90‑sec QuickStart

Prereqs: In Codespaces this is preinstalled. Locally use Dev Containers.

```bash
# Install deps at workspace root
pnpm i -w

# Start the web app (http://localhost:4321)
pnpm -F web dev

# In another terminal: set up API
cp api/.env.example api/.env
# edit api/.env with your OpenAI key (optional)
pnpm -F api dev  # http://localhost:8000
```

Hosted Pages uses a base path `/regimpact-ai/`. Local API only.

## Local AI vs Hosted Demo

- Local: The "Suggest with AI" button calls your local FastAPI (`http://localhost:8000`).
- Hosted Pages: No runtime AI. The button shows a tooltip to enable local API.
- Handoff: Use "Open prompt in ChatGPT" to copy/paste prompts.

## Progress

| Area | Acceptance | Status |
| --- | --- | --- |
| Task 0 — Bootstrap | AC-01..AC-09 | ✅ |
| Task 1 — Wizard | AC-11..AC-14 | ✅ |
| Task 2 — API | AC-21..AC-25 | ✅ |
| Task 3 — Pages+Docs | AC-31..AC-34 | ✅ |
| Task 4 — Finishing | AC-41..AC-44 | ✅ |

Each PR updates this table with ✅ entries.

## Repo Structure

```
web/  # Astro site (static)
api/  # FastAPI local API
docs/ # MkDocs site
.devcontainer/
.github/workflows/
```

## Pages Base Path

Astro is configured with base `/regimpact-ai/` so internal links work on GitHub Pages.

## Contributing

Guardrails:
- No secrets in repo; use `.env` locally (`api/.env`).
- No browser-side OpenAI calls; local API only.
- Deterministic CI; pinned versions.

## Next Steps

- Create the GitHub repo `regimpact-ai`, push, and enable Pages (GitHub Actions).
- Replace `OWNER` in README and `mkdocs.yml` with your GitHub username.
- Open PRs using the prepared bodies in `docs/prs/` (or copy from below).
- Iterate on the wizard UX; connect to richer local analysis if desired.

