# Local AI setup (safe)

This project uses a local FastAPI service for AI assistance. No secrets are embedded in the web app.

Steps
1. Install dependencies: `pnpm i -w` and `pip install -r api/requirements.txt`
2. Copy env: `cp api/.env.example api/.env` and set `OPENAI_API_KEY` (optional)
3. Run API: `pnpm -F api dev` (http://localhost:8000)
4. Run Web: `pnpm -F web dev` (http://localhost:4321)

When running locally, the Wizard’s “Suggest with AI (local)” calls `http://localhost:8000/api/analyze`.

