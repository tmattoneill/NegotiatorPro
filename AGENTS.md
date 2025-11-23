# Repository Guidelines

## Project Structure & Module Organization
`backend/` hosts the FastAPI services plus persistence helpers in `schema/`, `migrations/`, and `data/`. The React + TypeScript client lives in `frontend/src/`, serves static assets from `public/`, and emits builds to `dist/`. Tests stay in `tests/`, automation scripts in `scripts/`, and references in `docs/`, `dev-docs/`, and `TESTING_GUIDE.md`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt -r requirements-test.txt` — create the backend environment.
- `./run-api.sh` — start FastAPI with reload at `http://localhost:8000`.
- `cd frontend && npm install && npm run dev` (or `npm run build`) — develop or bundle the React app.
- `pytest tests -m "not requires_docker" --cov=.` — run backend tests with coverage; add `--cov-report=html` when needed.
- `npm run lint` then `docker compose up -d` — lint the frontend and verify the full stack.

## Coding Style & Naming Conventions
Python code targets 3.8+, four-space indentation, and type hints; keep snake_case for modules/functions, PascalCase for classes, and constants in UPPER_SNAKE. Run `flake8` (length 127, complexity 10) and `black --check .` before pushing. React components stay PascalCase inside `frontend/src/components`, hooks camelCase under `src/hooks`, and ESLint + TypeScript strict mode (`npm run lint`) enforce hook rules and dependency lists.

## Testing Guidelines
`pytest.ini` controls discovery (`test_*.py`, `Test*`, `test_*`), markers (`unit`, `integration`, `docker`, `requires_api_key`), and a 300s timeout—tag slow or external tests accordingly. Maintain ≥80% coverage; `pytest --cov=. --cov-report=html` outputs `htmlcov/index.html`. Integration or vectorstore tests that hit Postgres or cloud APIs must run with `docker compose up` and document required env vars. When UI flows change, pair pytest results with a quick Vite preview referencing `TESTING_GUIDE.md`.

## Commit & Pull Request Guidelines
Use short, imperative commit subjects similar to `making tweaks to the UI` and `fixing small account bugs`. Pull requests should summarize behavior changes, link to issues, call out config or migration updates, and include screenshots for UI shifts. List the verification steps (`pytest`, `npm run lint`, docker smoke) and refresh the relevant docs whenever workflows move.

## Security & Configuration Tips
Never commit `.env`, API keys, or generated encryption keys; bootstrap them with `install.sh` and populate `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `POSTGRES_*`, and `ENCRYPTION_KEY` locally. Rotate the default credentials mentioned in scripts (`admin123`), keep `config.json`, `prompt_config.json`, and `admin_config.json` aligned with admin settings, and scrub sensitive negotiation files from `sources/` or `uploads/` before sharing logs because RAG traces can echo raw content.
