#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Enable corepack"
corepack enable || true

echo "[2/6] Install Node deps"
pnpm i -w

echo "[3/6] Install Python deps"
python -m pip install --upgrade pip
pip install -r api/requirements.txt

echo "[4/6] Lint"
pnpm -F web lint
ruff check api

echo "[5/6] Type-check"
pnpm -F web typecheck
mypy --config-file api/mypy.ini api

echo "[6/6] Build & Test"
pnpm -F web build
pytest -q

echo "All checks passed."

