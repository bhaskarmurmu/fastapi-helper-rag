#!/usr/bin/env bash
# One-shot local setup. Run once after cloning.
set -euo pipefail

echo "==> Checking Python 3.11..."
python --version | grep -q "3.11" || { echo "Python 3.11 required"; exit 1; }

echo "==> Installing uv..."
pip install uv --quiet

echo "==> Creating virtualenv..."
uv venv

echo "==> Installing dependencies..."
uv pip install -r requirements.txt -r requirements-dev.txt

echo "==> Installing pre-commit hooks..."
.venv/bin/pre-commit install

echo "==> Copying .envrc.example to .envrc..."
if [[ ! -f .envrc ]]; then
  cp .envrc.example .envrc
  echo "    -> Edit .envrc and fill in your API keys before running."
fi

echo ""
echo "Setup complete. Next steps:"
echo "  1. Edit .envrc with your API keys"
echo "  2. docker compose up -d"
echo "  3. MAX_ISSUES=200 make ingest"
echo "  4. make dev"
