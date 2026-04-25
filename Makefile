# Windows note: install Make via 'winget install GnuWin32.Make' or use
# 'choco install make'. Alternatively run the commands directly in PowerShell.

.PHONY: dev ingest eval test test-int test-all lint format typecheck clean

dev:
	uvicorn fastapi_helper.api.main:app --reload --port 8000

# Linux/macOS: bash scripts/ingest_local.sh
# Windows PowerShell: pwsh scripts/ingest_local.ps1
ingest:
	bash scripts/ingest_local.sh

eval:
	python -m eval.run_eval --eval-set eval/eval_set.jsonl

eval-seed:
	python -m eval.run_eval --eval-set eval/eval_set_seed.jsonl --output eval/results/ci.json

test:
	pytest -m "not integration" --maxfail=1 -q

test-int:
	pytest -m integration -v

test-all:
	pytest --maxfail=1 -q

lint:
	ruff check src/ tests/ eval/

format:
	ruff format src/ tests/ eval/

typecheck:
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
