#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw

# Clone or pull FastAPI repo
if [[ ! -d data/raw/fastapi ]]; then
  git clone --depth 1 https://github.com/tiangolo/fastapi.git data/raw/fastapi
else
  git -C data/raw/fastapi pull --rebase
fi

# Run ingestion (MAX_ISSUES defaults to 2000 for dev; set to 5500 for full corpus)
python -m fastapi_helper.ingest.run --max-issues "${MAX_ISSUES:-2000}"
