# Windows PowerShell equivalent of ingest_local.sh
# Usage: $env:MAX_ISSUES=200; pwsh scripts/ingest_local.ps1

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Path "data/raw" -Force | Out-Null

if (-not (Test-Path "data/raw/fastapi")) {
    git clone --depth 1 https://github.com/tiangolo/fastapi.git data/raw/fastapi
} else {
    git -C data/raw/fastapi pull --rebase
}

$maxIssues = if ($env:MAX_ISSUES) { $env:MAX_ISSUES } else { "2000" }
python -m fastapi_helper.ingest.run --max-issues $maxIssues
