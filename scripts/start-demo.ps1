$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$VenvPy = Join-Path $Root "venv\Scripts\python.exe"

Write-Host "RetainAI Enterprise Demo" -ForegroundColor Cyan

if (-not (Test-Path $VenvPy)) {
    python -m venv (Join-Path $Root "venv")
    & (Join-Path $Root "venv\Scripts\pip.exe") install -r (Join-Path $Root "requirements.txt")
}

Write-Host "Pre-training model (skip if models/ exists)..." -ForegroundColor Gray
& $VenvPy (Join-Path $Root "scripts\train_model.py")

$url = "http://127.0.0.1:8501"
Write-Host "Starting Streamlit at $url" -ForegroundColor Green
Start-Process $url
& $VenvPy -m streamlit run (Join-Path $Root "app.py") --server.port 8501
