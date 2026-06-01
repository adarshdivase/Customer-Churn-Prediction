#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/venv/bin/python"

[[ -x "$PY" ]] || { python3 -m venv "$ROOT/venv" && "$ROOT/venv/bin/pip" install -r "$ROOT/requirements.txt"; }

"$PY" "$ROOT/scripts/train_model.py"
echo "Open http://127.0.0.1:8501"
command -v open >/dev/null && open "http://127.0.0.1:8501" || true
exec "$PY" -m streamlit run "$ROOT/app.py" --server.port 8501
