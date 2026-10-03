#!/bin/sh
set -eu

# 1. Make sure the model is present (downloads only when missing or changed).
python scripts/fetch_model.py

# 2. Start the server. Render/Railway/Fly give the port in $PORT.
exec python cli.py serve --config serve.yaml --port "${PORT:-8000}"
