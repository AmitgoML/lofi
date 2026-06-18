#!/bin/sh
set -eu

exec uvicorn lucy.app:app \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --access-log \
  --log-level info \
  --timeout-keep-alive 75
