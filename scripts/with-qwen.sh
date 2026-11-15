#!/usr/bin/env bash
# Run any forecast-log command against the Qwen gateway, keeping the key in .env.
#
#   scripts/with-qwen.sh forecast-log agent-eval --limit 96 --stride 60 --workers 6 --merge
#
# `.env` is git-ignored; this script exports FORECAST_LOG_* from the QWEN_* entries
# so the key never lands in shell history or in a committed file.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
set -a
# shellcheck disable=SC1091
. "$root/.env"
set +a
: "${QWEN_API_KEY:?set QWEN_API_KEY in .env}"
: "${QWEN_BASE_URL:?set QWEN_BASE_URL in .env}"
export FORECAST_LOG_BASE_URL="$QWEN_BASE_URL"
export FORECAST_LOG_API_KEY="$QWEN_API_KEY"
export FORECAST_LOG_MODEL="${FORECAST_LOG_MODEL:-${QWEN_MODEL:-tongyi-xiaomi-analysis-flash}}"
[ -n "${QWEN_PRICE_INPUT:-}" ] && export FORECAST_LOG_PRICE_INPUT="$QWEN_PRICE_INPUT"
[ -n "${QWEN_PRICE_OUTPUT:-}" ] && export FORECAST_LOG_PRICE_OUTPUT="$QWEN_PRICE_OUTPUT"
exec "$@"
