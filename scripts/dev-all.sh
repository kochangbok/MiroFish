#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

cleanup() {
  if [[ -n "${DEV_ALL_PID:-}" ]]; then
    kill "$DEV_ALL_PID" 2>/dev/null || true
    wait "$DEV_ALL_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

npx concurrently --kill-others \
  -n "bridge,backend,frontend" \
  -c "magenta,green,cyan" \
  "npm run bridge" \
  "npm run backend" \
  "npm run frontend" &
DEV_ALL_PID=$!

node "$ROOT_DIR/scripts/health-check.mjs" --wait

echo
echo "[health] All services look reachable."
echo "[health] Frontend: http://localhost:3000"
echo "[health] Backend : http://localhost:5001"
echo "[health] Bridge  : http://127.0.0.1:8787/health"
echo

wait "$DEV_ALL_PID"
