#!/usr/bin/env bash
set -euo pipefail

PORTS=(3000 5001 8787)

echo "[stop] Looking for listeners on: ${PORTS[*]}"

for port in "${PORTS[@]}"; do
  pids="$(lsof -tiTCP:${port} -sTCP:LISTEN || true)"
  if [[ -z "$pids" ]]; then
    echo "[stop] Port ${port}: no listener"
    continue
  fi

  echo "[stop] Port ${port}: stopping PID(s) $(echo "$pids" | tr '\n' ' ')"
  kill $pids 2>/dev/null || true
done

sleep 1

echo "[stop] Remaining listeners:"
for port in "${PORTS[@]}"; do
  lsof -nP -iTCP:${port} -sTCP:LISTEN || echo "[stop] Port ${port}: clear"
done
