#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/david/hitters"
PYTHON="/usr/bin/python3"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/lineups.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

echo "=== Lineups run started: $(date -Is) ===" | tee -a "$LOG_FILE"

"$PYTHON" -m scripts.mlb_hit_finder --date "$(date +%Y-%m-%d)" \
  >> "$LOG_FILE" 2>&1

echo "=== Lineups run finished: $(date -Is) ===" | tee -a "$LOG_FILE"