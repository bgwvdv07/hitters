#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/david/hitters"
PYTHON="/usr/bin/python3"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/retrain.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

echo "=== Retrain started: $(date -Is) ===" | tee -a "$LOG_FILE"

"$PYTHON" -m scripts.train_hit_model_v2 --history-file output/graded_history.csv -v \
  >> "$LOG_FILE" 2>&1

echo "=== Retrain finished: $(date -Is) ===" | tee -a "$LOG_FILE"