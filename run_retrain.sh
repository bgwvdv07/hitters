#!/usr/bin/env bash
# Standalone retrain. run_grading.sh already retrains after each grading
# pass, so this is for manual use after a history rebuild or backfill.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

LOG_FILE="$LOG_DIR/retrain.log"
HISTORY_FILE="${1:-output/graded_history.csv}"

exec 9>"$LOG_DIR/.retrain.lock"
if ! flock -n 9; then
  log "retrain already running, skipping" >> "$LOG_FILE"
  exit 0
fi

{
  log "=== Retrain started (history: $HISTORY_FILE) ==="

  if [ ! -f "$HISTORY_FILE" ]; then
    log "FATAL: $HISTORY_FILE not found"
    exit 1
  fi

  if "$PYTHON" -m scripts.train_hit_model_v2 --history-file "$HISTORY_FILE" -v; then
    log "=== Retrain finished OK ==="
  else
    rc=$?
    log "=== Retrain FAILED (exit $rc) ==="
    exit "$rc"
  fi
} >> "$LOG_FILE" 2>&1
