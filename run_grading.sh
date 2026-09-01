#!/usr/bin/env bash
# Grade yesterday's picks, fold them into graded_history.csv, retrain.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

LOG_FILE="$LOG_DIR/grading.log"

# Same "yesterday" convention grade_picks.py uses internally.
DATE="$(date -d yesterday +%Y-%m-%d)"
CANDIDATES_FILE="output/hit_candidates_${DATE}.csv"
GRADED_FILE="output/graded_picks_${DATE}.csv"

exec 9>"$LOG_DIR/.grading.lock"
if ! flock -n 9; then
  log "grading already running, skipping" >> "$LOG_FILE"
  exit 0
fi

{
  log "=== Grading run started for $DATE ==="

  # grade_picks.py raises FileNotFoundError when the candidates file is
  # missing, which under `set -e` aborts the whole run. Check first and
  # exit cleanly instead -- a no-lineup day is not an error.
  if [ ! -f "$CANDIDATES_FILE" ]; then
    log "SKIP: $CANDIDATES_FILE not found (no candidates generated for $DATE)"
    log "=== Grading run finished (nothing to do) ==="
    exit 0
  fi

  # 1. Grade against the MLB boxscore.
  if ! "$PYTHON" -m scripts.grade_picks; then
    rc=$?
    log "FAILED: grade_picks exited $rc"
    exit "$rc"
  fi

  if [ ! -f "$GRADED_FILE" ]; then
    log "WARN: grade_picks succeeded but $GRADED_FILE was not written"
    log "=== Grading run finished (no history update) ==="
    exit 0
  fi

  # 2. Fold into the cumulative history the trainer reads. Without this,
  #    the retrain below just re-fits the same stale rows forever.
  if ! "$PYTHON" -m scripts.build_graded_history "$GRADED_FILE" \
        --out output/graded_history.csv; then
    rc=$?
    log "FAILED: build_graded_history exited $rc"
    exit "$rc"
  fi

  # 3. Retrain had_hit_1 / had_hit_2 / had_run_1 on the updated history.
  #    Non-fatal: a failed retrain leaves yesterday's bundles in place,
  #    which is better than losing the freshly graded rows.
  if "$PYTHON" -m scripts.train_hit_model_v2 \
        --history-file output/graded_history.csv -v; then
    log "retrain OK"
  else
    log "WARN: retrain failed, keeping previous model bundles"
  fi

  log "=== Grading run finished ==="
} >> "$LOG_FILE" 2>&1
