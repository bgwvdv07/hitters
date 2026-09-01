#!/usr/bin/env bash
# Build today's hit candidates. Run mid-afternoon, after lineups post.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

LOG_FILE="$LOG_DIR/lineups.log"
DATE="$(date +%Y-%m-%d)"

# Prevent overlapping runs. The grading log shows three invocations
# inside 60 seconds on 2026-08-11, which means cron double-fired.
exec 9>"$LOG_DIR/.lineups.lock"
if ! flock -n 9; then
  log "lineups already running, skipping" >> "$LOG_FILE"
  exit 0
fi

{
  log "=== Lineups run started for $DATE ==="

  if "$PYTHON" -m scripts.mlb_hit_finder --date "$DATE" -v; then
    log "=== Lineups run finished OK ==="
  else
    rc=$?
    log "=== Lineups run FAILED (exit $rc) ==="
    exit "$rc"
  fi
} >> "$LOG_FILE" 2>&1
