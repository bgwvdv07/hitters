#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/david/Desktop/screen/hitters"
PYTHON="/usr/bin/python3"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/grading.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# Same "yesterday" convention grade_picks.py uses internally.
DATE="$(date -d yesterday +%Y-%m-%d)"
GRADED_FILE="output/graded_picks_${DATE}.csv"

echo "=== Grading run started: $(date -Is) ===" | tee -a "$LOG_FILE"

# 1. Grade yesterday's picks against the MLB boxscore.
"$PYTHON" -m scripts.grade_picks \
  >> "$LOG_FILE" 2>&1

# 2. Fold the freshly graded picks into the cumulative history file that
#    train_hit_model.py reads. Without this step the retrain below just
#    keeps re-training on the same old graded_history.csv forever.
if [ -f "$GRADED_FILE" ]; then
  "$PYTHON" -m scripts.build_graded_history "$GRADED_FILE" --out output/graded_history.csv \
    >> "$LOG_FILE" 2>&1
else
  echo "WARNING: $GRADED_FILE not found, skipping history update and retrain" | tee -a "$LOG_FILE"
fi

# 3. Retrain models (had_hit_1, had_hit_2, had_run_1) on the updated history.
if [ -f "$GRADED_FILE" ]; then
  "$PYTHON" -m scripts.train_hit_model --history-file output/graded_history.csv -v \
    >> "$LOG_FILE" 2>&1
fi

echo "=== Grading run finished: $(date -Is) ===" | tee -a "$LOG_FILE"
