# Shared setup for all run_*.sh scripts. Sourced, not executed.
#
# Derives PROJECT_DIR from this file's own location so the scripts keep
# working if the repo is moved or checked out somewhere else. Previously
# each script hardcoded a different absolute path, which is how
# run_grading.sh ended up pointing at a stale Desktop/screen/ copy.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT_DIR/venv/bin/python"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

# `python -m scripts.foo` needs the project root importable. Cron does not
# inherit your shell's environment, so set it explicitly.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -x "$PYTHON" ]; then
  echo "FATAL: no interpreter at $PYTHON -- create the venv first" >&2
  exit 1
fi

# Fail loudly at startup rather than 40 minutes into a scrape.
if ! "$PYTHON" - <<'PYCHECK' 2>/dev/null
import pandas, numpy, requests, bs4, lxml, sklearn
PYCHECK
then
  echo "FATAL: venv at $PYTHON is missing required packages" >&2
  echo "       run: $PYTHON -m pip install -r requirements.txt" >&2
  exit 1
fi

log() { echo "$(date -Is) | $*"; }
