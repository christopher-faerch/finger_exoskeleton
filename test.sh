#!/usr/bin/env bash
# Run linters, type checks and tests (Linux / macOS).
# Usage: ./test.sh            (set PYTHON=python3.12 to pick an interpreter)
set -u
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=".venv"

if [ ! -d "$VENV" ]; then
    echo "==> Creating virtualenv in $VENV"
    "$PYTHON" -m venv "$VENV" || exit 1
fi
PY="$VENV/bin/python"

echo "==> Installing dev dependencies"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r requirements-dev.txt || exit 1
# PyTorch is optional: without it the GRU model tests are skipped
"$PY" -m pip install --quiet -r requirements-ml.txt || echo "==> PyTorch not installed: GRU model tests are skipped"

status=0
run() {
    echo "==> $*"
    "$PY" -m "$@" || status=1
}

run flake8 .
run pylint source tests
run pylint run_emg.py train_gru.py evaluate_gru.py evaluate_injection.py plot_recording.py settings.py
run mypy
run pytest

if [ "$status" -eq 0 ]; then
    echo "==> All checks passed"
else
    echo "==> Some checks FAILED"
fi
exit "$status"
