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

status=0
run() {
    echo "==> $*"
    "$PY" -m "$@" || status=1
}

run flake8 .
run pylint main_sensor.py source tests # If more main's should be tested add a similar line. 
run mypy
run pytest

if [ "$status" -eq 0 ]; then
    echo "==> All checks passed"
else
    echo "==> Some checks FAILED"
fi
exit "$status"
