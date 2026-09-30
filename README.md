# finger_exoskeleton

An adaptive hand exoskeleton project for Parkinsonian tremor suppression, using surface EMG to detect grasp intent and tremor-related muscle activity. The system focuses on a 2-DOF thumb–index design and aims to apply adaptive stiffness or damping only when tremor is present, reducing involuntary motion while preserving natural voluntary movement.

## Goals

- Detect grasp intent and tremor-related muscle activity from surface EMG.
- Actuate a 2-DOF thumb–index mechanism.
- Apply adaptive stiffness or damping only while tremor is present.
- Preserve natural voluntary movement.

## Requirements

- Python 3.10 or newer

## How to run

_To be written._

## Testing

The test scripts create a `.venv`, install the dev dependencies from `requirements-dev.txt`, and then run flake8, pylint, mypy and pytest.

Windows:

```bat
test.bat
```

Linux / macOS:

```bash
./test.sh
```

Set the `PYTHON` environment variable to pick the interpreter used to create the virtualenv (default: `py -3` on Windows, `python3` on Linux).

## Project structure

_To be written._

## How the project works
_To be written._

# Interpreting EMG signals
_To be written._

# Control loop
_To be written._

# Actuation
_To be written._