# finger_exoskeleton

An adaptive hand exoskeleton project for Parkinsonian tremor suppression, using surface EMG to detect grasp intent and tremor-related muscle activity. The system focuses on a 2-DOF thumb–index design and aims to apply adaptive stiffness or damping only when tremor is present, reducing involuntary motion while preserving natural voluntary movement.

## Goals

- Detect grasp intent and tremor-related muscle activity from surface EMG.
- Actuate a 2-DOF thumb–index mechanism.
- Apply adaptive stiffness or damping only while tremor is present.
- Preserve natural voluntary movement.

## Requirements

- Python 3.12 or newer

## How to run

Install the dependencies once (the test scripts also do this). On Windows:

```bat
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

On Linux / macOS, use `python3 -m venv .venv` and `.venv/bin/python` instead.

All modes use the **dry-run actuator**: every command is logged to
`logs/commands-<time>.csv` and nothing physical moves (see [docs/SAFETY.md](docs/SAFETY.md)).
Settings (channels and muscles, Trigno, filters, thresholds, safety) are in
[config/exoskeleton.toml](config/exoskeleton.toml).

| Mode | Command |
|---|---|
| Synthetic EMG, live monitor | `.venv\Scripts\python.exe run_exoskeleton.py --source synthetic` |
| Recorded EMG, live monitor | `.venv\Scripts\python.exe run_exoskeleton.py --source recorded --file 10_raw.csv` |
| Live Delsys Trigno, live monitor | `.venv\Scripts\python.exe run_exoskeleton.py --source trigno` |
| Trigno backend against a mock TCU (no hardware) | `.venv\Scripts\python.exe run_exoskeleton.py --source trigno --mock-tcu` |
| Headless (status lines, no window) | add `--headless --duration 40` |
| Fast offline evaluation of a recording | `.venv\Scripts\python.exe run_exoskeleton.py --source recorded --offline` |
| Offline analysis plots (original scripts) | `.venv\Scripts\python.exe main_sensor.py` |

Monitor keys: `a` arm, `e` emergency stop, `r` reset e-stop, `c` recalibrate, `q` quit.
With synthetic EMG or the mock TCU, `d` simulates a disconnection and `s` a data dropout.

The session starts **disarmed**. Calibration runs automatically from the start of the
stream:
1. Rest during the baseline window.
2. Make the reference grip during the reference window.

The monitor shows which phase you are in. Then press `a` to arm.
`--auto-arm` arms once after calibration, for demos. Re-arming after a fault is always manual.
Use `--save-calibration cal.json` and `--calibration cal.json` to reuse a calibration.

### First connection to the Delsys Trigno (DS-T03 / SP-W02 / SP-W06)

1. Install the Delsys Trigno Control Utility (TCU, from the free "Trigno SDK Server"
   installer) on a Windows PC. Connect the base station by USB and start the TCU.
2. In the TCU, pair the sensors to slots 1-4, the slots in `[[channels]]` of the config.
   Put each sensor on the muscle named there: FDS, EDC, FDI, APB. Close EMGworks and any
   other software that might be using the base station.
3. Run `.venv\Scripts\python.exe run_exoskeleton.py --source trigno`. On start-up the
   program queries the TCU and refuses to run if a slot is unpaired or the units are not
   volts. It reports the sample rate it derived. If the TCU runs on another PC, add
   `--host <ip>`.
4. Check the monitor:
   - Raw EMG should be near zero at rest and rise with contraction.
   - The sample rate should be 2000 Hz with Backwards Compatibility and Upsample on, or
     1111.111 Hz on port 50043 with Upsample off.
   - Look for warnings in the status panel.

**Not yet verified on hardware.** The Trigno client follows the Delsys SDK guide
MAN-025-3-5 and was tested only against the mock TCU in `source/mock_tcu.py`. Check on
the real system:
- **STARTINDEX base:** whether `SENSOR n STARTINDEX?` counts from 0 or 1. The guide's
  examples disagree. Contract one muscle and check the right trace moves; if the channels
  look shifted, set `trigno.start_index_base = 1`.
- **Reply format:** the exact replies (`UPSAMPLING?`, `FRAME INTERVAL?`, ...) and the banner.
- **Pause before data:** the delay between START and the first data.
- **Latency:** the real end-to-end delay.

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

```
run_exoskeleton.py         Entry point for the real-time pipeline (all modes)
main_sensor.py             Offline analysis plots of the recorded dataset
config/exoskeleton.toml    Hardware, channel/muscle mapping, filters, thresholds, safety
source/
  config.py                Loads and validates the configuration
  sensor.py                EMGSource interface, SyntheticSource, RecordedSource
  trigno.py                TrignoSource: Delsys Trigno Control Utility TCP/IP client
  mock_tcu.py              Mock TCU server for tests and demos (not hardware)
  streaming.py             Causal, stateful multi-channel filtering and envelopes
  activation.py            Calibration and normalized muscle activation
  controller.py            State machine: DISARMED, IDLE, GRASP_ARMED, STIFFENED, FAULT
  safety.py                Fault evaluation and e-stop latch
  actuator.py              Actuator interface and DryRunActuator (command log)
  runtime.py               Threads, queue, control loop, diagnostics
  monitor.py               Live matplotlib monitor and console status
  import_data.py           Dataset loading (offline)
  process_signal.py        Offline (zero-phase) analysis and filter replication
  visualize_data.py        Offline plots
tests/                     pytest suite (mock TCU, filters, controller, safety, end to end)
docs/SAFETY.md             What must exist before physical actuation
```

Data flow: EMG source → acquisition thread → bounded queue → control thread
(streaming processor → activation estimator → safety supervisor → controller) →
actuator. A watchdog thread checks the actuator watchdog and that the control thread is
alive.

## How the project works
_To be written._

# Interpreting EMG signals
_To be written._

# Control loop
_To be written._

# Actuation
_To be written._
