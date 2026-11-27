#!/usr/bin/env bash
# 2. Velocity control + traditional signal processing (DSP): live Trigno EMG drives the tendon motor.
# Press k in the window to lay simulated tremor over the live EMG, k again to stop it.
# The run is saved in measurements/ (label vel-dsp) with the tremor times in injections.csv.
# Extra options are passed on, e.g.:  ./run_velocity_dsp.sh --motor-port /dev/tty.usbserial-XXXX
cd "$(dirname "$0")"
exec .venv/bin/python run_emg.py --source trigno --motor --control velocity --method dsp \
    --calibration measurements/plac_calibration.json --out measurements --label vel-dsp "$@"
