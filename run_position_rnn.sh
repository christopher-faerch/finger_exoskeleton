#!/usr/bin/env bash
# 2. Position control (the finger follows the grasp, with hysteresis) + RNN (GRU, models/gru.pt): live Trigno EMG drives the tendon motor.
# Press k in the window to lay simulated tremor over the live EMG, k again to stop it.
# The run is saved in measurements/ (label pos-gru): EMG, motor (motor.csv) and tremor times
# (injections.csv), and is committed and pushed to GitHub at the end (--push).
# Without a valid models/gru.pt the GRU falls back to the DSP and says so.
# Extra options are passed on, e.g.:  ./run_position_rnn.sh --motor-port /dev/tty.usbserial-XXXX
cd "$(dirname "$0")"
exec .venv/bin/python run_emg.py --source trigno --motor --control position --method gru \
    --calibration measurements/plac_calibration.json --out measurements --label pos-gru --push "$@"
