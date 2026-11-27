#!/usr/bin/env bash
# 4. Velocity control (the grasp sets the closing speed, with a hold band) + RNN (GRU, models/gru.pt): live Trigno EMG drives the tendon motor.
# Press k in the window to lay simulated tremor over the live EMG, k again to stop it.
# The run is saved in measurements/ (label vel-gru): EMG, motor (motor.csv) and tremor times
# (injections.csv), and is committed and pushed to GitHub at the end (--push).
# Without a valid models/gru.pt the GRU falls back to the DSP and says so.
# Extra options are passed on, e.g.:  ./run_velocity_rnn.sh --motor-port /dev/tty.usbserial-XXXX
cd "$(dirname "$0")"
exec .venv/bin/python run_emg.py --source trigno --motor --control velocity --method gru \
    --calibration measurements/plac_calibration.json --out measurements --label vel-gru --push "$@"
