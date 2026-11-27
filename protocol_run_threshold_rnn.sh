#!/usr/bin/env bash
# Threshold position control + RNN (GRU) with the fixed test protocol: the same grips and the
# same simulated tremor in every run, so the runs can be compared and scored against the cues.
# Follow the cues on the screen (HVIL / GRIB / KLEM). Do NOT press k: the tremor switches on
# by itself at 35:37.5 40:42 51:54 63.5:66 68.5:72 s (3 grips and 2 rests; 3 grips stay clean).
# Close the window when it says FAERDIG (about 85 s). The run is saved as thr-gru-proto and pushed.
exec "$(dirname "$0")/run_threshold_rnn.sh" --cues protocols/greb_1.csv --inject 35:37.5 40:42 51:54 63.5:66 68.5:72 --label thr-gru-proto "$@"
