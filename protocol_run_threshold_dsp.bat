@echo off
rem Threshold position control + DSP with the fixed test protocol: the same grips and the
rem same simulated tremor in every run, so the runs can be compared and scored against the cues.
rem Follow the cues on the screen (HVIL / GRIB / KLEM). Do NOT press k: the tremor switches on
rem by itself at 35:37.5 40:42 51:54 63.5:66 68.5:72 s (3 grips and 2 rests; 3 grips stay clean).
rem Close the window when it says FAERDIG (about 85 s). The run is saved as thr-dsp-proto and pushed.
rem Extra options are passed on, e.g.:  protocol_run_threshold_dsp.bat --motor-port COM4
call "%~dp0run_threshold_dsp.bat" --cues protocols\greb_1.csv --inject 35:37.5 40:42 51:54 63.5:66 68.5:72 --label thr-dsp-proto %*
