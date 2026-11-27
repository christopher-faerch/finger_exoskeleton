@echo off
rem 1. Position control (the finger follows the grasp, with hysteresis) + traditional signal processing (DSP): live Trigno EMG drives the tendon motor.
rem Press k in the window to lay simulated tremor over the live EMG, k again to stop it.
rem The run is saved in measurements\ (label pos-dsp): EMG, motor (motor.csv) and tremor times
rem (injections.csv), and is committed and pushed to GitHub at the end (--push).
rem Extra options are passed on, e.g.:  run_position_dsp.bat --motor-port COM4   or   --calibrate
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" run_emg.py --source trigno --motor --control position --method dsp --calibration measurements\plac_calibration.json --out measurements --label pos-dsp --push %*
pause
