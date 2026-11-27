@echo off
rem 1. Position control + traditional signal processing (DSP): live Trigno EMG drives the tendon motor.
rem Press k in the window to lay simulated tremor over the live EMG, k again to stop it.
rem The run is saved in measurements\ (label pos-dsp) with the tremor times in injections.csv.
rem Extra options are passed on, e.g.:  run_position_dsp.bat --motor-port COM4   or   --calibrate
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" run_emg.py --source trigno --motor --control position --method dsp --calibration measurements\plac_calibration.json --out measurements --label pos-dsp %*
pause
