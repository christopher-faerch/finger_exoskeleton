@echo off
rem 3. Velocity control (the grasp sets the closing speed, with a hold band) + traditional signal processing (DSP): live Trigno EMG drives the tendon motor.
rem Press k in the window to lay simulated tremor over the live EMG, k again to stop it.
rem The run is saved in measurements\ (label vel-dsp): EMG, motor (motor.csv) and tremor times
rem (injections.csv), and is committed and pushed to GitHub at the end (--push).
rem Extra options are passed on, e.g.:  run_velocity_dsp.bat --motor-port COM4   or   --calibrate
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" run_emg.py --source trigno --motor --control velocity --method dsp --calibration measurements\plac_calibration.json --out measurements --label vel-dsp --push %*
pause
