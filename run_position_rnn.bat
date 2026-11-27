@echo off
rem 3. Threshold control (open or closed) + RNN (GRU, models\gru.pt): live Trigno EMG drives the tendon motor.
rem Press k in the window to lay simulated tremor over the live EMG, k again to stop it.
rem The run is saved in measurements\ (label thr-gru): EMG, motor (motor.csv) and tremor times
rem (injections.csv), and is committed and pushed to GitHub at the end (--push).
rem Without a valid models\gru.pt the GRU falls back to the DSP and says so.
rem Extra options are passed on, e.g.:  run_position_rnn.bat --motor-port COM4   or   --calibrate
setlocal
cd /d "%~dp0"
".venv\Scripts\python.exe" run_emg.py --source trigno --motor --control threshold --method gru --calibration measurements\plac_calibration.json --out measurements --label thr-gru --push %*
pause
