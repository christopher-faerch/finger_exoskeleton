@echo off
rem Run linters, type checks and tests (Windows).
rem Usage: test.bat            (set PYTHON=... to pick an interpreter, default "py -3")
setlocal
cd /d "%~dp0"

if not defined PYTHON set "PYTHON=py -3"
set "VENV=.venv"

if not exist "%VENV%\Scripts\python.exe" (
    echo ==^> Creating virtualenv in %VENV%
    %PYTHON% -m venv "%VENV%" || exit /b 1
)
set "PY=%VENV%\Scripts\python.exe"

echo ==^> Installing dev dependencies
"%PY%" -m pip install --quiet --upgrade pip
"%PY%" -m pip install --quiet -r requirements-dev.txt || exit /b 1

set STATUS=0

echo ==^> flake8
"%PY%" -m flake8 . || set STATUS=1
echo ==^> pylint
"%PY%" -m pylint main_sensor.py source tests || set STATUS=1
"%PY%" -m pylint run_exoskeleton.py || set STATUS=1
echo ==^> mypy
"%PY%" -m mypy || set STATUS=1
echo ==^> pytest
"%PY%" -m pytest || set STATUS=1

if "%STATUS%"=="0" (
    echo ==^> All checks passed
) else (
    echo ==^> Some checks FAILED
)
exit /b %STATUS%
