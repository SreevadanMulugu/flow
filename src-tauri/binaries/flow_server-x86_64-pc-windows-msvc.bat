@echo off
:: Flow server sidecar for Windows
:: Resolves engine\ relative to this script in src-tauri\binaries\
set SCRIPT_DIR=%~dp0
set ENGINE_DIR=%SCRIPT_DIR%..\..\engine

:: Prefer ~/.flow/nemo_env Python (installed by setup_windows.ps1), else system python
set VENV_PY=%USERPROFILE%\.flow\nemo_env\Scripts\python.exe
if exist "%VENV_PY%" (
    set PYTHON=%VENV_PY%
) else (
    set PYTHON=python
)

cd /d "%ENGINE_DIR%"
"%PYTHON%" flow_server.py %*
