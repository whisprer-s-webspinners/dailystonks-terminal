@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_windows.ps1"
)
"%~dp0.venv\Scripts\python.exe" -m dailystonks.live_terminal %*
