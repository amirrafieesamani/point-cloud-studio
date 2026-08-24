@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install_packages
where py >nul 2>nul
if errorlevel 1 (
  echo Python 3.10, 3.11, or 3.12 is required. Install it from python.org first.
  pause
  exit /b 1
)
py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || py -3.10 -m venv .venv
if errorlevel 1 (
  echo Could not create the virtual environment. Install 64-bit Python 3.10, 3.11, or 3.12.
  pause
  exit /b 1
)
:install_packages
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Installation failed. Review the message above.
  pause
  exit /b 1
)
echo.
echo Installation complete. Run run_windows.bat.
pause
