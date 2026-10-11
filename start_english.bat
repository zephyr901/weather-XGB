@echo off
setlocal
cd /d "%~dp0"
if exist "release\WeatherXGB-English\WeatherXGB-English.exe" (
    start "" "release\WeatherXGB-English\WeatherXGB-English.exe" %*
    exit /b
)
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m weather_xgb %*
    exit /b
)
where pythonw >nul 2>nul
if errorlevel 1 (
    echo Install Python 3.13 and the requirements first. See README.md.
    pause
    exit /b 1
)
start "" pythonw -m weather_xgb %*
