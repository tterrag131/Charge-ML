@echo off
echo Starting First-Time Setup
echo =======================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    echo Please install Python from https://www.python.org/downloads/
    echo Be sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

:: Install required packages
echo Installing required packages...
python -m pip install --upgrade pip
python -m pip install -r src\requirements.txt

echo.
echo Setup completed!
echo You can now use run_prediction.bat to run the model.
pause
