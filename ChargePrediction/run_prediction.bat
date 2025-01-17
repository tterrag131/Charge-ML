@echo off
setlocal EnableDelayedExpansion

echo Checking Python environment...

:: Get the directory where the BAT file is located
set "ROOT_DIR=%~dp0"
set "SCRIPT_PATH=%ROOT_DIR%src\SafeML2.py"
set "REQ_PATH=%ROOT_DIR%src\requirements.txt"

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check if required packages are installed
python -c "import pandas" >nul 2>&1
if errorlevel 1 (
    echo Required packages not found. Installing now...
    python -m pip install -r "%REQ_PATH%"
    if errorlevel 1 (
        echo Error installing requirements
        pause
        exit /b 1
    )
)

echo Running Charge Prediction Model...

:: Check if Python script exists
if not exist "!SCRIPT_PATH!" (
    echo Error: Cannot find SafeML2.py
    echo Expected location: !SCRIPT_PATH!
    echo Please ensure the folder structure is maintained.
    pause
    exit /b 1
)

:: Run the script
cd "!ROOT_DIR!"
python "!SCRIPT_PATH!"

echo.
echo Process completed.
pause
