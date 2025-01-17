@echo off
echo Installing/Updating Python packages...

:: Get the directory where the BAT file is located
set "ROOT_DIR=%~dp0"
set "REQ_PATH=%ROOT_DIR%src\requirements.txt"

:: Check if pip is available
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo Error: pip is not installed
    echo Please install Python with pip from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Install requirements
python -m pip install -r "%REQ_PATH%"
if errorlevel 1 (
    echo Error installing requirements
    pause
    exit /b 1
)

echo Installation completed successfully!
pause
