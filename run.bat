@echo off
setlocal

set "UNDERFIT_DIR=%~dp0"
cd /d "%UNDERFIT_DIR%"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No .venv found - run install.bat first.
    exit /b 1
)

echo.
echo ==============================
echo Python Environment
echo ==============================

.venv\Scripts\python.exe --version

echo.
echo Checking PyTorch...

.venv\Scripts\python.exe -c "import torch;print('PyTorch:',torch.__version__);print('CUDA Build:',torch.version.cuda);print('CUDA Available:',torch.cuda.is_available());print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

if errorlevel 1 (
    echo [ERROR] PyTorch not installed.
    pause
    exit /b 1
)

echo.
echo Starting Dashboard...
echo.

REM Force Python to write logs to disk instantly on Windows
set PYTHONUNBUFFERED=1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

.venv\Scripts\python.exe dashboard\server.py %*

endlocal