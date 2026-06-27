@echo off
:: Underfit installer for Windows (Pure CMD)
::
:: Usage:
::     install.bat                  # full flow: install uv (if missing) + uv sync + underfit-setup
::     install.bat --no-setup       # stop after `uv sync`, skip the underfit-setup wizard
::     install.bat --backend sat    # opt into stable-audio-tools (default is sa3)
::
:: Idempotent: re-running upgrades anything missing and leaves the rest alone.

setlocal enabledelayedexpansion

set "UNDERFIT_DIR=%~dp0"
cd /d "%UNDERFIT_DIR%"

set "SKIP_SETUP=0"
set "BACKEND="

:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--no-setup" (
    set "SKIP_SETUP=1"
    shift
    goto parse_args
)
if "%~1"=="--backend" (
    if "%~2"=="" (
        echo ✗ --backend requires a value ^(sa3 ^| sat^)
        exit /b 1
    )
    set "BACKEND=%~2"
    shift
    shift
    goto parse_args
)
:: Handle --backend=value format
set "arg=%~1"
if "%arg:~0,10%"=="--backend=" (
    set "BACKEND=%arg:~10%"
    shift
    goto parse_args
)
if "%~1"=="-h" goto show_help
if "%~1"=="--help" goto show_help

echo unknown flag: %~1
echo use --help for usage
exit /b 1

:show_help
echo Usage:
echo     install.bat                  # full flow
echo     install.bat --no-setup       # skip setup wizard
echo     install.bat --backend sat    # specify backend
exit /b 0

:end_parse

:: ── 1. uv ──────────────────────────────────────────────────────────────────
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo ▸ uv not found, installing via native curl...
    
    :: Use native Windows curl to fetch the standalone Windows installer
    curl -L -s -o "%TEMP%\uv-installer.exe" https://astral.sh/uv/install.exe
    if %errorlevel% neq 0 (
        echo ✗ Failed to download uv using curl. Please ensure you are connected to the internet.
        exit /b 1
    )
    
    :: Run the standalone installer silently
    echo ▸ Running uv installer...
    "%TEMP%\uv-installer.exe" /quiet
    del "%TEMP%\uv-installer.exe" >nul 2>nul
    
    :: Add typical Astral Windows path to current session PATH
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    
    where uv >nul 2>nul
    if %errorlevel% neq 0 (
        echo ✗ uv installed, but not on PATH. Open a new command prompt and re-run.
        exit /b 1
    )
)

:: Get uv version
for /f "tokens=2" %%i in ('uv --version') do set "UV_VER=%%i"
echo ▸ uv %UV_VER% ready

:: ── 2. deps ────────────────────────────────────────────────────────────────
echo ▸ syncing dependencies ^(uv sync --inexact^) ...
uv sync --inexact
if %errorlevel% neq 0 (
    echo ✗ uv sync failed.
    exit /b 1
)

:: ── 3. wizard ──────────────────────────────────────────────────────────────
if "%SKIP_SETUP%"=="1" (
    echo ▸ skipping underfit-setup ^(--no-setup passed^)
    echo ▸ done — now run run.bat to start the dashboard.
    exit /b 0
)

echo ▸ launching underfit-setup ...
if not "%BACKEND%"=="" (
    uv run python -m underfit.cli.setup --backend "%BACKEND%"
) else (
    uv run python -m underfit.cli.setup
)

echo.
echo ▸ all done — now run run.bat to start the dashboard.
endlocal