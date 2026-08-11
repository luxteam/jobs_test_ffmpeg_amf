@echo off
setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0

if "%AMF_SRC_DIR%" == "" (set AMF_SRC_DIR=%SCRIPT_DIR%AMF\)
set AMF_SAMPLES_DIR=%AMF_SRC_DIR%amf\public\samples\

set ACTION=%~1
set TARGET=%~2

if "%ACTION%" == "build" (
    call :build_sample %TARGET%
) else if "%ACTION%" == "clean" (
    call :build_sample %ACTION%
    if %errorlevel% neq 0 exit /b %errorlevel%
    rmdir /S /Q %AMF_SRC_DIR%amf\bin
    rmdir /S /Q %AMF_SAMPLES_DIR%\bin
) else (
    echo "Unknown command: %ACTION%"
    exit /b 1
)

if %errorlevel% neq 0 exit /b 1
exit /b 0

:build_sample
    setlocal
    set SAMPLE_NAME=%~1

    echo ======^> Building AMF sample "%SAMPLE_NAME%"

    set SLN_FILE=
    for /f "tokens=*" %%F in ('dir /b /s "%AMF_SAMPLES_DIR%*.sln"') do call set SLN_FILE="%%F"

    msbuild %SLN_FILE% -target:%SAMPLE_NAME%

    if %errorlevel% neq 0 (echo ======^> Failed to build AMF sample "%SAMPLE_NAME%" & endlocal & exit /b 1)

    echo ======^> Done building AMF sample "%SAMPLE_NAME%"
    endlocal & exit /b 0