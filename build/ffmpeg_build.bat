@echo off
setlocal EnableDelayedExpansion

set SCRIPT_DIR=%~dp0

if "%FFMPEG_BUILD_DIR%" == "" (set FFMPEG_BUILD_DIR=%SCRIPT_DIR%build_ffmpeg\)
call :make_unix_path %FFMPEG_BUILD_DIR%
set FFMPEG_UNIX_BUILD_DIR=^!UNIX_PATH^!

if "%FFMPEG_SRC_DIR%" == "" (set FFMPEG_SRC_DIR=%SCRIPT_DIR%ffmpeg\)
call :make_unix_path %FFMPEG_SRC_DIR%
set FFMPEG_UNIX_SRC_DIR=^!UNIX_PATH^!

if "%MSYS2_ROOT%" == "" (set MSYS2_ROOT=C:\msys64\)

if not defined VSCMD_ARG_TGT_ARCH (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
)
	  
set TARGET=%~1

if "%TARGET%" == "configure" (
    call :configure_ffmpeg
) else if "%TARGET%" == "build" (
    call :build_ffmpeg
) else if "%TARGET%" == "clean" (
    call :clean_ffmpeg
) else if "%TARGET%" == "rebuild" (
    call :rebuild_ffmpeg
) else (
    echo "Unknown command: %TARGET%"
    exit /b 1
)

if %errorlevel% neq 0 exit /b %errorlevel%

exit /b 0

:make_unix_path
    setlocal
    set INPUT=%~1
    set FORWARD_SLASH=/
    call set UNIX_PATH=%%INPUT:\=%FORWARD_SLASH%%%
    endlocal & set UNIX_PATH=%UNIX_PATH%
    exit /b 0

:invoke_msys2
    setlocal

    set TARGET_DIRECTORY=%~1
    set COMMAND=%~2
    set ARGS=%~3

    call %MSYS2_ROOT%msys2_shell.cmd -use-full-path -defterm -no-start -mingw64 -where "%TARGET_DIRECTORY%" -c "export PATH=/mingw64/bin:$PATH && export PKG_CONFIG_PATH=/c/deps/dav1d/lib/pkgconfig && echo Working... && %COMMAND% %ARGS%"

    endlocal & exit /b %errorlevel%

:configure_ffmpeg
    setlocal
    if "%FFMPEG_BUILD_TYPE%" == "" (set FFMPEG_BUILD_TYPE=debug)
    if "%AMF_HEADERS_DIR%" == "" (set AMF_HEADERS_DIR=%SCRIPT_DIR%AMF\amf\public\include\)

    if "%FFMPEG_INSTALL_DIR%" == "" (set FFMPEG_INSTALL_DIR=%SCRIPT_DIR%ffmpeg_install\)
    call :make_unix_path %FFMPEG_INSTALL_DIR%
    set FFMPEG_INSTALL_DIR=^!UNIX_PATH^!

    set AMF_INCLUDE_DIR=%FFMPEG_BUILD_DIR%extrainclude\
    call :make_unix_path %AMF_INCLUDE_DIR%
    set AMF_UNIX_INCLUDE_DIR=^!UNIX_PATH^!

    set FFMPEG_BUILD_ARGS="--enable-amf --enable-libdav1d --extra-cflags=-I%AMF_UNIX_INCLUDE_DIR% --extra-cflags=-I%FFMPEG_UNIX_SRC_DIR% --target-os=win64 --arch=x86_64 --toolchain=msvc --disable-doc --disable-ffplay --enable-ffprobe --prefix=%FFMPEG_INSTALL_DIR%"

    if "%FFMPEG_BUILD_TYPE%" == "debug" (set FFMPEG_BUILD_ARGS=%FFMPEG_BUILD_ARGS% --enable-debug --disable-optimizations)

    echo ======^> Configuring ffmpeg
    echo:
    echo Building in %FFMPEG_BUILD_DIR%
    echo Building from source in %FFMPEG_SRC_DIR%
    echo With configure args: %FFMPEG_BUILD_ARGS%
    echo:

    if NOT EXIST %FFMPEG_BUILD_DIR% mkdir %FFMPEG_BUILD_DIR%
    if NOT EXIST %AMF_INCLUDE_DIR% (
        mkdir %AMF_INCLUDE_DIR%\AMF
        xcopy /s /e %AMF_HEADERS_DIR% %AMF_INCLUDE_DIR%\AMF
        if %errorlevel% neq 0 exit /b 1
    )

    call :invoke_msys2 %FFMPEG_UNIX_BUILD_DIR% , %FFMPEG_UNIX_SRC_DIR%configure , %FFMPEG_BUILD_ARGS%
    if %errorlevel% neq 0 (echo ======^> ffmpeg configure failed & endlocal & exit /b 1)

    echo ======^> ffmpeg configure done
    endlocal & exit /b 0

:build_ffmpeg
    setlocal

    if NOT EXIST %FFMPEG_BUILD_DIR% (
        call :configure_ffmpeg
        if %errorlevel% neq 0 (endlocal & exit /b 1)
    )

    echo ======^> Running ffmpeg build

    call :invoke_msys2 %FFMPEG_UNIX_BUILD_DIR% , make, -j`nproc`

    if %errorlevel% neq 0 (echo ======^> ffmpeg build failed & endlocal & exit /b 1)

    echo ======^> ffmpeg build done
    endlocal & exit /b 0

:clean_ffmpeg
    setlocal

    echo ======^> Cleaning ffmpeg

    if NOT EXIST %FFMPEG_BUILD_DIR% (
        echo ======^> ffmpeg is not built, nothing to clean
        endlocal & exit /b 0
    )

    call :invoke_msys2 %FFMPEG_UNIX_BUILD_DIR% , make , "clean -j`nproc`"

    if %errorlevel% neq 0 (echo ======^> ffmpeg cleaning failed & endlocal & exit /b 1)

    echo ======^> Done cleaning ffmpeg
    endlocal & exit /b 0

:rebuild_ffmpeg
    call :clean_ffmpeg
    call :build_ffmpeg
    exit /b