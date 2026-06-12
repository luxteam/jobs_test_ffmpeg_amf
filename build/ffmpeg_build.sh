#!/bin/sh

SCRIPT_PATH=`realpath $0`
SCRIPT_DIR=`dirname ${SCRIPT_PATH}`

if [ -z ${FFMPEG_BUILD_DIR} ]; then
    export FFMPEG_BUILD_DIR=${SCRIPT_DIR}/build_ffmpeg
fi

if [ -z ${FFMPEG_SRC_DIR} ]; then
    export FFMPEG_SRC_DIR=${SCRIPT_DIR}/FFmpeg
fi

configure_ffmpeg() {
    if [ -z ${FFMPEG_INSTALL_DIR} ]; then
        export FFMPEG_INSTALL_DIR=${SCRIPT_DIR}/ffmpeg_install
    fi

    if [ -z ${FFMPEG_BUILD_TYPE} ]; then
        export FFMPEG_BUILD_TYPE=debug
    fi

    if [ -z ${AMF_HEADERS_DIR} ]; then
        export AMF_HEADERS_DIR="${SCRIPT_DIR}/AMF/amf/public/include"
    fi

    AMF_INCLUDE_DIR="${FFMPEG_BUILD_DIR}/extrainclude"

    # Linux/GCC build args (corrected per Ubuntu_build_instructions.txt).
    # NOTE: differs from the Windows MSVC build in ffmpeg_build.bat — no quoted
    # commas in --extra-cflags, no --target-os/--toolchain=msvc, and uses
    # FFMPEG_SRC_DIR (not the undefined FFMPEG_SRC_PATH).
    FFMPEG_BUILD_ARGS="--enable-amf --enable-libdav1d --enable-libx264 --enable-libx265 --enable-libsvtav1 --enable-gpl --extra-cflags=-I${AMF_INCLUDE_DIR} --extra-cflags=-I${FFMPEG_SRC_DIR} --disable-doc --disable-ffplay --enable-ffprobe --prefix=${FFMPEG_INSTALL_DIR}"

    if [ ${FFMPEG_BUILD_TYPE} = "debug" ]; then
        FFMPEG_BUILD_ARGS="${FFMPEG_BUILD_ARGS} --enable-debug --disable-optimizations"
    fi

    echo "======> Configuring ffmpeg build\n"
    echo "Building in ${FFMPEG_BUILD_DIR}"
    echo "Building from source in ${FFMPEG_SRC_DIR}"
    echo "With configure args: ${FFMPEG_BUILD_ARGS}\n"

    mkdir -p ${FFMPEG_BUILD_DIR}
    if [ ! -d ${AMF_INCLUDE_DIR}/AMF ]; then
        mkdir -p ${AMF_INCLUDE_DIR}
        cp -r ${AMF_HEADERS_DIR} ${AMF_INCLUDE_DIR}/AMF
    fi

    (cd ${FFMPEG_BUILD_DIR} && ${FFMPEG_SRC_DIR}/configure ${FFMPEG_BUILD_ARGS})

    if [ $? -eq 0 ]; then
        echo "\n======> Configure done"
    else
        echo "\n======> Configure failed"
        exit 1
    fi
}

build_ffmpeg() {
    echo "======> Running ffmpeg build\n"

    FFMPEG_BUILD_COMMAND=""
    if [ -e ${FFMPEG_BUILD_DIR}/ffmpeg ]; then
        FFMPEG_BUILD_COMMAND="make -j`nproc`"
    else
        FFMPEG_BUILD_COMMAND="bear --append --output ../compile_commands.json -- make -j`nproc`"
    fi

    (cd ${FFMPEG_BUILD_DIR} && ${FFMPEG_BUILD_COMMAND})

    if [ $? -eq 0 ]; then
        echo "\n======> Build done"
    else
        echo "\n======> Build failed"
        exit 1
    fi
}

clean_ffmpeg() {
    echo "======> Cleaning ffmpeg\n"
    rm compile_commands.json

    (cd ${FFMPEG_BUILD_DIR} && make clean -j`nproc`)

    if [ $? -eq 0 ]; then
        echo "\n======> Clean done"
    else
        echo "\n======> Clean failed"
        exit 1
    fi
}

rebuild_ffmpeg() {
    clean_ffmpeg
    build_ffmpeg
}

case ${1} in

configure)
configure_ffmpeg
;;

build)
build_ffmpeg
;;

clean)
clean_ffmpeg
;;

rebuild)
rebuild_ffmpeg
;;

*)
echo "Unknown command: ${1}"
exit 1

esac