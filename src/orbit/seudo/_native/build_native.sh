#!/usr/bin/env bash
# Builds the optional native (C++/FFTW) FISTA accelerator extension.
# Not required for Real-SEUDO to work -- estimate.py falls back to the
# pure-Python solver (fista_nonneg_weighted_l1) if this hasn't been built.
#
# Requires: a C++14 compiler (g++/clang++), pybind11, and FFTW3.
#   Debian/Ubuntu: apt-get install libfftw3-dev && pip install pybind11
#   macOS (Homebrew): brew install fftw && pip install pybind11
#
# If FFTW's header/library aren't on the compiler's default search path
# (e.g. no libfftw3-dev package and no sudo to install one, but the
# runtime libfftw3.so IS present -- a real situation encountered building
# this the first time), override the paths explicitly:
#   FFTW_INCLUDE_DIR=/path/to/fftw3.h's/dir FFTW_LINK_FLAGS="-l:libfftw3.so.3" bash build_native.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CXX="${CXX:-g++}"
if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "error: no C++ compiler found ($CXX). Install one (e.g. 'sudo apt-get install build-essential') and retry." >&2
    exit 1
fi

if ! python3 -c "import pybind11" >/dev/null 2>&1; then
    echo "error: pybind11 not installed. Run: pip install pybind11" >&2
    exit 1
fi

PY_INCLUDES=$(python3 -m pybind11 --includes)
PY_EXT_SUFFIX=$(python3-config --extension-suffix)

FFTW_INCLUDE_FLAGS=""
if [ -n "${FFTW_INCLUDE_DIR:-}" ]; then
    FFTW_INCLUDE_FLAGS="-I${FFTW_INCLUDE_DIR}"
fi
FFTW_LINK_FLAGS="${FFTW_LINK_FLAGS:--lfftw3}"

echo "building _fista_native${PY_EXT_SUFFIX} ..."
"$CXX" -std=c++14 -O3 -fPIC -shared $PY_INCLUDES $FFTW_INCLUDE_FLAGS \
    fista_native.cpp -o "_fista_native${PY_EXT_SUFFIX}" $FFTW_LINK_FLAGS
echo "done: $(pwd)/_fista_native${PY_EXT_SUFFIX}"
