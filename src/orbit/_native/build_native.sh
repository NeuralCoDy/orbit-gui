#!/usr/bin/env bash
# Builds the optional native (C++) accelerator extension for orbit.
# Not required for orbit to work -- local_correlation_projection and
# mode_projection fall back to their pure-Python/numpy implementations
# if this hasn't been built.
#
# Requires: a C++14 compiler (g++/clang++) and `pip install pybind11`.
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

echo "building _orbit_native${PY_EXT_SUFFIX} ..."
"$CXX" -std=c++14 -O3 -fPIC -pthread -shared $PY_INCLUDES \
    orbit_native.cpp -o "_orbit_native${PY_EXT_SUFFIX}"
echo "done: $(pwd)/_orbit_native${PY_EXT_SUFFIX}"
