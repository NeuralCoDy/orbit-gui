"""Builds orbit's two optional C++ extensions. This is additive to
pyproject.toml's [project] metadata -- setuptools reads both.

Both extensions are marked optional: if a compiler (or, for the second
one, FFTW3) isn't available, setuptools skips that extension and warns
instead of aborting `pip install`. Everything that can use them falls
back to an equivalent pure-Python/numpy implementation transparently
either way -- see orbit._native's and orbit.seudo._native's own
docstrings.
"""

import os
import sys

from setuptools import Extension, setup

if sys.platform == "win32":
    # MSVC's flag syntax differs from GCC/Clang's ("/" not "-", /std:c++14
    # not -std=c++14) -- std::thread needs no extra link flag on Windows
    # (unlike -pthread on Unix), the runtime provides it directly.
    EXTRA_COMPILE_ARGS = ["/O2", "/std:c++14"]
    EXTRA_LINK_ARGS: list[str] = []
else:
    EXTRA_COMPILE_ARGS = ["-O3", "-std=c++14", "-pthread"]
    EXTRA_LINK_ARGS = ["-pthread"]


def _native_extension() -> list[Extension]:
    """orbit._native._orbit_native: local correlation, half-sample mode,
    OASIS deconvolution, the Ljung-Box Q statistic -- pybind11 + the
    standard library only (std::thread for parallelism), no system
    dependency beyond a C++ compiler."""
    try:
        import pybind11
    except ImportError:
        return []

    return [
        Extension(
            "orbit._native._orbit_native",
            sources=["src/orbit/_native/orbit_native.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=EXTRA_COMPILE_ARGS,
            extra_link_args=EXTRA_LINK_ARGS,
            optional=True,
        )
    ]


# (include_dir, lib_dir) pairs worth trying even with no env override --
# covers the common package-manager default prefixes. Harmless if a pair
# doesn't exist: an -I/-L flag pointing at a missing directory is a
# silent no-op for gcc/clang/MSVC, so listing all of these unconditionally
# is safe. This matters most on macOS: cibuildwheel cross-compiles an
# x86_64 wheel from an arm64 (macos-14) runner, and Homebrew installs
# each architecture's own FFTW3 to a DIFFERENT prefix (/opt/homebrew for
# native arm64, /usr/local for an x86_64 build under Rosetta) -- a single
# fixed path can't cover both, but trying both candidates and letting the
# linker pick whichever actually has the right-architecture library does.
_FFTW_CANDIDATE_DIRS = [
    ("/opt/homebrew/include", "/opt/homebrew/lib"),  # Homebrew, Apple Silicon
    ("/usr/local/include", "/usr/local/lib"),  # Homebrew (Intel/Rosetta) / common Linux local prefix
    ("/usr/include", "/usr/lib/x86_64-linux-gnu"),  # Debian/Ubuntu system package
    ("/usr/include", "/usr/lib64"),  # RHEL/AlmaLinux/manylinux system package
]


def _seudo_native_extension() -> list[Extension]:
    """orbit.seudo._native._fista_native: Real-SEUDO's per-cell FISTA
    solve, compiled -- needs FFTW3 in addition to a compiler, unlike the
    extension above. Tries FFTW_INCLUDE_DIR/FFTW_LIB_DIR from the
    environment first (same override src/orbit/seudo/_native/
    build_native.sh already supports for a manual build -- set instead
    by .github/workflows/wheels.yml on Windows, where there's no
    standard install prefix to guess), then falls back to
    _FFTW_CANDIDATE_DIRS' common package-manager prefixes."""
    try:
        import pybind11
    except ImportError:
        return []

    include_dirs = [pybind11.get_include()]
    library_dirs = []

    fftw_include_dir = os.environ.get("FFTW_INCLUDE_DIR")
    fftw_lib_dir = os.environ.get("FFTW_LIB_DIR")
    if fftw_include_dir or fftw_lib_dir:
        if fftw_include_dir:
            include_dirs.append(fftw_include_dir)
        if fftw_lib_dir:
            library_dirs.append(fftw_lib_dir)
    else:
        for inc, lib in _FFTW_CANDIDATE_DIRS:
            if os.path.isdir(inc):
                include_dirs.append(inc)
            if os.path.isdir(lib):
                library_dirs.append(lib)

    return [
        Extension(
            "orbit.seudo._native._fista_native",
            sources=["src/orbit/seudo/_native/fista_native.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=["fftw3"],
            language="c++",
            extra_compile_args=EXTRA_COMPILE_ARGS,
            extra_link_args=EXTRA_LINK_ARGS,
            optional=True,
        )
    ]


setup(ext_modules=_native_extension() + _seudo_native_extension())
