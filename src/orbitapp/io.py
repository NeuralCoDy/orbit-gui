"""Movie loading for orbitapp: TIFF stack/folder, .npy, .h5/.hdf5, or .mat
-> a single (H, W, T) array. Adapted from pyGraFT's graftapp/io.py.
Format libraries are imported lazily per-loader so importing this module
doesn't require all of them (see tests/test_io.py's importorskip use).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_movie(path: str | Path) -> np.ndarray:
    """Load a movie, returning a fully materialized, contiguous (H, W, T)
    array -- not a lazy memmap view, even though _load_tiff_stack tries
    memmap first. A memmap's actual disk read is deferred to whatever
    touches the array first; left lazy, that cost lands wherever that
    first touch happens to be (measured: 5s in MovieSliderWidget.
    show_movie's histogram, called synchronously on the GUI thread with
    no busy-bar coverage -- looks exactly like "the movie slider is slow
    to start"). Forcing it here means it's paid once, inside this
    function, which every caller already runs in a background
    FunctionWorker with a busy indicator.
    """
    path = Path(path)
    if path.is_dir():
        movie = _load_tiff_folder(path)
    else:
        suffix = path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            movie = _load_tiff_stack(path)
        elif suffix == ".npy":
            movie = np.load(path)
        elif suffix in (".h5", ".hdf5"):
            movie = _load_h5(path)
        elif suffix == ".mat":
            movie = _load_mat(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix!r}")
    return np.ascontiguousarray(movie)


def _load_tiff_stack(path: Path) -> np.ndarray:
    import tifffile

    try:
        # Memory-mapped: load_movie() materializes the result immediately
        # afterward regardless (see its docstring), so this doesn't buy a
        # faster perceived "load" -- but tifffile can read straight from
        # the mapped file into that final contiguous buffer, one copy
        # fewer than imread() (fresh buffer) + our own moveaxis/
        # ascontiguousarray (a second copy to fix the strided view).
        arr = tifffile.memmap(str(path), mode="r")
    except Exception:
        # Most real multi-page stacks aren't memmap-able (IFD metadata
        # interspersed between frame data breaks whole-series contiguity)
        # -- full load into RAM is the fallback.
        arr = tifffile.imread(path)
    if arr.ndim == 2:
        arr = arr[None, ...]
    return np.moveaxis(arr, 0, -1)  # (T, H, W) -> (H, W, T), a view either way


def _load_tiff_folder(path: Path) -> np.ndarray:
    import tifffile

    files = sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
    if not files:
        raise ValueError(f"No .tif/.tiff files found in {path}")
    return np.stack([tifffile.imread(f) for f in files], axis=-1)  # (H, W, n_files)


def _load_h5(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"No datasets found in {path}")
        return np.asarray(f[keys[0]][()])


def _load_mat(path: Path) -> np.ndarray:
    from scipy.io import loadmat

    data = loadmat(path)
    keys = [k for k in data if not k.startswith("__")]
    if not keys:
        raise ValueError(f"No variables found in {path}")
    return np.asarray(data[keys[0]])
