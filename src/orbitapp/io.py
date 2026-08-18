"""Movie loading for orbitapp: TIFF stack/folder, .npy, .h5/.hdf5, or .mat
-> a single (H, W, T) array. Adapted from pyGraFT's graftapp/io.py.
Format libraries are imported lazily per-loader so importing this module
doesn't require all of them (see tests/test_io.py's importorskip use).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_movie(path: str | Path) -> np.ndarray:
    """Load a movie, returning an (H, W, T) array."""
    path = Path(path)
    if path.is_dir():
        return _load_tiff_folder(path)

    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        return _load_tiff_stack(path)
    if suffix == ".npy":
        return np.load(path)
    if suffix in (".h5", ".hdf5"):
        return _load_h5(path)
    if suffix == ".mat":
        return _load_mat(path)
    raise ValueError(f"Unsupported file type: {suffix!r}")


def _load_tiff_stack(path: Path) -> np.ndarray:
    import tifffile

    try:
        # Memory-mapped: near-instant "load" for a large, memmap-able
        # stack, since pixel data is only read from disk on actual access
        # (e.g. by a projection or the movie player) rather than upfront.
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
