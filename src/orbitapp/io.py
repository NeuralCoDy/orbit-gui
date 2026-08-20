"""Movie loading for orbitapp: TIFF stack/folder, .npy, .h5/.hdf5, or .mat
-> a single (H, W, T) array. Adapted from pyGraFT's graftapp/io.py.
Format libraries are imported lazily per-loader so importing this module
doesn't require all of them (see tests/test_io.py's importorskip use).
"""

from __future__ import annotations

import tempfile
import urllib.request
from pathlib import Path

import numpy as np

# Dropbox share links serve an HTML preview page unless dl=1 forces the raw
# file -- swapping that one query param is enough, no API/auth needed.
_DEFAULT_DATASET_URL = (
    "https://www.dropbox.com/scl/fi/agql0ng6zbilpmf5dk4cw/file_00001.tif"
    "?rlkey=67fjgkves4y9rj2msd3cvg40d&st=t6p7taa4&dl=1"
)
DEFAULT_DATASET_PATH = Path(tempfile.gettempdir()) / "orbit_gui_default_dataset.tif"


def load_default_dataset() -> np.ndarray:
    """Loads the bundled example movie, downloading it from Dropbox into a
    local temp-dir cache the first time it's needed -- later calls (even
    from a future app launch) reuse that same cached file instead of
    re-downloading."""
    if not DEFAULT_DATASET_PATH.exists():
        urllib.request.urlretrieve(_DEFAULT_DATASET_URL, DEFAULT_DATASET_PATH)
    return load_movie(DEFAULT_DATASET_PATH)


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


def _pick_largest_array(candidates: dict) -> np.ndarray:
    """Picks the largest array among several candidate variables in a
    loaded .h5/.mat file. In practice the movie is overwhelmingly bigger
    than any metadata stored alongside it (framerate, notes, ROI info,
    ...) -- a far more reliable signal than "whichever key happens to
    sort first", which an unrelated small variable could easily win."""
    arrays = {key: np.asarray(value) for key, value in candidates.items()}
    return arrays[max(arrays, key=lambda key: arrays[key].size)]


def _load_h5(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        candidates = {k: f[k][()] for k in f.keys() if isinstance(f[k], h5py.Dataset)}
        if not candidates:
            raise ValueError(f"No datasets found in {path}")
        return _pick_largest_array(candidates)


def _load_mat(path: Path) -> np.ndarray:
    from scipy.io import loadmat

    try:
        data = loadmat(path)
    except NotImplementedError:
        # MATLAB v7.3+ .mat files are HDF5 under the hood -- scipy's
        # legacy reader can't open them (raises exactly this, saying so).
        # h5py can open the file, but two MATLAB-specific quirks need
        # handling that a plain .h5 file wouldn't have (confirmed against
        # real MATLAB-compatible v7.3 output): every array comes back
        # axis-reversed relative to its MATLAB size() (HDF5 is row-major,
        # MATLAB column-major -- .T undoes it), and cell arrays/strings
        # spill into a "#refs#" bookkeeping group that must be skipped,
        # not mistaken for the movie.
        import h5py

        with h5py.File(path, "r") as f:
            candidates = {
                k: f[k][()] for k in f.keys() if not k.startswith("#") and isinstance(f[k], h5py.Dataset)
            }
            if not candidates:
                raise ValueError(f"No variables found in {path}")
            return _pick_largest_array(candidates).T

    candidates = {k: v for k, v in data.items() if not k.startswith("__")}
    if not candidates:
        raise ValueError(f"No variables found in {path}")
    return _pick_largest_array(candidates)
