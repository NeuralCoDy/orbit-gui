"""Human-readable descriptions of loaded movie data, shared between the
Load tab's sidebar and the persistent header bar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def describe_source(path: Path) -> tuple[int, float]:
    """(n_files, total_size_mb) for a single movie file or a folder of TIFFs."""
    if path.is_dir():
        files = sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
        total_bytes = sum(f.stat().st_size for f in files)
        return len(files), total_bytes / 1024**2
    return 1, path.stat().st_size / 1024**2


def format_movie_summary(path: str | Path, movie: np.ndarray) -> str:
    """Multi-line, detailed summary for the Load tab sidebar."""
    path = Path(path)
    n_files, size_mb = describe_source(path)
    height, width, n_frames = movie.shape
    file_word = "file" if n_files == 1 else "files"
    return (
        f"Loaded file: {path.name}\n"
        f"Location: {path}\n"
        f"{n_files} {file_word}, {size_mb:.1f} MB\n"
        f"Frame size of {height} x {width} for {n_frames} time-steps"
    )


def format_header_summary(path: str | Path, movie: np.ndarray) -> str:
    """One-line summary for the persistent header strip."""
    path = Path(path)
    height, width, n_frames = movie.shape
    return f"{path.name}  --  {height} x {width}, {n_frames} time-steps"
