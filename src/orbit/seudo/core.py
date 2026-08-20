"""SeudoData: the data holder passed between every function in this
package, ported from the `seudo` package's core.py Seudo class (itself a
minimal Python port of SEUDO's MATLAB @seudo/seudo.m class). Dropped
relative to upstream: lazy/HDF5 movie sources (matlab_io.Hdf5Movie) --
orbit-gui's movies are always fully materialized (H, W, T) arrays by the
time a stage tab sees them (see orbitapp.io.load_movie), so that
indirection isn't needed here.
"""

from __future__ import annotations

import numpy as np

from .auto_classify import auto_classify_transients as _auto_classify_transients
from .estimate import estimate_time_courses_with_seudo as _estimate_time_courses_with_seudo
from .run_on_transients import run_seudo_restricted_to_transients as _run_seudo_restricted_to_transients
from .transients import compute_transient_info as _compute_transient_info


class SeudoData:
    def __init__(
        self, movie: np.ndarray, profiles: np.ndarray, name: str = "untitled", zero_level: float = 0.0,
        time_courses: np.ndarray | None = None,
    ) -> None:
        """movie: (Y, X, F) array. profiles: (Y, X, nCells) array.
        time_courses: optional (F, nCells) array of externally-provided
        time courses, stored as tc_default['tc']."""
        movie = np.asarray(movie)
        if movie.ndim != 3:
            raise ValueError("movie must be a (Y, X, F) array")
        self.mov_y, self.mov_x, self.mov_f = movie.shape

        profiles = np.asarray(profiles)
        if profiles.shape[:2] != (self.mov_y, self.mov_x):
            raise ValueError(
                f"size of profiles {profiles.shape[:2]} does not match size of movie ({self.mov_y}, {self.mov_x})"
            )

        self.movie = movie
        self.profiles = profiles
        self.n_cells = profiles.shape[2]
        self.zero_level = zero_level
        self.name = name

        self.tc_default: dict | None = None
        self.tc_contam: dict | None = None
        self.tc_seudo: list[dict] = []

        if time_courses is not None:
            time_courses = np.asarray(time_courses)
            if time_courses.shape != (self.mov_f, self.n_cells):
                raise ValueError(
                    f"time_courses shape {time_courses.shape} does not match (mov_f={self.mov_f}, n_cells={self.n_cells})"
                )
            self.tc_default = {"tc": time_courses}

    def get_frame(self, frame_index: int) -> np.ndarray:
        return self.movie[:, :, frame_index].astype(float) - self.zero_level

    def _resolve_tc_struct(self, which_struct) -> dict:
        """which_struct: 'default', 'contam', 'seudo', or ('seudo', index)
        with a (possibly negative) index into tc_seudo. Returns the actual
        dict (mutable, so callers can fill in fields like 'transient_info'
        in place)."""
        if isinstance(which_struct, str):
            key = which_struct.lower()
        elif isinstance(which_struct, (tuple, list)) and len(which_struct) == 2:
            key = which_struct[0].lower()
        else:
            raise ValueError(f"time course struct specification not recognized: {which_struct!r}")

        if key in ("default", "tcdefault"):
            if self.tc_default is None:
                raise ValueError("tc_default has not been set (pass time_courses to the constructor)")
            return self.tc_default
        if key in ("contam", "tccontam"):
            if self.tc_contam is None:
                raise ValueError("tc_contam has not been set")
            return self.tc_contam
        if key in ("seudo", "tcseudo"):
            if not self.tc_seudo:
                raise ValueError("tc_seudo is empty")
            index = which_struct[1] if isinstance(which_struct, (tuple, list)) else -1
            return self.tc_seudo[index]

        raise ValueError(f"time course struct specification not recognized: {which_struct!r}")

    def estimate_time_courses_with_seudo(self, **kwargs) -> dict:
        result = _estimate_time_courses_with_seudo(self.movie, self.profiles, zero_level=self.zero_level, **kwargs)
        self.tc_seudo.append(result)
        return result

    def compute_transient_info(self, which_struct: str = "default", **kwargs) -> list[dict]:
        return _compute_transient_info(self, which_struct, **kwargs)

    def auto_classify_transients(self, which_struct: str = "default", **kwargs) -> list[dict]:
        return _auto_classify_transients(self, which_struct, **kwargs)

    def run_seudo_restricted_to_transients(self, which_struct: str = "default", **kwargs) -> dict:
        return _run_seudo_restricted_to_transients(self, which_struct, **kwargs)
