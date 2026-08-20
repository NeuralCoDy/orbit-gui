"""Run SEUDO for one or more cells, each restricted to just the frames
spanned by its own detected transients -- ported from the `seudo`
package's run_seudo_on_transients.py (itself a port of the core idea in
SEUDO's parallelSEUDO.m; cells run sequentially here rather than via
parfor, a wall-clock simplification only -- each cell's computation is
already independent of the others).
"""

from __future__ import annotations

import numpy as np

from .estimate import estimate_time_courses_with_seudo


def frame_blocks_for_cell(transient_info: list[dict], cell_idx: int) -> list[tuple[int, int]]:
    """(start, end) 0-indexed inclusive frame ranges spanning cell_idx's
    detected transients (already t_pre/t_post padded by compute_transient_info)."""
    times = transient_info[cell_idx]["times"]
    return [(int(s), int(e)) for s, e in times]


def run_seudo_restricted_to_transients(
    se,
    which_struct: str = "default",
    which_cells=None,
    progress_callback=None,
    frame_progress_callback=None,
    **seudo_params,
) -> dict:
    """Run estimate_time_courses_with_seudo for each of which_cells
    (default: all), restricted to that cell's own detected-transient
    frames. progress_callback(done, total, cell_idx) fires after each
    cell; frame_progress_callback(col, cell_idx, frames_done,
    total_frames) fires as each frame within the current cell finishes.

    Returns a combined result dict (tc/tc_lsq of shape (mov_f,
    len(which_cells))) and appends it to se.tc_seudo."""
    tc_struct = se._resolve_tc_struct(which_struct)
    transient_info = tc_struct["transient_info"]
    which_cells = list(range(se.n_cells)) if which_cells is None else list(which_cells)

    tc = np.full((se.mov_f, len(which_cells)), np.nan)
    tc_lsq = np.full((se.mov_f, len(which_cells)), np.nan)
    per_cell_results = {}

    for col, cell_idx in enumerate(which_cells):
        blocks = frame_blocks_for_cell(transient_info, cell_idx)
        if blocks:
            result = estimate_time_courses_with_seudo(
                se.movie, se.profiles, which_cells=[cell_idx], zero_level=se.zero_level, frame_blocks=blocks,
                progress_callback=(
                    None if frame_progress_callback is None
                    else lambda done, total, col=col, cell_idx=cell_idx: frame_progress_callback(
                        col, cell_idx, done, total
                    )
                ),
                **seudo_params,
            )
            tc[:, col] = result["tc"][:, 0]
            tc_lsq[:, col] = result["tc_lsq"][:, 0]
            per_cell_results[cell_idx] = result

        if progress_callback is not None:
            progress_callback(col + 1, len(which_cells), cell_idx)

    combined = dict(
        tc=tc, tc_lsq=tc_lsq, params=dict(which_cells=which_cells, which_struct=which_struct, **seudo_params),
        extras=dict(per_cell_results=per_cell_results),
    )
    se.tc_seudo.append(combined)
    return combined
