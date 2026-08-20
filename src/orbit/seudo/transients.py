"""Transient detection and per-transient spatial profile computation,
ported from the `seudo` package's transients.py (itself a port of
seudo.identifyTransients and se.computeTransientInfo).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import ndimage

from .geometry import compute_roi_coms
from .stats import correlation_vector_matrix, robust_std


def default_thresh_fcn(x: np.ndarray) -> float:
    return max(robust_std(x) * 5, np.percentile(x, 95))


def identify_transients(
    time_courses: np.ndarray, thresh_fcn=None, blur_radius: int = 2, min_duration: int = 6
) -> np.ndarray:
    """Identify candidate transients in each column (cell) of time_courses
    (F, C). thresh_fcn: scalar threshold per cell's time course; points
    above it are candidate transient frames. blur_radius: candidate frames
    are dilated by this many frames each direction before measuring
    duration. min_duration: components shorter than this are discarded.
    Returns a boolean (F, C) array."""
    if thresh_fcn is None:
        thresh_fcn = default_thresh_fcn

    time_courses = np.asarray(time_courses, dtype=float)
    n_frames, n_cells = time_courses.shape
    transient_frames = np.zeros((n_frames, n_cells), dtype=bool)
    structure = np.ones(blur_radius * 2 + 1, dtype=bool)

    for cc in range(n_cells):
        tc = time_courses[:, cc]
        tc_th = tc > thresh_fcn(tc)
        tc_th = ndimage.binary_dilation(tc_th, structure=structure)

        labeled, n_labels = ndimage.label(tc_th)
        if n_labels == 0:
            continue

        durations = np.bincount(labeled, minlength=n_labels + 1)[1:]
        too_short = np.flatnonzero(durations < min_duration) + 1
        if too_short.size:
            labeled[np.isin(labeled, too_short)] = 0

        transient_frames[:, cc] = labeled > 0

    return transient_frames


def _compute_one_active_shape(tc: np.ndarray, mov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mov: (Y, X, T) mini-movie. tc: length-T time course over that span."""
    w_y, w_x, n_t = mov.shape
    mov_reshape = mov.reshape(w_y * w_x, n_t)

    if np.all(np.isnan(tc)):
        nan_img = np.full((w_y, w_x), np.nan)
        return nan_img, nan_img.copy(), nan_img.copy()

    safe = ~np.isnan(tc)
    tc_safe = tc[safe]
    denom = np.dot(tc_safe, tc_safe)
    fit = np.full(w_y * w_x, np.nan) if denom == 0 else (mov_reshape[:, safe] @ tc_safe) / denom
    shape = fit.reshape(w_y, w_x)

    shape_corr = correlation_vector_matrix(tc, mov_reshape.T).reshape(w_y, w_x)
    mean_image = np.mean(mov, axis=2)
    return shape, shape_corr, mean_image


def compute_transient_info(
    se,
    which_struct: str = "default",
    transient_frames: np.ndarray | None = None,
    thresh_fcn=None,
    blur_radius: int = 2,
    win_radius: int = 10,
    use_com: bool = False,
    min_duration: int = 6,
    t_pre: int = 5,
    t_post: int = 10,
) -> list[dict]:
    """Identify transients and compute per-transient spatial shape info for
    a SeudoData object's chosen time-course struct (se._resolve_tc_struct).
    Stores the result into that struct's 'transient_info' key and returns
    it. If the struct already has a 'transient_info' with a classification,
    that classification is preserved for transients whose time span didn't
    change (resolved/unclassified when transients were split/merged)."""
    tc_struct = se._resolve_tc_struct(which_struct)
    profiles = se.profiles
    time_courses = np.asarray(tc_struct["tc"], dtype=float)
    n_cells = profiles.shape[2]

    if transient_frames is None:
        transient_frames = identify_transients(
            time_courses, thresh_fcn=thresh_fcn, blur_radius=blur_radius, min_duration=min_duration
        )

    params = dict(
        thresh_fcn=thresh_fcn, blur_radius=blur_radius, win_radius=win_radius, use_com=use_com,
        min_duration=min_duration, t_pre=t_pre, t_post=t_post,
    )

    # ---- per-cell: segment into transients (times) and compute spatial window ----

    trans_records: list[tuple[int, int, int]] = []  # (cell_idx, start, stop), 0-indexed inclusive
    cell_trans_indices: list[list[int]] = [[] for _ in range(n_cells)]
    roi_windows = np.zeros((n_cells, 4), dtype=int)

    for cc in range(n_cells):
        labeled, n_trans = ndimage.label(transient_frames[:, cc])
        for tt in range(1, n_trans + 1):
            idx = np.flatnonzero(labeled == tt)
            start = max(0, idx[0] - t_pre)
            stop = min(se.mov_f - 1, idx[-1] + t_post)
            cell_trans_indices[cc].append(len(trans_records))
            trans_records.append((cc, start, stop))

        prof = profiles[:, :, cc]
        _, outer_bounds = compute_roi_coms(prof)
        if use_com:
            coms, _ = compute_roi_coms(prof)
            cy, cx = int(round(coms[0, 1])), int(round(coms[0, 0]))
            y0 = y1 = cy
            x0 = x1 = cx
        else:
            y0, y1, x0, x1 = outer_bounds[0]

        y0 = max(0, y0 - win_radius)
        y1 = min(se.mov_y - 1, y1 + win_radius)
        x0 = max(0, x0 - win_radius)
        x1 = min(se.mov_x - 1, x1 + win_radius)
        roi_windows[cc] = [y0, y1, x0, x1]

    transient_info = []
    for cc in range(n_cells):
        times = np.array([trans_records[i][1:] for i in cell_trans_indices[cc]], dtype=int).reshape(-1, 2)
        transient_info.append({
            "times": times, "window": tuple(int(v) for v in roi_windows[cc]),
            "classification": np.full(len(times), np.nan), "is_artifact": False,
            "shapes": None, "shapes_corr": None, "shapes_mean": None, "corr_with_profile": None, "params": params,
        })

    # ---- load each needed frame once, accumulate per-transient mini-movies ----

    trans_at_frame = defaultdict(list)
    for ti_idx, (_, s, e) in enumerate(trans_records):
        for ff in range(s, e + 1):
            trans_at_frame[ff].append(ti_idx)

    shapes_lists: list[list] = [[] for _ in range(n_cells)]
    shapes_corr_lists: list[list] = [[] for _ in range(n_cells)]
    shapes_mean_lists: list[list] = [[] for _ in range(n_cells)]
    mini_movies: dict[int, np.ndarray] = {}

    for ff in sorted(trans_at_frame.keys()):
        frame = se.get_frame(ff)
        for ti_idx in trans_at_frame[ff]:
            cc, s, e = trans_records[ti_idx]
            y0, y1, x0, x1 = roi_windows[cc]

            if ff == s:
                mini_movies[ti_idx] = np.full((y1 - y0 + 1, x1 - x0 + 1, e - s + 1), np.nan)
            mini_movies[ti_idx][:, :, ff - s] = frame[y0 : y1 + 1, x0 : x1 + 1]

            if ff == e:
                tc_segment = time_courses[s : e + 1, cc]
                shape, shape_corr, shape_mean = _compute_one_active_shape(tc_segment, mini_movies[ti_idx])
                del mini_movies[ti_idx]
                shapes_lists[cc].append(shape)
                shapes_corr_lists[cc].append(shape_corr)
                shapes_mean_lists[cc].append(shape_mean)

    for cc in range(n_cells):
        if shapes_lists[cc]:
            transient_info[cc]["shapes"] = np.stack(shapes_lists[cc], axis=2)
            transient_info[cc]["shapes_corr"] = np.stack(shapes_corr_lists[cc], axis=2)
            transient_info[cc]["shapes_mean"] = np.stack(shapes_mean_lists[cc], axis=2)

    # ---- preserve classification/isArtifact from an existing transient_info, where possible ----

    existing = tc_struct.get("transient_info")
    if existing is not None:
        for cc in range(n_cells):
            old_times = existing[cc]["times"]
            new_times = transient_info[cc]["times"]
            transient_info[cc]["is_artifact"] = existing[cc]["is_artifact"]

            if np.array_equal(old_times, new_times):
                transient_info[cc]["classification"] = np.array(existing[cc]["classification"], copy=True)
                continue

            tc_old = np.zeros(se.mov_f, dtype=int)
            for tt in range(old_times.shape[0]):
                tc_old[old_times[tt, 0] : old_times[tt, 1] + 1] = tt + 1

            new_class = transient_info[cc]["classification"]
            for tt in range(new_times.shape[0]):
                overlap = np.unique(tc_old[new_times[tt, 0] : new_times[tt, 1] + 1])
                overlap = overlap[overlap != 0] - 1

                if overlap.size == 0:
                    continue
                elif overlap.size == 1:
                    new_class[tt] = existing[cc]["classification"][overlap[0]]
                else:
                    old_vals = existing[cc]["classification"][overlap]
                    finite = old_vals[~np.isnan(old_vals)]
                    if finite.size == old_vals.size and np.all(old_vals == old_vals[0]):
                        new_class[tt] = old_vals[0]
                    else:
                        new_class[tt] = np.nan
            transient_info[cc]["classification"] = new_class

    # ---- correlation between each transient's shape and the cell's profile ----

    for cc in range(n_cells):
        y0, y1, x0, x1 = transient_info[cc]["window"]
        prof_win = profiles[y0 : y1 + 1, x0 : x1 + 1, cc]
        shapes = transient_info[cc]["shapes"]
        if shapes is not None:
            shapes_flat = shapes.reshape(-1, shapes.shape[2])
            transient_info[cc]["corr_with_profile"] = correlation_vector_matrix(prof_win.ravel(), shapes_flat)
        else:
            transient_info[cc]["corr_with_profile"] = np.array([])

    tc_struct["transient_info"] = transient_info
    return transient_info
