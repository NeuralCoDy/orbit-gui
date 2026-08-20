"""Automatic true/false transient classification, ported from the `seudo`
package's auto_classify.py (itself a port of SEUDO's
autoClassifyTransients.m). Supports all three criteria the GUI exposes:

- 'corr': plain correlation between a transient's (unblurred) spatial
  shape and this cell's profile, thresholded directly.
- 'res_ratio': fits [this cell's profile, every other cell's profile] to
  the transient's (blurred) shape; how much residual is added by leaving
  this cell's profile OUT of the fit vs. including it. High means the
  shape isn't well explained by this cell (likely contamination) --
  *low* resRatio is the "true" direction, unlike corr.
- 'seudo_residual': fraction of this cell's own least-squares coefficient
  that SEUDO's sparse fit diverts into the unmodeled "blob" basis
  instead. *High* is the "true" direction here (per upstream's explicit
  convention) -- unlike resRatio.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .constants import VAL_FALSE, VAL_TRUE
from .seudo_residual import SEUDO_RESIDUAL_DEFAULTS, compute_seudo_residual_fractions
from .stats import correlation_vector_matrix


def _blur_stack(images: np.ndarray, sigma: float) -> np.ndarray:
    """images: (Y, X, T). Blur each frame spatially, independently."""
    if images.size == 0 or sigma == 0:
        return images
    return gaussian_filter(images, sigma=(sigma, sigma, 0), mode="nearest")


def _crop_to_window(arr: np.ndarray, win_sub_y: np.ndarray, win_sub_x: np.ndarray) -> np.ndarray:
    return arr[win_sub_y, :][:, win_sub_x]


def auto_classify_transients(
    se,
    which_struct: str = "default",
    which_cells=None,
    overwrite: bool = False,
    save_results: bool = True,
    profile_field_name: str = "shapes",
    ignore_zeros: bool = False,
    minimum_window: bool = True,
    blur_radius: float = 1.0,
    blur_profiles_for_fitting: bool = False,
    corr_thresh: float = 0.4,
    weighted_trans: bool = False,
    res_ratio_mean: bool = True,
    criterion: str = "corr",
    res_ratio_thresh: float = 0.5,
    seudo_residual_thresh: float = 0.5,
    seudo_kwargs: dict | None = None,
    progress_callback=None,
) -> list[dict]:
    """progress_callback, if given, is called as progress_callback(done,
    total, cell_id) after each cell finishes."""
    if criterion not in ("corr", "res_ratio", "seudo_residual"):
        raise ValueError(f"criterion must be 'corr', 'res_ratio', or 'seudo_residual', got {criterion!r}")
    tc_struct = se._resolve_tc_struct(which_struct)
    if tc_struct.get("transient_info") is None:
        se.compute_transient_info(which_struct)
        tc_struct = se._resolve_tc_struct(which_struct)

    transient_info = tc_struct["transient_info"]
    n_cells = se.n_cells
    which_cells = list(range(n_cells)) if which_cells is None else which_cells

    if save_results:
        has_class = {cc: not np.all(np.isnan(transient_info[cc]["classification"])) for cc in which_cells}
        if any(has_class.values()) and not overwrite:
            n_bad = sum(has_class.values())
            raise ValueError(
                f"classification exists for {n_bad} of {len(which_cells)} cells, to overwrite call with overwrite=True"
            )

    params = dict(
        which_cells=which_cells, overwrite=overwrite, save_results=save_results,
        profile_field_name=profile_field_name, ignore_zeros=ignore_zeros, minimum_window=minimum_window,
        blur_radius=blur_radius, blur_profiles_for_fitting=blur_profiles_for_fitting, corr_thresh=corr_thresh,
        weighted_trans=weighted_trans, res_ratio_mean=res_ratio_mean, criterion=criterion,
        res_ratio_thresh=res_ratio_thresh, seudo_residual_thresh=seudo_residual_thresh, seudo_kwargs=seudo_kwargs,
    )

    results = []
    n_total = len(which_cells)

    for done, cell_id in enumerate(which_cells, start=1):
        ti = transient_info[cell_id]
        n_trans = ti["times"].shape[0]

        if n_trans == 0:
            metric = dict(cfrac=np.nan, rfrac=np.nan, corrs=np.array([]), classification=np.array([]), params=params)
            results.append(metric)
            if save_results:
                ti["classification"] = np.array([])
                ti["auto_class"] = metric
            if progress_callback is not None:
                progress_callback(done, n_total, cell_id)
            continue

        tc = tc_struct["tc"][:, cell_id]
        evts = ti["times"]
        tc_amps = np.array([np.max(tc[s : e + 1]) for s, e in evts], dtype=float)

        y0, y1, x0, x1 = ti["window"]
        x_prof = se.profiles[y0 : y1 + 1, x0 : x1 + 1, cell_id]
        other_idx = [c for c in range(n_cells) if c != cell_id]
        x_other = se.profiles[y0 : y1 + 1, x0 : x1 + 1, other_idx]

        if minimum_window:
            win_sub_y = np.any(x_prof > 0, axis=1)
            win_sub_x = np.any(x_prof > 0, axis=0)
        else:
            win_sub_y = np.ones(x_prof.shape[0], dtype=bool)
            win_sub_x = np.ones(x_prof.shape[1], dtype=bool)

        x_prof = _crop_to_window(x_prof, win_sub_y, win_sub_x)
        f_fits = ti[profile_field_name][win_sub_y, :, :][:, win_sub_x, :]
        x_other = x_other[win_sub_y, :, :][:, win_sub_x, :]
        w_y, w_x = x_prof.shape

        if blur_profiles_for_fitting:
            x_prof_for_fit = gaussian_filter(x_prof, sigma=blur_radius, mode="nearest")
            x_other_for_fit = _blur_stack(x_other, blur_radius)
        else:
            x_prof_for_fit = x_prof
            x_other_for_fit = x_other

        x_prof_for_fit = x_prof_for_fit.reshape(-1)
        x_other_for_fit = x_other_for_fit.reshape(w_y * w_x, -1)
        x_other_for_fit = x_other_for_fit[:, x_other_for_fit.sum(axis=0) != 0]

        x_prof_flat = x_prof.reshape(-1).astype(float)
        f_fits_nofilt = f_fits.reshape(w_y * w_x, n_trans)
        f_fits_blurred = _blur_stack(f_fits, blur_radius).reshape(w_y * w_x, n_trans)

        design = np.column_stack([x_prof_for_fit, x_other_for_fit])
        fit_vals, *_ = np.linalg.lstsq(design, f_fits_blurred, rcond=None)

        res_evt = f_fits_blurred - x_other_for_fit @ fit_vals[1:, :]
        res = f_fits_blurred - design @ fit_vals
        res_img = res.reshape(w_y, w_x, n_trans)

        res_norms = np.sum(res**2, axis=0)
        res_no_cell = np.sum(res_evt**2, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            res_ratios = res_norms / res_no_cell

        metric = dict(
            resNorms=res_norms, resNoCell=res_no_cell, resRatios=res_ratios, residuals=res_img,
            residualsWithoutSource=res_evt.reshape(w_y, w_x, n_trans),
        )

        x_prof_corr = x_prof_flat.copy()
        f_fits_corr = f_fits_nofilt.astype(float).copy()
        if ignore_zeros:
            zero_pix = x_prof_flat == 0
            x_prof_corr[zero_pix] = np.nan
            f_fits_corr[zero_pix, :] = np.nan

        corrs = correlation_vector_matrix(x_prof_corr, f_fits_corr)
        metric["corrs"] = corrs

        new_classification = np.full(n_trans, VAL_FALSE)
        if criterion == "corr":
            new_classification[corrs >= corr_thresh] = VAL_TRUE
        elif criterion == "res_ratio":
            new_classification[res_ratios <= res_ratio_thresh] = VAL_TRUE
        else:
            normalized_kwargs = dict(SEUDO_RESIDUAL_DEFAULTS)
            normalized_kwargs.update(seudo_kwargs or {})
            cache = ti.get("seudo_residual_cache")
            if cache is not None and cache["seudo_kwargs"] == normalized_kwargs and cache["values"].shape[0] == n_trans:
                seudo_fractions = cache["values"]
            else:
                seudo_fractions = compute_seudo_residual_fractions(se, cell_id, ti, **normalized_kwargs)
                ti["seudo_residual_cache"] = dict(seudo_kwargs=normalized_kwargs, values=seudo_fractions)
            metric["seudoResidual"] = seudo_fractions
            new_classification[seudo_fractions >= seudo_residual_thresh] = VAL_TRUE

        is_false = new_classification == VAL_FALSE
        is_true = ~is_false
        res_den2 = res_no_cell[is_true].sum()
        res_den = res_no_cell[is_false].sum()
        weights = (tc_amps / tc_amps.sum()) if weighted_trans else np.full(n_trans, 1.0 / n_trans)

        cfrac = rfrac = rfrac2 = rfrac3 = 0.0
        for kk in range(n_trans):
            tmp_res = res_img[:, :, kk].copy()
            if ignore_zeros:
                tmp_res.reshape(-1)[x_prof_flat == 0] = np.nan
            res_sq_sum = np.nansum(tmp_res**2)

            if is_false[kk]:
                cfrac += weights[kk]
                rd = res_den
            else:
                rd = res_den2

            if res_ratio_mean:
                contrib = res_sq_sum / rd
                rfrac3_contrib = res_sq_sum / res_no_cell.sum()
            else:
                denom = np.nansum(res_evt[:, kk] ** 2)
                contrib = res_sq_sum / denom
                rfrac3_contrib = contrib

            if is_false[kk]:
                rfrac += contrib
            else:
                rfrac2 += contrib
            rfrac3 += rfrac3_contrib

        if not res_ratio_mean:
            if is_false.sum() > 0:
                rfrac /= is_false.sum()
            if is_true.sum() > 0:
                rfrac2 /= is_true.sum()
            rfrac3 /= n_trans

        metric["cfrac"] = cfrac
        metric["rfrac"] = rfrac
        metric["rfrac2"] = rfrac2
        metric["rfrac3"] = rfrac3
        metric["fullVal"] = np.sum(tc_amps * res_ratios) / np.sum(tc_amps)
        metric["classification"] = new_classification
        metric["params"] = params
        results.append(metric)

        if save_results:
            ti["classification"] = new_classification
            ti["auto_class"] = metric
        if progress_callback is not None:
            progress_callback(done, n_total, cell_id)

    return results
