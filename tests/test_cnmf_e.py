import warnings

import numpy as np
import pytest

from orbit.cnmf import cnmf_source_extraction
from orbit.cnmf_e import cnmf_e_source_extraction, patch_cnmf_e_source_extraction


def _calcium_trace(n_frames, rng, rate=0.03, decay=0.9, amp=4.0):
    spikes = (rng.random(n_frames) < rate).astype(float) * rng.uniform(1, 2, n_frames)
    trace = np.zeros(n_frames)
    for t in range(1, n_frames):
        trace[t] = decay * trace[t - 1] + spikes[t]
    return amp * trace


def _synthetic_1p_movie(height=40, width=40, n_frames=200, seed=0):
    rng = np.random.default_rng(seed)
    movie = rng.standard_normal((height, width, n_frames)).astype(np.float64) * 0.05 + 1.0
    movie[8:13, 8:13, :] += _calcium_trace(n_frames, rng)
    movie[28:33, 28:33, :] += _calcium_trace(n_frames, rng)
    return np.clip(movie, 0, None)


def _synthetic_multi_cell_1p_movie(height, width, centers, n_frames=150, radius=4.0, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    movie = rng.standard_normal((height, width, n_frames)) * 0.05 + 1.0
    for cy, cx in centers:
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius**2))
        activity = _calcium_trace(n_frames, rng)
        movie += blob[:, :, None] * activity[None, None, :]
    return np.clip(movie, 0, None)


_CNMF_E_KWARGS = dict(
    gauss_sigma=1.0, init_radius=4, search_radius=8, min_corr=0.8, min_pnr=8.0,
    ring_inner_radius=6, ring_outer_radius=10, ring_downsample=2, ring_max_fit_frames=150,
)


def test_cnmf_e_source_extraction_recovers_both_synthetic_blobs():
    movie = _synthetic_1p_movie()

    result = cnmf_e_source_extraction(movie, n_components=4, n_iterations=2, **_CNMF_E_KWARGS)

    assert len(result.masks) == len(result.traces) == len(result.spike_traces)
    centers = [np.argwhere(mask).mean(axis=0) for mask in result.masks if mask.any()]
    near_a = any(np.hypot(cy - 10, cx - 10) < 5 for cy, cx in centers)
    near_b = any(np.hypot(cy - 30, cx - 30) < 5 for cy, cx in centers)
    assert near_a and near_b

    for trace, spikes in zip(result.traces, result.spike_traces):
        assert trace.shape == (movie.shape[2],)
        assert spikes.shape == (movie.shape[2],)


def test_cnmf_e_source_extraction_traces_are_measured_not_the_oasis_reconstruction():
    movie = _synthetic_1p_movie()
    result = cnmf_e_source_extraction(movie, n_components=4, n_iterations=2, **_CNMF_E_KWARGS)

    for mask, trace in zip(result.masks, result.traces):
        rows, cols = np.nonzero(mask)
        expected = movie[rows, cols, :].mean(axis=0)
        assert np.allclose(trace, expected)


def test_cnmf_e_source_extraction_returns_empty_result_when_no_seeds_qualify():
    movie = _synthetic_1p_movie()
    result = cnmf_e_source_extraction(movie, n_components=4, min_corr=0.999999, min_pnr=1000.0)
    assert result.masks == [] and result.traces == [] and result.spike_traces == []


def _best_trace_correlation(masks, traces, true_center, true_trace, radius=5):
    best = -1.0
    for mask, trace in zip(masks, traces):
        if not mask.any():
            continue
        cy, cx = np.argwhere(mask).mean(axis=0)
        if np.hypot(cy - true_center[0], cx - true_center[1]) < radius:
            best = max(best, np.corrcoef(trace, true_trace)[0, 1])
    return best


def test_cnmf_e_source_extraction_outperforms_plain_cnmf_under_ring_structured_background():
    # Without a test like this, CNMF-E could be plumbing-correct but
    # algorithmically inert (e.g. silently equivalent to plain CNMF) --
    # this is what actually validates the ring-model background helps.
    height, width, n_frames = 40, 40, 200
    rng = np.random.default_rng(1)
    true_trace = _calcium_trace(n_frames, rng, rate=0.04, amp=5.0)
    movie = rng.standard_normal((height, width, n_frames)).astype(np.float64) * 0.05 + 1.0
    movie[18:23, 18:23, :] += true_trace

    # Several spatially-localized (NOT global-rank-1) background sources.
    n_sources = 6
    yy, xx = np.mgrid[0:height, 0:width]
    rng_bg = np.random.default_rng(2)
    centers = rng_bg.integers(0, height, size=(n_sources, 2))
    for cy, cx in centers:
        weight = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 8.0**2))
        source = rng.standard_normal(n_frames) * 0.8
        movie += weight[:, :, None] * source[None, None, :]
    movie = np.clip(movie, 0, None)

    cnmf_result = cnmf_source_extraction(movie, n_components=3, gauss_sigma=1.0, init_radius=4, search_radius=8, n_iterations=2)
    cnmf_e_result = cnmf_e_source_extraction(
        movie, n_components=3, n_iterations=2, gauss_sigma=1.0, init_radius=4, search_radius=8,
        min_corr=0.7, min_pnr=5.0, ring_inner_radius=6, ring_outer_radius=12, ring_downsample=2,
        ring_max_fit_frames=150,
    )

    cnmf_quality = _best_trace_correlation(cnmf_result.masks, cnmf_result.traces, (20, 20), true_trace)
    cnmf_e_quality = _best_trace_correlation(cnmf_e_result.masks, cnmf_e_result.traces, (20, 20), true_trace)

    assert cnmf_e_quality > cnmf_quality


def test_patch_cnmf_e_source_extraction_finds_cells_in_separate_patches():
    centers = [(15, 15), (95, 95)]
    movie = _synthetic_multi_cell_1p_movie(110, 110, centers)

    result = patch_cnmf_e_source_extraction(
        movie, patch_size=(65, 65), overlap=20, n_components_per_patch=3, n_iterations=2,
        ring_inner_radius=4, ring_outer_radius=8, ring_downsample=2, ring_max_fit_frames=100,
        min_corr=0.6, min_pnr=5.0,
    )

    assert len(result.masks) == len(result.traces) == len(result.spike_traces)
    for cy, cx in centers:
        dists = [np.hypot(*(np.argwhere(mask).mean(axis=0) - (cy, cx))) for mask in result.masks if mask.any()]
        # A looser tolerance than plain CNMF's equivalent patch test:
        # the ring model's own spatial downsampling adds some centroid
        # imprecision on top of ordinary spatial-update noise.
        assert min(dists) < 8


def test_patch_cnmf_e_source_extraction_merges_a_component_straddling_a_patch_boundary():
    # n_components_per_patch=1, matching this scene's single true cell:
    # unlike plain CNMF's greedy_roi_init (which always seeds exactly N
    # components regardless of whether real signal remains), CNMF-E's
    # corr*PNR seeding only returns candidates that clear min_corr/
    # min_pnr -- requesting more candidates than there are real cells
    # risks picking up spurious noise-based seeds elsewhere in a patch,
    # which is a real (and separately meaningful) algorithmic property,
    # not what this test is about.
    movie = _synthetic_multi_cell_1p_movie(90, 60, [(45, 30)], radius=6.0)

    result = patch_cnmf_e_source_extraction(
        movie, patch_size=(55, 55), overlap=25, n_components_per_patch=1, n_iterations=2, merge_thresh=0.8,
        ring_inner_radius=4, ring_outer_radius=8, ring_downsample=2, ring_max_fit_frames=100,
        min_corr=0.6, min_pnr=5.0,
    )

    assert len(result.masks) == 1


def test_patch_cnmf_e_source_extraction_warns_when_overlap_is_smaller_than_ring_outer_radius():
    movie = _synthetic_multi_cell_1p_movie(60, 60, [(30, 30)])

    with pytest.warns(UserWarning, match="overlap"):
        patch_cnmf_e_source_extraction(
            movie, patch_size=(40, 40), overlap=5, n_components_per_patch=2, n_iterations=1,
            ring_inner_radius=4, ring_outer_radius=8, ring_downsample=2, ring_max_fit_frames=100,
        )


def test_patch_cnmf_e_source_extraction_does_not_warn_when_overlap_is_sufficient():
    movie = _synthetic_multi_cell_1p_movie(60, 60, [(30, 30)])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        patch_cnmf_e_source_extraction(
            movie, patch_size=(40, 40), overlap=10, n_components_per_patch=2, n_iterations=1,
            ring_inner_radius=4, ring_outer_radius=8, ring_downsample=2, ring_max_fit_frames=100,
        )


def test_patch_cnmf_e_source_extraction_reports_progress():
    movie = _synthetic_multi_cell_1p_movie(90, 60, [(15, 15), (75, 45)], radius=4.0)

    calls = []
    patch_cnmf_e_source_extraction(
        movie, patch_size=(50, 50), overlap=20, n_components_per_patch=2, n_iterations=1,
        ring_inner_radius=4, ring_outer_radius=8, ring_downsample=2, ring_max_fit_frames=100,
        progress_callback=lambda done, total: calls.append((done, total)),
    )

    assert calls
    assert calls[-1][0] == calls[-1][1]
