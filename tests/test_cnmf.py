import numpy as np

from orbit.cnmf import (
    cnmf_source_extraction,
    merge_overlapping_components,
    threshold_footprint,
    update_spatial_components,
    update_temporal_components,
)


def _synthetic_cell_movie(height=30, width=30, n_frames=150, seed=0):
    rng = np.random.default_rng(seed)
    movie = rng.standard_normal((height, width, n_frames)).astype(np.float64) * 0.1 + 1.0
    movie[5:10, 5:10, :] += 3 * np.clip(rng.standard_normal(n_frames), 0, None)
    movie[20:25, 20:25, :] += 3 * np.clip(rng.standard_normal(n_frames), 0, None)
    return np.clip(movie, 0, None)


def test_threshold_footprint_drops_low_weights_and_keeps_the_peak_component():
    footprint = np.zeros((10, 10))
    footprint[3:6, 3:6] = 1.0  # the "real" blob
    footprint[8, 8] = 0.01  # a disconnected, low-weight noise pixel elsewhere

    cleaned = threshold_footprint(footprint, quantile=0.5)

    assert cleaned[4, 4] > 0  # peak survives
    assert cleaned[8, 8] == 0  # isolated low-weight noise pixel is gone


def test_threshold_footprint_handles_all_zero_input():
    footprint = np.zeros((5, 5))
    assert threshold_footprint(footprint).sum() == 0


def test_update_spatial_components_shapes_and_nonnegative():
    movie = _synthetic_cell_movie()
    footprints = np.zeros((2, 30, 30))
    footprints[0, 5:10, 5:10] = 1.0
    footprints[1, 20:25, 20:25] = 1.0
    traces = np.stack([movie[5:10, 5:10, :].mean(axis=(0, 1)), movie[20:25, 20:25, :].mean(axis=(0, 1))])
    background_temporal = np.ones((1, movie.shape[2]))

    new_footprints, new_background_spatial = update_spatial_components(
        movie, footprints, traces, background_temporal, search_radius=8
    )

    assert new_footprints.shape == footprints.shape
    assert new_background_spatial.shape == (30, 30, 1)
    assert (new_footprints >= 0).all()
    assert (new_background_spatial >= 0).all()
    # Recovered weight should be concentrated near each blob's own footprint.
    assert new_footprints[0, 5:10, 5:10].sum() > new_footprints[0, 20:25, 20:25].sum()
    assert new_footprints[1, 20:25, 20:25].sum() > new_footprints[1, 5:10, 5:10].sum()


def test_update_temporal_components_shapes_and_spikes_nonnegative():
    movie = _synthetic_cell_movie()
    footprints = np.zeros((2, 30, 30))
    footprints[0, 5:10, 5:10] = 1.0
    footprints[1, 20:25, 20:25] = 1.0
    traces = np.stack([movie[5:10, 5:10, :].mean(axis=(0, 1)), movie[20:25, 20:25, :].mean(axis=(0, 1))])
    background_spatial = np.full((30, 30, 1), 0.1)
    background_temporal = np.ones((1, movie.shape[2]))
    g_list = [0.9, 0.9]
    noise_stds = [0.2, 0.2]

    new_c, new_s, new_bg_temporal = update_temporal_components(
        movie, footprints, traces, background_spatial, background_temporal, g_list, noise_stds
    )

    assert new_c.shape == traces.shape
    assert new_s.shape == traces.shape
    assert new_bg_temporal.shape == background_temporal.shape
    assert (new_s >= 0).all()
    assert (new_bg_temporal >= 0).all()


def test_merge_overlapping_components_combines_correlated_overlapping_pair():
    footprint_a = np.zeros((10, 10))
    footprint_a[2:6, 2:6] = 1.0
    footprint_b = np.zeros((10, 10))
    footprint_b[3:7, 3:7] = 1.0  # overlaps footprint_a
    footprint_c = np.zeros((10, 10))
    footprint_c[8, 8] = 1.0  # far away, no overlap

    shared_trace = np.abs(np.sin(np.linspace(0, 10, 50)))
    footprints = np.stack([footprint_a, footprint_b, footprint_c])
    traces = np.stack([shared_trace, shared_trace + 0.001, shared_trace[::-1]])
    spikes = np.zeros_like(traces)
    g_list = [0.9, 0.9, 0.9]

    merged_footprints, merged_traces, merged_spikes, merged_g = merge_overlapping_components(
        footprints, traces, spikes, g_list, merge_thresh=0.9
    )

    assert len(merged_footprints) == 2  # a+b merged, c stays separate
    assert len(merged_traces) == 2
    assert len(merged_spikes) == 2
    assert len(merged_g) == 2


def test_cnmf_source_extraction_recovers_both_synthetic_blobs():
    movie = _synthetic_cell_movie()

    result = cnmf_source_extraction(
        movie, n_components=2, gauss_sigma=1.0, init_radius=4, search_radius=8, n_iterations=2
    )

    assert len(result.masks) == len(result.traces) == len(result.spike_traces) == 2
    centers = [np.argwhere(mask).mean(axis=0) for mask in result.masks if mask.any()]
    near_a = any(np.hypot(cy - 7.5, cx - 7.5) < 4 for cy, cx in centers)
    near_b = any(np.hypot(cy - 22.5, cx - 22.5) < 4 for cy, cx in centers)
    assert near_a and near_b

    for trace, spikes in zip(result.traces, result.spike_traces):
        assert trace.shape == (movie.shape[2],)
        assert spikes.shape == (movie.shape[2],)
        assert (spikes >= 0).all()


def test_cnmf_source_extraction_traces_are_measured_not_the_oasis_reconstruction():
    # ROI.trace should be the actual masked-mean signal from the movie
    # (matching every other extraction method's convention), not OASIS's
    # denoised/reconstructed calcium -- a model fit, not the measurement.
    movie = _synthetic_cell_movie()

    result = cnmf_source_extraction(
        movie, n_components=2, gauss_sigma=1.0, init_radius=4, search_radius=8, n_iterations=2
    )

    for mask, trace in zip(result.masks, result.traces):
        rows, cols = np.nonzero(mask)
        expected = movie[rows, cols, :].mean(axis=0)
        assert np.allclose(trace, expected)
