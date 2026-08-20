import numpy as np

from orbit.cnmf import (
    _make_patches,
    _patch_bounds,
    cnmf_source_extraction,
    merge_overlapping_components,
    patch_cnmf_source_extraction,
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


def test_patch_bounds_covers_frame_without_a_near_duplicate_final_patch():
    # 256 % 75 == 31, so the naive stride sequence [0, 75, 150] lands only
    # 6px short of the true edge (156) -- appending a *new* patch there
    # (rather than shifting the last one) would nearly double coverage for
    # 6 extra pixels of frame.
    starts = _patch_bounds(size=256, patch_extent=100, overlap=25)
    assert starts == [0, 75, 156]
    assert starts[-1] + 100 == 256  # last patch flush with the far edge
    assert starts[0] == 0

    # a size that fits inside one patch collapses to a single patch
    assert _patch_bounds(size=50, patch_extent=100, overlap=25) == [0]


def test_make_patches_bounds_tile_the_full_frame():
    patches = _make_patches(height=120, width=90, patch_size=(70, 70), overlap=20)
    covered = np.zeros((120, 90), dtype=bool)
    for r0, r1, c0, c1 in patches:
        assert 0 <= r0 < r1 <= 120
        assert 0 <= c0 < c1 <= 90
        covered[r0:r1, c0:c1] = True
    assert covered.all()  # every pixel is inside at least one patch


def _synthetic_multi_cell_movie(height, width, centers, n_frames=150, radius=4.0, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    movie = rng.standard_normal((height, width, n_frames)) * 0.05 + 1.0
    for cy, cx in centers:
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius**2))
        activity = np.clip(rng.standard_normal(n_frames), 0, None) * 3.0
        movie += blob[:, :, None] * activity[None, None, :]
    return np.clip(movie, 0, None)


def test_patch_cnmf_source_extraction_finds_cells_in_separate_patches():
    centers = [(15, 15), (95, 95)]
    movie = _synthetic_multi_cell_movie(110, 110, centers)

    result = patch_cnmf_source_extraction(
        movie, patch_size=(65, 65), overlap=20, n_components_per_patch=3, n_iterations=2
    )

    assert len(result.masks) == len(result.traces) == len(result.spike_traces)
    for cy, cx in centers:
        dists = [
            np.hypot(*(np.argwhere(mask).mean(axis=0) - (cy, cx))) for mask in result.masks if mask.any()
        ]
        assert min(dists) < 5


def test_patch_cnmf_source_extraction_merges_a_component_straddling_a_patch_boundary():
    # A cell dead-center in the frame straddles the boundary between any
    # patch split; merge_overlapping_components should collapse the
    # duplicate detections from adjacent/overlapping patches into one
    # component rather than leaving two overlapping ROIs for the same cell.
    movie = _synthetic_multi_cell_movie(90, 60, [(45, 30)], radius=6.0)

    result = patch_cnmf_source_extraction(
        movie, patch_size=(55, 55), overlap=25, n_components_per_patch=3, n_iterations=2, merge_thresh=0.8
    )

    real_cell_masks = [
        mask for mask in result.masks if mask.any() and np.hypot(*(np.argwhere(mask).mean(axis=0) - (45, 30))) < 5
    ]
    assert len(real_cell_masks) == 1


def test_patch_cnmf_source_extraction_traces_are_measured_from_full_movie():
    centers = [(15, 15), (95, 95)]
    movie = _synthetic_multi_cell_movie(110, 110, centers)

    result = patch_cnmf_source_extraction(
        movie, patch_size=(65, 65), overlap=20, n_components_per_patch=3, n_iterations=2
    )

    for mask, trace in zip(result.masks, result.traces):
        rows, cols = np.nonzero(mask)
        expected = movie[rows, cols, :].mean(axis=0)
        assert np.allclose(trace, expected)


def test_patch_cnmf_source_extraction_handles_a_patch_with_no_components_found():
    # A patch with no components must not break patch placement/merging for
    # the rest of the frame.
    movie = np.random.default_rng(0).standard_normal((50, 50, 40)) * 0.05 + 1.0
    result = patch_cnmf_source_extraction(
        movie, patch_size=(30, 30), overlap=10, n_components_per_patch=2, n_iterations=1
    )
    assert len(result.masks) == len(result.traces) == len(result.spike_traces)


def test_patch_cnmf_source_extraction_reports_progress():
    movie = _synthetic_multi_cell_movie(70, 70, [(35, 35)], n_frames=60)
    calls = []
    patch_cnmf_source_extraction(
        movie,
        patch_size=(40, 40),
        overlap=10,
        n_components_per_patch=2,
        n_iterations=1,
        progress_callback=lambda done, total: calls.append((done, total)),
    )
    assert len(calls) > 0
    assert calls[-1][0] == calls[-1][1]  # last call reports completion
    assert all(done <= total for done, total in calls)
