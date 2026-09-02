import numpy as np

from orbit.cnmf_e_background import (
    _ZOOM_CHUNK_ELEMENTS,
    RingModel,
    _zoom_chunk_frames,
    fit_ring_model,
    fit_ring_weights,
    predict_ring_background,
    predict_ring_model_background,
    ring_model_background,
    ring_offsets,
)


def test_zoom_chunk_frames_keeps_the_temporarys_own_size_roughly_constant_across_fov_sizes():
    # predict_ring_model_background's own upsample processes a chunk of
    # frames at a time so its temporary's own memory stays bounded
    # regardless of field-of-view size (H * width_ds * chunk_size held
    # near _ZOOM_CHUNK_ELEMENTS, the exact configuration profiling
    # measured -- see that constant's own comment) -- a much bigger FOV
    # should get a proportionally SMALLER chunk automatically, not the
    # same chunk size (which would make the temporary itself grow
    # unbounded with FOV size instead).
    small = _zoom_chunk_frames(height=128, width_ds=32)
    baseline = _zoom_chunk_frames(height=256, width_ds=64)
    large = _zoom_chunk_frames(height=1024, width_ds=256)

    assert baseline == 50  # the exact profiled/measured configuration
    assert small > baseline > large
    # each dimension is held near _ZOOM_CHUNK_ELEMENTS, not exactly equal
    # to it (integer division), so allow a small amount of slack either
    # side rather than requiring an exact match.
    for height, width_ds, chunk in ((128, 32, small), (256, 64, baseline), (1024, 256, large)):
        assert abs(height * width_ds * chunk - _ZOOM_CHUNK_ELEMENTS) <= height * width_ds


def test_zoom_chunk_frames_never_returns_less_than_one():
    # An enormous field of view shouldn't make this divide down to 0 --
    # a chunk size of 0 would make predict_ring_model_background's own
    # loop never advance.
    assert _zoom_chunk_frames(height=100_000, width_ds=100_000) >= 1


def test_ring_offsets_excludes_inner_disk_and_includes_only_up_to_outer_radius():
    offsets = ring_offsets(2, 5)
    dists = np.hypot(offsets[:, 0], offsets[:, 1])
    assert (dists > 2).all()
    assert (dists <= 5).all()

    # brute-force reference count over the same bounding box
    radius = int(np.ceil(5))
    expected = sum(
        1
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if 2 < np.hypot(dr, dc) <= 5
    )
    assert len(offsets) == expected


def test_ring_offsets_is_symmetric():
    offsets = ring_offsets(2, 5)
    offset_set = {tuple(o) for o in offsets.tolist()}
    assert all((-dr, -dc) in offset_set for dr, dc in offsets.tolist())


def _shared_source_residual(height=20, width=20, n_frames=300, seed=0):
    rng = np.random.default_rng(seed)
    source_a = rng.standard_normal(n_frames)
    source_b = rng.standard_normal(n_frames)
    residual = np.zeros((height, width, n_frames))
    for r in range(height):
        for c in range(width):
            residual[r, c] = 0.6 * source_a + 0.3 * source_b + 0.01 * rng.standard_normal(n_frames)
    return residual


def test_fit_ring_weights_shapes():
    residual = _shared_source_residual(height=10, width=10, n_frames=100)
    neuron_mask = np.zeros((10, 10), dtype=bool)
    offsets = ring_offsets(1, 3)

    weights, valid_mask = fit_ring_weights(residual, neuron_mask, offsets, max_fit_frames=100)

    assert weights.shape == (10, 10, len(offsets))
    assert valid_mask.shape == (10, 10, len(offsets))


def test_fit_ring_weights_recovers_a_known_linear_ring_relationship():
    # Every non-neuron pixel's trace is EXACTLY 0.6*source_a + 0.3*source_b
    # (plus tiny noise) -- any of a pixel's ring neighbors carries the
    # same underlying signal, so the fitted prediction should closely
    # match the true (noise-free) relationship, not just correlate.
    residual = _shared_source_residual(height=20, width=20, n_frames=300)
    neuron_mask = np.zeros((20, 20), dtype=bool)
    offsets = ring_offsets(2, 4)

    weights, valid_mask = fit_ring_weights(residual, neuron_mask, offsets, max_fit_frames=300, ridge=1e-3)
    predicted = predict_ring_background(residual, weights, valid_mask, offsets)

    # interior pixels only -- edge pixels have an incomplete ring
    interior_pred = predicted[6:14, 6:14, :]
    interior_true = residual[6:14, 6:14, :]
    corr = np.corrcoef(interior_pred.ravel(), interior_true.ravel())[0, 1]
    assert corr > 0.99


def test_fit_ring_weights_skips_neuron_pixels():
    residual = _shared_source_residual(height=10, width=10, n_frames=100)
    neuron_mask = np.zeros((10, 10), dtype=bool)
    neuron_mask[4:6, 4:6] = True
    offsets = ring_offsets(1, 3)

    weights, valid_mask = fit_ring_weights(residual, neuron_mask, offsets, max_fit_frames=100)

    assert not valid_mask[4:6, 4:6, :].any()
    assert (weights[4:6, 4:6, :] == 0).all()


def test_predict_ring_background_shape_matches_full_resolution_and_full_frame_count():
    residual = _shared_source_residual(height=12, width=14, n_frames=80)
    neuron_mask = np.zeros((12, 14), dtype=bool)
    offsets = ring_offsets(1, 3)
    weights, valid_mask = fit_ring_weights(residual, neuron_mask, offsets, max_fit_frames=80)

    background = predict_ring_background(residual, weights, valid_mask, offsets)
    assert background.shape == (12, 14, 80)


def test_fit_ring_model_and_predict_round_trip_matches_one_shot_helper():
    residual = _shared_source_residual(height=20, width=20, n_frames=200)
    neuron_mask = np.zeros((20, 20), dtype=bool)

    model = fit_ring_model(residual, neuron_mask, ring_inner_radius=2, ring_outer_radius=4, ds_ratio=1, max_fit_frames=200)
    assert isinstance(model, RingModel)
    predicted = predict_ring_model_background(residual, model)

    one_shot = ring_model_background(residual, neuron_mask, ring_inner_radius=2, ring_outer_radius=4, ds_ratio=1, max_fit_frames=200)
    assert np.allclose(predicted, one_shot)


def test_ring_model_background_shape_is_full_resolution_regardless_of_downsampling():
    residual = _shared_source_residual(height=37, width=41, n_frames=60)
    neuron_mask = np.zeros((37, 41), dtype=bool)

    background = ring_model_background(residual, neuron_mask, ring_inner_radius=4, ring_outer_radius=8, ds_ratio=4, max_fit_frames=60)
    assert background.shape == (37, 41, 60)


def test_ring_model_background_end_to_end_recovers_ring_structured_background():
    # A background that's genuinely spatially-varying (several localized
    # smooth "sources", not one global rank-1 field), plus a bright
    # compact "neuron" blob excluded from the fit via neuron_mask.
    height, width, n_frames = 40, 40, 200
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:height, 0:width]

    n_sources = 4
    centers = rng.integers(0, 40, size=(n_sources, 2))
    sources = [rng.standard_normal(n_frames) for _ in range(n_sources)]
    true_background = np.zeros((height, width, n_frames))
    for (cy, cx), source in zip(centers, sources):
        weight = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2)) / (2 * 8.0**2))
        true_background += weight[:, :, None] * source[None, None, :]

    neuron_mask = np.zeros((height, width), dtype=bool)
    neuron_mask[18:23, 18:23] = True
    neuron_signal = np.zeros((height, width, n_frames))
    neuron_signal[18:23, 18:23, :] = 5.0 * np.clip(rng.standard_normal(n_frames), 0, None)[None, None, :]

    residual = true_background + neuron_signal + 0.02 * rng.standard_normal((height, width, n_frames))

    predicted = ring_model_background(residual, neuron_mask, ring_inner_radius=6, ring_outer_radius=12, ds_ratio=2, max_fit_frames=150)

    # Compare away from the neuron and away from the edges (incomplete rings).
    mask = np.ones((height, width), dtype=bool)
    mask[14:27, 14:27] = False  # neuron + its immediate surroundings
    mask[:6, :], mask[-6:, :], mask[:, :6], mask[:, -6:] = False, False, False, False

    corr = np.corrcoef(predicted[mask].ravel(), true_background[mask].ravel())[0, 1]
    assert corr > 0.6
