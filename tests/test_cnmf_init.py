import numpy as np

from orbit.cnmf_init import estimate_background, finetune_component, gaussian_blur_movie, greedy_roi_init


def _synthetic_cell_movie(height=40, width=40, n_frames=200, seed=0):
    rng = np.random.default_rng(seed)
    movie = rng.standard_normal((height, width, n_frames)).astype(np.float64) * 0.1 + 1.0
    movie[5:12, 5:12, :] += 3 * np.clip(rng.standard_normal(n_frames), 0, None)
    movie[25:32, 25:32, :] += 3 * np.clip(rng.standard_normal(n_frames), 0, None)
    return np.clip(movie, 0, None)


def test_gaussian_blur_movie_preserves_shape_and_smooths_spatially():
    movie = _synthetic_cell_movie()
    blurred = gaussian_blur_movie(movie, sigma=2.0)

    assert blurred.shape == movie.shape
    # Blurring reduces frame-to-frame spatial variance (a rough proxy for
    # "single-pixel noise got smoothed out").
    assert blurred[..., 0].var() < movie[..., 0].var()


def test_finetune_component_recovers_a_single_blob():
    height, width, n_frames = 30, 30, 100
    rng = np.random.default_rng(0)
    trace_true = np.clip(rng.standard_normal(n_frames), 0, None)
    movie = rng.standard_normal((height, width, n_frames)) * 0.05
    footprint_true = np.zeros((height, width))
    footprint_true[10:15, 10:15] = 1.0
    movie += footprint_true[:, :, None] * trace_true[None, None, :]

    mask = np.zeros((height, width), dtype=bool)
    mask[8:17, 8:17] = True  # a slightly loose seed mask around the blob
    footprint, trace = finetune_component(movie, mask)

    assert footprint.shape == (height, width)
    assert (footprint >= 0).all()
    # Recovered footprint should peak inside the true blob.
    peak = np.unravel_index(np.argmax(footprint), footprint.shape)
    assert 10 <= peak[0] < 15 and 10 <= peak[1] < 15
    assert np.corrcoef(trace, trace_true)[0, 1] > 0.9


def test_greedy_roi_init_recovers_both_synthetic_blobs():
    movie = _synthetic_cell_movie()
    footprints, traces = greedy_roi_init(movie, n_components=2, gauss_sigma=1.0, init_radius=5)

    assert footprints.shape == (2, 40, 40)
    assert traces.shape == (2, 200)

    centers = [np.unravel_index(np.argmax(f), f.shape) for f in footprints]
    near_a = any(abs(c[0] - 8) < 4 and abs(c[1] - 8) < 4 for c in centers)
    near_b = any(abs(c[0] - 28) < 4 and abs(c[1] - 28) < 4 for c in centers)
    assert near_a and near_b


def test_estimate_background_shapes():
    movie = _synthetic_cell_movie()
    spatial, temporal = estimate_background(movie, n_components=1)

    assert spatial.shape == (40, 40, 1)
    assert temporal.shape == (1, 200)
    assert (spatial >= 0).all()
    assert (temporal >= 0).all()


def test_estimate_background_rank1_matches_sklearn_nmf_reconstruction_quality():
    # n_components=1 uses a hand-rolled rank-1 ALS instead of sklearn's
    # NMF (see estimate_background's docstring) -- factors from the two
    # methods aren't directly comparable (NMF's W/H split is only unique
    # up to a scale factor), so this instead checks the RECONSTRUCTION
    # (outer(spatial, temporal)) is at least as good as sklearn's own
    # fully-converged fit, on data with genuine smooth spatial/temporal
    # background structure (not just noise, which trivially "converges"
    # in one step regardless of method).
    from sklearn.decomposition import NMF

    height, width, n_frames = 60, 60, 120
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:height, 0:width]
    true_spatial = 1.0 + 0.5 * np.exp(-((yy - height / 2) ** 2 + (xx - width / 2) ** 2) / (2 * 20.0**2))
    true_temporal = np.clip(1.0 + 0.3 * np.sin(np.linspace(0, 6 * np.pi, n_frames)), 0, None)
    movie = true_spatial[:, :, None] * true_temporal[None, None, :]
    movie += rng.standard_normal(movie.shape) * 0.05
    movie = np.clip(movie, 0, None)

    spatial, temporal = estimate_background(movie.copy(), n_components=1)
    als_recon = spatial.reshape(-1, 1) @ temporal
    als_err = np.linalg.norm(movie.reshape(-1, n_frames) - als_recon)

    flat = np.clip(movie.reshape(-1, n_frames), 0, None)
    nmf = NMF(n_components=1, init="nndsvda", max_iter=200)
    sk_spatial = nmf.fit_transform(flat)
    sk_recon = sk_spatial @ nmf.components_
    sk_err = np.linalg.norm(flat - sk_recon)

    assert als_err < sk_err * 1.01  # ALS shouldn't reconstruct meaningfully worse than sklearn's converged fit
