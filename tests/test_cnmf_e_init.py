import numpy as np

from orbit.cnmf_deconvolution import estimate_noise_std
from orbit.cnmf_e_init import cnmf_e_init, cnmf_e_seed_candidates, noise_std_projection, peak_to_noise_ratio_projection


def _calcium_trace(n_frames: int, rng: np.random.Generator, rate: float = 0.03, decay: float = 0.9, amp: float = 4.0):
    # A smooth exponential-decay transient, not per-frame i.i.d. noise --
    # PNR/noise-std estimation assumes signal power is concentrated at
    # LOW frequencies (see estimate_noise_std), so a "cell" trace needs
    # actual temporal structure to read as different from noise.
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


def test_noise_std_projection_matches_per_trace_estimate_noise_std():
    movie = _synthetic_1p_movie(height=10, width=10, n_frames=200)
    projected = noise_std_projection(movie)

    assert projected.shape == (10, 10)
    for r in range(10):
        for c in range(10):
            assert np.isclose(projected[r, c], estimate_noise_std(movie[r, c, :]))


def test_peak_to_noise_ratio_projection_higher_at_active_pixel_than_background():
    movie = _synthetic_1p_movie()
    pnr = peak_to_noise_ratio_projection(movie)

    assert pnr.shape == movie.shape[:2]
    assert pnr[10, 10] > pnr[0, 0]


def test_cnmf_e_seed_candidates_finds_both_synthetic_blobs():
    movie = _synthetic_1p_movie()
    seeds = cnmf_e_seed_candidates(
        movie, gauss_sigma=1.0, min_corr=0.8, min_pnr=8.0, n_components=4, min_separation_frac=0.15
    )

    near_a = any(np.hypot(r - 10, c - 10) < 5 for r, c in seeds)
    near_b = any(np.hypot(r - 30, c - 30) < 5 for r, c in seeds)
    assert near_a and near_b


def test_cnmf_e_seed_candidates_respects_min_corr_and_min_pnr_thresholds():
    movie = _synthetic_1p_movie()
    # Thresholds set above what this synthetic movie's pixels can reach.
    seeds = cnmf_e_seed_candidates(movie, gauss_sigma=1.0, min_corr=0.999999, min_pnr=1000.0, n_components=4)
    assert seeds == []


def test_cnmf_e_seed_candidates_returns_at_most_n_components():
    movie = _synthetic_1p_movie()
    seeds = cnmf_e_seed_candidates(movie, gauss_sigma=1.0, min_corr=0.5, min_pnr=1.0, n_components=3)
    assert len(seeds) <= 3


def test_cnmf_e_init_recovers_both_synthetic_blobs():
    movie = _synthetic_1p_movie()
    footprints, traces = cnmf_e_init(
        movie, n_components=4, gauss_sigma=1.0, init_radius=4, min_corr=0.8, min_pnr=8.0
    )

    assert footprints.shape[1:] == (40, 40)
    assert traces.shape[1] == 200
    assert (footprints >= 0).all()

    centers = [np.unravel_index(np.argmax(f), f.shape) for f in footprints]
    near_a = any(abs(r - 10) < 5 and abs(c - 10) < 5 for r, c in centers)
    near_b = any(abs(r - 30) < 5 and abs(c - 30) < 5 for r, c in centers)
    assert near_a and near_b


def test_cnmf_e_init_returns_empty_when_no_seeds_qualify():
    movie = _synthetic_1p_movie()
    footprints, traces = cnmf_e_init(
        movie, n_components=4, gauss_sigma=1.0, init_radius=4, min_corr=0.999999, min_pnr=1000.0
    )
    assert footprints.shape == (0, 40, 40)
    assert traces.shape == (0, 200)
