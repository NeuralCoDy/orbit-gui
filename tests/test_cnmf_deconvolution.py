import numpy as np

from orbit.cnmf_deconvolution import constrained_oasis_ar1, estimate_ar1_coefficient, estimate_noise_std, oasis_ar1


def test_oasis_ar1_recovers_a_single_isolated_spike_exactly():
    # A hand-computable case: pure AR(1) decay from one spike at t=5, no
    # noise -- OASIS with lam=0 should recover it exactly, since the input
    # already satisfies every constraint (a trivially "optimal" solution).
    g = 0.9
    n_frames = 20
    c_true = np.zeros(n_frames)
    c_true[5:] = g ** np.arange(n_frames - 5)

    c, s = oasis_ar1(c_true, g=g, lam=0.0)

    assert np.allclose(c, c_true, atol=1e-8)
    expected_s = np.zeros(n_frames)
    expected_s[5] = 1.0
    assert np.allclose(s, expected_s, atol=1e-8)


def test_oasis_ar1_spikes_are_nonnegative_and_fit_tracks_the_data():
    # OASIS's only hard constraint is s_t >= 0 (no baseline term in this
    # minimal port, so c itself can dip slightly negative in a noisy
    # quiet period before the first spike -- that's expected, not checked
    # here).
    rng = np.random.default_rng(0)
    g = 0.85
    y = np.zeros(100)
    y[10] = 5.0
    y[50] = 3.0
    for t in range(1, 100):
        y[t] = max(y[t], g * y[t - 1])
    y += rng.normal(scale=0.05, size=100)

    c, s = oasis_ar1(y, g=g, lam=0.1)

    assert (s >= 0).all()
    assert np.std(y - c) < 0.2


def test_constrained_oasis_ar1_residual_does_not_exceed_noise_floor_by_much():
    rng = np.random.default_rng(1)
    g = 0.9
    n_frames = 300
    c_true = np.zeros(n_frames)
    for spike_t in (30, 120, 200):
        c_true[spike_t] += 2.0
    for t in range(1, n_frames):
        c_true[t] = max(c_true[t], g * c_true[t - 1])
    noise_std = 0.2
    y = c_true + rng.normal(scale=noise_std, size=n_frames)

    c, s = constrained_oasis_ar1(y, g=g, noise_std=noise_std)

    # Bisection targets the noise floor, not an exact match -- just check
    # it's in the right ballpark rather than wildly over- or under-fit.
    residual_std = np.std(y - c)
    assert residual_std < noise_std * 2.5

    # The actual point of constraining: recovers close to the 3 true
    # spikes, not the ~one-per-frame count raw AR(1) differencing of the
    # noisy data would give (regression test for a bug where the
    # bisection silently returned the unsparsified lam=0 fit whenever it
    # happened to already sit under the noise floor).
    assert (s > 1e-6).sum() <= 6


def test_estimate_ar1_coefficient_recovers_known_decay():
    rng = np.random.default_rng(2)
    g_true = 0.9
    n_frames = 2000
    x = np.zeros(n_frames)
    spikes = rng.random(n_frames) < 0.02
    for t in range(1, n_frames):
        x[t] = g_true * x[t - 1] + spikes[t]

    g_hat = estimate_ar1_coefficient(x)

    assert abs(g_hat - g_true) < 0.05


def test_estimate_noise_std_scales_with_injected_noise():
    rng = np.random.default_rng(3)
    trace = rng.normal(scale=2.0, size=2000)

    sn = estimate_noise_std(trace)

    assert 1.5 < sn < 2.5
