"""_phase_correlate_shift is a leaner reimplementation of skimage's
phase_cross_correlation (same Guizar-Sicairos algorithm) -- these tests
pin it to skimage's own output across the configurations orbit actually
uses, plus its two memory-bounding helpers in isolation.
"""

import numpy as np
import pytest
from scipy.ndimage import fourier_shift

from orbit.motion_correction import _argmax_magnitude, _phase_correlate_shift


def _shifted_pair(shape, true_shift, seed):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(shape).astype(np.float32) * 100 + 500
    moving = np.real(np.fft.ifftn(fourier_shift(np.fft.fftn(base), true_shift))).astype(np.float32)
    return base, moving


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("upsample_factor", [1, 5, 20, 50])
@pytest.mark.parametrize("normalization", [None, "phase"])
def test_matches_skimage_phase_cross_correlation(ndim, upsample_factor, normalization):
    skimage = pytest.importorskip("skimage.registration")

    rng = np.random.default_rng(0)
    shape = tuple(int(x) for x in rng.integers(24, 70, size=ndim))
    true_shift = rng.uniform(-5, 5, size=ndim).astype(np.float32)
    reference, moving = _shifted_pair(shape, true_shift, seed=1)

    expected, _error, _phasediff = skimage.phase_cross_correlation(
        reference, moving, upsample_factor=upsample_factor, normalization=normalization
    )
    got = _phase_correlate_shift(reference.copy(), moving.copy(), upsample_factor, normalization)

    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_matches_skimage_across_many_random_configs():
    # Broader sweep than the parametrized grid above, one assertion --
    # catches a config-specific edge case the fixed grid might miss.
    skimage = pytest.importorskip("skimage.registration")
    rng = np.random.default_rng(7)
    for _ in range(40):
        ndim = int(rng.choice([2, 3]))
        shape = tuple(int(x) for x in rng.integers(15, 55, size=ndim))
        upsample_factor = int(rng.choice([1, 3, 10, 20, 50]))
        normalization = rng.choice([None, "phase"])
        true_shift = rng.uniform(-6, 6, size=ndim).astype(np.float32)
        reference, moving = _shifted_pair(shape, true_shift, seed=int(rng.integers(0, 1_000_000)))

        expected, _e, _p = skimage.phase_cross_correlation(
            reference, moving, upsample_factor=upsample_factor, normalization=normalization
        )
        got = _phase_correlate_shift(reference.copy(), moving.copy(), upsample_factor, normalization)
        np.testing.assert_allclose(got, expected, atol=1e-4, err_msg=f"{shape=} {upsample_factor=} {normalization=}")


def test_recovers_a_known_shift_directly():
    reference, moving = _shifted_pair((60, 80, 40), np.array([2.4, -3.1, 1.2]), seed=2)
    shift = _phase_correlate_shift(reference, moving, upsample_factor=20, normalization=None)
    np.testing.assert_allclose(shift, [-2.4, 3.1, -1.2], atol=0.05)


def test_rejects_an_unknown_normalization():
    reference, moving = _shifted_pair((20, 20), np.array([1.0, 0.0]), seed=3)
    with pytest.raises(ValueError, match="normalization"):
        _phase_correlate_shift(reference, moving, upsample_factor=10, normalization="bogus")


def test_does_not_mutate_its_inputs():
    reference, moving = _shifted_pair((20, 24, 16), np.array([1.0, -0.5, 0.2]), seed=4)
    ref_before, mov_before = reference.copy(), moving.copy()
    _phase_correlate_shift(reference, moving, upsample_factor=10, normalization=None)
    np.testing.assert_array_equal(reference, ref_before)
    np.testing.assert_array_equal(moving, mov_before)


def test_peak_memory_stays_within_a_few_whole_array_buffers():
    # Regression guard for the whole point of this reimplementation:
    # skimage's phase_cross_correlation measured ~10x one array's size at
    # a real volume's shape; this version measured ~4x. Traced here at a
    # smaller (but still not tiny) size so the test is fast -- bounded
    # loosely (6x), to catch a regression back toward skimage's ~10x
    # without being sensitive to allocator noise at this size.
    import tracemalloc

    rng = np.random.default_rng(0)
    a = rng.random((80, 120, 60)).astype(np.float32)
    b = rng.random((80, 120, 60)).astype(np.float32)

    tracemalloc.start()
    try:
        _phase_correlate_shift(a, b, upsample_factor=20, normalization=None)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    ratio = peak / a.nbytes
    assert ratio < 6.0, f"_phase_correlate_shift peak was {ratio:.1f}x one array's size (expected well under 6x)"


class TestArgmaxMagnitude:
    def test_matches_numpy_for_real_and_complex_arrays(self):
        rng = np.random.default_rng(5)
        for shape in [(7,), (8, 9), (5, 6, 7), (3, 4, 5, 2)]:
            real = rng.standard_normal(shape)
            complex_arr = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
            for arr in (real, complex_arr):
                expected = np.unravel_index(np.argmax(np.abs(arr)), arr.shape)
                assert _argmax_magnitude(arr) == expected

    def test_finds_a_planted_maximum(self):
        arr = np.zeros((10, 12, 8), dtype=np.complex64)
        arr[3, 7, 2] = 5 + 5j  # |.| = 5*sqrt(2), the largest magnitude here
        assert _argmax_magnitude(arr) == (3, 7, 2)
