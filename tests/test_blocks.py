import tracemalloc

import numpy as np
import pytest

from orbit._blocks import chunked_median, chunked_variance, iter_axis_slices


def test_iter_axis_slices_covers_the_whole_axis_without_gaps_or_overlap():
    slices = list(iter_axis_slices((10, 4), 0, target_bytes=64, itemsize=8))
    covered = np.zeros(10, dtype=bool)
    for sl in slices:
        assert not covered[sl].any(), "overlapping slices"
        covered[sl] = True
    assert covered.all()


def test_iter_axis_slices_max_block_caps_the_byte_budget_derived_step():
    # target_bytes alone would allow a huge step here; max_block must win.
    slices = list(iter_axis_slices((1000, 2), 0, target_bytes=10**9, itemsize=8, max_block=7))
    assert all(sl.stop - sl.start <= 7 for sl in slices)
    assert sum(sl.stop - sl.start for sl in slices) == 1000


def test_iter_axis_slices_byte_budget_still_applies_under_a_generous_max_block():
    # the reverse: a small byte budget must still win even with a large max_block cap
    # -- other_elems=100_000 * itemsize=8 already exceeds target_bytes=1024, so the
    # budget floors the step to 1, well under the generous max_block=500 cap.
    slices = list(iter_axis_slices((1000, 100_000), 0, target_bytes=1024, itemsize=8, max_block=500))
    assert all(sl.stop - sl.start == 1 for sl in slices)
    assert len(slices) == 1000


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_chunked_median_matches_plain_numpy_across_a_chunk_boundary(dtype):
    rng = np.random.default_rng(0)
    H = 340  # spans multiple default (150-pixel) blocks
    arr = (rng.random((H, 3, 11)) * 200).astype(dtype)

    got = chunked_median(arr, axis=2)
    want = np.median(arr, axis=2)

    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want)


def test_chunked_median_keepdims_matches_numpy_shape_and_values():
    rng = np.random.default_rng(1)
    arr = rng.random((340, 3, 11))

    got = chunked_median(arr, axis=2, keepdims=True)
    want = np.median(arr, axis=2, keepdims=True)

    assert got.shape == want.shape
    np.testing.assert_array_equal(got, want)


def test_chunked_median_axis_none_falls_through_to_plain_np_median():
    rng = np.random.default_rng(6)
    arr = rng.random((40, 3, 11))

    assert chunked_median(arr) == np.median(arr)
    np.testing.assert_array_equal(
        chunked_median(arr, keepdims=True), np.median(arr, keepdims=True)
    )


def test_chunked_median_works_on_a_non_zero_reduction_axis():
    # volumetric convention: reduce axis 0 (time), chunk axis 1 (first spatial axis)
    rng = np.random.default_rng(2)
    vol = (rng.random((6, 340, 3, 2)) * 200).astype(np.uint16)

    np.testing.assert_array_equal(chunked_median(vol, axis=0), np.median(vol, axis=0))


def test_chunked_median_out_dtype_override_forces_that_dtype():
    rng = np.random.default_rng(3)
    arr = (rng.random((340, 3, 11)) * 200).astype(np.uint8)

    got = chunked_median(arr, axis=2, out_dtype=np.float32)

    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, np.median(arr.astype(np.float32), axis=2))


def test_chunked_median_max_block_none_still_bounded_by_the_byte_budget():
    # motion_correction passes max_block=None -- the byte budget alone must still chunk.
    rng = np.random.default_rng(7)
    arr = (rng.random((500, 4, 4)) * 200).astype(np.float32)
    np.testing.assert_array_equal(
        chunked_median(arr, axis=2, max_block=None), np.median(arr, axis=2)
    )


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_chunked_variance_matches_plain_numpy_across_a_chunk_boundary(dtype):
    rng = np.random.default_rng(8)
    arr = (rng.random((340, 3, 11)) * 200).astype(dtype)

    got = chunked_variance(arr, axis=2)
    want = arr.var(axis=2)

    assert got.dtype == want.dtype
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_chunked_median_does_not_mutate_the_input():
    rng = np.random.default_rng(4)
    arr = (rng.random((340, 3, 11)) * 200).astype(np.uint8)
    original = arr.copy()

    chunked_median(arr, axis=2)

    np.testing.assert_array_equal(arr, original)


def test_chunked_median_peak_memory_stays_near_one_block_not_the_whole_array():
    rng = np.random.default_rng(5)
    H, W, T = 150 * 40, 8, 25
    arr = (rng.random((H, W, T)) * 200).astype(np.float32)
    whole_array_partition_bytes = arr.nbytes  # what an unchunked np.median would copy

    tracemalloc.start()
    chunked_median(arr, axis=2)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < whole_array_partition_bytes / 10
