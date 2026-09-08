import numpy as np
import pytest

from orbit.volume_render import downsample_volume, normalize_volume, render_volume


def _blob_volume(shape=(60, 50, 70)) -> np.ndarray:
    v = np.zeros(shape, np.float32)
    z, y, x = (s // 2 for s in shape)
    v[z - 5 : z + 5, y - 5 : y + 5, x - 5 : x + 5] = 1.0
    return v


def test_downsample_hits_target_axis_and_is_float32():
    vol = np.random.default_rng(0).random((400, 100, 800)).astype(np.float32)
    ds = downsample_volume(vol, target_max_axis=200)
    assert ds.dtype == np.float32
    assert max(ds.shape) <= 220  # ~200, allowing the rounding of one factor
    # mean-pool, not subsample: the downsampled mean matches the original's
    np.testing.assert_allclose(ds.mean(), vol.mean(), rtol=1e-3)


def test_downsample_is_a_noop_when_already_small():
    vol = np.random.default_rng(1).random((30, 20, 25)).astype(np.float64)
    ds = downsample_volume(vol, target_max_axis=200)
    assert ds.shape == vol.shape
    np.testing.assert_allclose(ds, vol.astype(np.float32))


def test_normalize_clips_to_unit_range_at_the_percentile():
    vol = np.concatenate([np.zeros(990), np.full(10, 1000.0)]).reshape(10, 10, 10)
    n = normalize_volume(vol, high_percentile=95.0)
    assert n.min() == 0.0 and n.max() == 1.0
    assert n.dtype == np.float32


def test_render_volume_returns_unit_range_square_image():
    v = normalize_volume(_blob_volume())
    for mode in ("composite", "mip"):
        img = render_volume(v, az_deg=30, el_deg=20, mode=mode)
        assert img.ndim == 2 and img.shape[0] == img.shape[1]
        assert 0.0 <= img.min() and img.max() <= 1.0 + 1e-6
        assert img.max() > 0.0  # the blob shows up


def test_render_volume_all_zero_volume_is_all_zero_image():
    img = render_volume(np.zeros((20, 20, 20), np.float32), az_deg=15, el_deg=-40, mode="composite")
    assert np.all(img == 0.0)


def test_render_volume_rotation_changes_the_image():
    v = normalize_volume(_blob_volume((50, 50, 90)))  # elongated along X so rotation is visible
    a = render_volume(v, az_deg=0, el_deg=0, mode="mip")
    b = render_volume(v, az_deg=70, el_deg=0, mode="mip")
    assert not np.allclose(a, b, atol=1e-3)


def test_render_volume_order0_and_order1_agree_roughly():
    v = normalize_volume(_blob_volume())
    fast = render_volume(v, az_deg=25, el_deg=15, mode="composite", order=0)
    fine = render_volume(v, az_deg=25, el_deg=15, mode="composite", order=1)
    assert fast.shape == fine.shape
    assert np.corrcoef(fast.ravel(), fine.ravel())[0, 1] > 0.95


def test_composite_gamma_controls_transparency_of_dim_voxels():
    # A dim shell around a bright core. Higher gamma pushes the dim
    # shell further toward transparent, so less of the pre-normalization
    # signal comes from it -- the rendered image concentrates more on
    # the bright core (higher peak-to-total ratio).
    v = np.full((40, 40, 40), 0.15, np.float32)
    v[15:25, 15:25, 15:25] = 1.0
    low_g = render_volume(v, az_deg=15, el_deg=15, mode="composite", gamma=0.5)
    high_g = render_volume(v, az_deg=15, el_deg=15, mode="composite", gamma=3.0)
    assert (high_g > 0.5).mean() < (low_g > 0.5).mean()
