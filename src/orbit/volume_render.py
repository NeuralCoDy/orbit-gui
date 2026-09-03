"""CPU volume rendering for a (Z, Y, X) intensity volume -- no OpenGL, so
it works headless and over X-forwarding. The Data Projections tab uses
this to show a volumetric movie's time projection as a drag-rotatable 3D
view (see orbitapp.widgets.volume_view).

Two modes, both after rotating the volume into view:

- ``"mip"``: maximum-intensity projection along the view ray.
- ``"composite"``: emission-absorption compositing -- each voxel's
  opacity rises with its intensity (``intensity**gamma * density``), so
  near-zero voxels are transparent and bright structure is opaque.

``render_volume`` returns a scalar (n, n) float image in [0, 1]; the
caller applies a colormap (the core stays matplotlib-free -- that's an
optional dependency).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import affine_transform


def downsample_volume(vol: np.ndarray, target_max_axis: int = 200) -> np.ndarray:
    """Mean-pool ``vol`` by an integer factor per axis so its longest
    axis is about ``target_max_axis`` -- rendering cost scales with the
    cube of the working size, so a multi-hundred-voxel axis has to come
    down first. Returns float32; a no-op copy when already small."""
    factors = [max(1, round(n / target_max_axis)) for n in vol.shape]
    if all(f == 1 for f in factors):
        return np.asarray(vol, dtype=np.float32)
    z, y, x = (n // f * f for n, f in zip(vol.shape, factors))
    fz, fy, fx = factors
    trimmed = np.asarray(vol[:z, :y, :x], dtype=np.float32)
    return trimmed.reshape(z // fz, fz, y // fy, fy, x // fx, fx).mean(axis=(1, 3, 5))


def normalize_volume(vol: np.ndarray, high_percentile: float = 99.5) -> np.ndarray:
    """Scale ``vol`` to [0, 1] by its ``high_percentile`` value (clipping
    the brightest tail), so the opacity transfer function has a stable
    input range regardless of the movie's native dtype/scaling."""
    vol = np.ascontiguousarray(vol, dtype=np.float32)
    hi = float(np.percentile(vol, high_percentile))
    return np.clip(vol / max(hi, 1e-6), 0.0, 1.0)


def _rotation_matrix(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = np.radians(az_deg), np.radians(el_deg)
    rz = np.array([[np.cos(az), -np.sin(az), 0.0], [np.sin(az), np.cos(az), 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(el), -np.sin(el)], [0.0, np.sin(el), np.cos(el)]])
    return rx @ rz


def render_volume(
    vol01: np.ndarray,
    az_deg: float,
    el_deg: float,
    *,
    gamma: float = 1.6,
    density: float = 3.0,
    mode: str = "composite",
    order: int = 1,
    box_scale: float = 1.15,
) -> np.ndarray:
    """``vol01``: (Z, Y, X) float in [0, 1]. Rotate it by (az, el) into a
    cube and project along the (new) Z axis. Returns an (n, n) float
    image in [0, 1].

    ``box_scale`` sizes the cube relative to the longest volume axis;
    >~1.7 fits every rotation with no corner clipping but costs more,
    the default trades a little clipping at extreme angles for speed.
    ``order=0`` (nearest) is the fast path for a drag preview; ``order=1``
    (trilinear) for the settled render."""
    n = int(round(box_scale * max(vol01.shape)))
    center_in = 0.5 * (np.asarray(vol01.shape) - 1.0)
    center_out = 0.5 * (n - 1.0)
    matrix = _rotation_matrix(az_deg, el_deg)
    offset = center_in - matrix @ np.full(3, center_out)
    resampled = np.clip(
        affine_transform(vol01, matrix, offset=offset, output_shape=(n, n, n), order=order, mode="constant", cval=0.0),
        0.0, 1.0,
    )

    if mode == "mip":
        proj = resampled.max(axis=0)
    else:  # emission-absorption, vectorised front-to-back over Z
        alpha = np.clip(resampled**gamma * density, 0.0, 1.0)
        transmittance = np.empty_like(alpha)
        transmittance[0] = 1.0
        np.cumprod(1.0 - alpha[:-1], axis=0, out=transmittance[1:])
        proj = (transmittance * alpha * resampled).sum(axis=0)

    return proj / max(float(proj.max()), 1e-6)
