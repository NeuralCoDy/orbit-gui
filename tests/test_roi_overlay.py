import numpy as np

from orbitapp.state import ROI
from orbitapp.widgets.roi_overlay import render_roi_overlay


def _make_roi(roi_id, mask, status="pending", source_method="correlation"):
    return ROI(id=roi_id, mask=mask, trace=np.zeros(5), source_method=source_method, status=status)


def test_render_roi_overlay_returns_none_for_no_rois():
    assert render_roi_overlay([], (4, 4)) is None


def test_render_roi_overlay_shape_and_dtype():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    overlay = render_roi_overlay([_make_roi(1, mask)], (4, 4))

    assert overlay.shape == (4, 4, 4)
    assert overlay.dtype == np.uint8


def test_render_roi_overlay_alpha_zero_outside_every_mask():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    overlay = render_roi_overlay([_make_roi(1, mask)], (4, 4))

    assert overlay[3, 3, 3] == 0
    assert overlay[0, 0, 3] > 0


def test_render_roi_overlay_accepted_more_opaque_than_rejected():
    mask = np.ones((2, 2), dtype=bool)
    accepted_overlay = render_roi_overlay([_make_roi(1, mask, status="accepted")], (2, 2))
    rejected_overlay = render_roi_overlay([_make_roi(1, mask, status="rejected")], (2, 2))

    assert accepted_overlay[0, 0, 3] > rejected_overlay[0, 0, 3]


def test_render_roi_overlay_distinct_colors_for_different_rois():
    mask_a = np.zeros((4, 4), dtype=bool)
    mask_a[0, 0] = True
    mask_b = np.zeros((4, 4), dtype=bool)
    mask_b[3, 3] = True

    overlay = render_roi_overlay([_make_roi(1, mask_a), _make_roi(2, mask_b)], (4, 4))

    assert not np.array_equal(overlay[0, 0, :3], overlay[3, 3, :3])
