import numpy as np

from orbit.neuropil import compute_neuropil_traces, neuropil_ring_mask


class _StubROI:
    def __init__(self, mask):
        self.mask = mask
        self.neuropil_trace = None


def test_neuropil_ring_mask_is_a_ring_around_the_roi_not_overlapping_it():
    mask = np.zeros((30, 30), dtype=bool)
    mask[15, 15] = True

    ring = neuropil_ring_mask(mask, inner_radius=2, outer_radius=4)

    assert ring.any()
    assert not np.any(ring & mask)  # ring never overlaps the ROI's own pixels
    # ring pixels should all be within outer_radius of the seed, none within inner_radius
    rows, cols = np.nonzero(ring)
    dist = np.hypot(rows - 15, cols - 15)
    assert dist.min() > 2
    assert dist.max() <= 4 + 1e-9


def test_neuropil_ring_mask_excludes_given_pixels():
    mask = np.zeros((30, 30), dtype=bool)
    mask[15, 15] = True
    ring_unrestricted = neuropil_ring_mask(mask, inner_radius=2, outer_radius=4)

    exclusion = np.zeros((30, 30), dtype=bool)
    exclusion[13:18, 13:18] = True  # covers part of the ring

    ring_excluded = neuropil_ring_mask(mask, inner_radius=2, outer_radius=4, exclusion_mask=exclusion)

    assert ring_excluded.sum() < ring_unrestricted.sum()
    assert not np.any(ring_excluded & exclusion)


def test_compute_neuropil_traces_excludes_pixels_claimed_by_other_rois():
    height, width, n_frames = 30, 30, 10
    movie = np.zeros((height, width, n_frames))
    movie[:, :, :] = 1.0  # background trace value everywhere
    # neighbor ROI right next to roi_a -- squarely inside roi_a's ring --
    # gets a very different trace value so the exclusion is easy to detect.
    movie[16, 16, :] = 100.0

    mask_a = np.zeros((height, width), dtype=bool)
    mask_a[15, 15] = True
    mask_b = np.zeros((height, width), dtype=bool)
    mask_b[16, 16] = True

    roi_a = _StubROI(mask_a)
    roi_b = _StubROI(mask_b)
    compute_neuropil_traces(movie, [roi_a, roi_b], inner_radius=0, outer_radius=2)

    # roi_a's ring would otherwise include (16,16), which is roi_b's own
    # pixel -- excluded, so roi_a's neuropil trace stays at the background
    # value rather than being pulled toward 100.
    assert roi_a.neuropil_trace is not None
    assert np.allclose(roi_a.neuropil_trace, 1.0)


def test_compute_neuropil_traces_is_a_noop_for_empty_list():
    compute_neuropil_traces(np.zeros((5, 5, 3)), [])  # must not raise
