import numpy as np

from orbit._merge import find_merge_groups


def test_find_merge_groups_combines_an_overlapping_correlated_pair():
    masks = [np.zeros((10, 10), dtype=bool), np.zeros((10, 10), dtype=bool)]
    masks[0][2:6, 2:6] = True
    masks[1][4:8, 4:8] = True  # overlaps masks[0] in the [4:6, 4:6] corner
    rng = np.random.default_rng(0)
    base = rng.standard_normal(50)
    traces = [base, base + rng.standard_normal(50) * 0.01]  # near-identical

    groups = find_merge_groups(masks, traces, merge_thresh=0.8)

    assert sorted(groups, key=len) == [[0, 1]] or groups == [[0, 1]]


def test_find_merge_groups_keeps_non_overlapping_masks_separate():
    masks = [np.zeros((10, 10), dtype=bool), np.zeros((10, 10), dtype=bool)]
    masks[0][0:3, 0:3] = True
    masks[1][7:10, 7:10] = True  # no overlap at all
    rng = np.random.default_rng(1)
    base = rng.standard_normal(50)
    traces = [base, base.copy()]  # perfectly correlated, but that alone isn't enough

    groups = find_merge_groups(masks, traces, merge_thresh=0.8)

    assert sorted(len(g) for g in groups) == [1, 1]


def test_find_merge_groups_keeps_overlapping_but_uncorrelated_masks_separate():
    masks = [np.zeros((10, 10), dtype=bool), np.zeros((10, 10), dtype=bool)]
    masks[0][2:6, 2:6] = True
    masks[1][4:8, 4:8] = True  # overlaps, but...
    rng = np.random.default_rng(2)
    traces = [rng.standard_normal(50), rng.standard_normal(50)]  # ...uncorrelated

    groups = find_merge_groups(masks, traces, merge_thresh=0.8)

    assert sorted(len(g) for g in groups) == [1, 1]


def test_find_merge_groups_is_transitive_across_three_components():
    # A overlaps B, B overlaps C, A does NOT overlap C directly -- all
    # three should still land in one group via B as the connecting link.
    masks = [np.zeros((12, 12), dtype=bool) for _ in range(3)]
    masks[0][0:4, 0:4] = True
    masks[1][2:6, 2:6] = True  # overlaps masks[0]
    masks[2][4:8, 4:8] = True  # overlaps masks[1], not masks[0]
    rng = np.random.default_rng(3)
    base = rng.standard_normal(50)
    traces = [base + rng.standard_normal(50) * 0.001 for _ in range(3)]

    groups = find_merge_groups(masks, traces, merge_thresh=0.8)

    assert len(groups) == 1
    assert sorted(groups[0]) == [0, 1, 2]


def test_find_merge_groups_handles_a_single_component():
    masks = [np.ones((5, 5), dtype=bool)]
    traces = [np.zeros(20)]
    groups = find_merge_groups(masks, traces, merge_thresh=0.8)
    assert groups == [[0]]
