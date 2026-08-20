import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from orbit.seudo import (
    SeudoData,
    auto_classify_transients,
    classification_color,
    cycle_classification,
    load_classification,
    save_classification,
)
from orbit.seudo.blob import make_seudo_blob
from orbit.seudo.constants import VAL_FALSE, VAL_MIX, VAL_TRUE
from orbit.seudo.estimate import estimate_time_courses_with_seudo
from orbit.seudo.geometry import compute_roi_coms
from orbit.seudo.run_on_transients import frame_blocks_for_cell
from orbit.seudo.solver import fista_nonneg_weighted_l1
from orbit.seudo.stats import correlation_vector_matrix, robust_std
from orbit.seudo.transients import identify_transients


def test_cycle_classification_goes_unclassified_false_true_mixed_unclassified():
    value = float("nan")
    seen = []
    for _ in range(4):
        value = cycle_classification(value)
        seen.append(value)
    assert seen[0] == VAL_FALSE
    assert seen[1] == VAL_TRUE
    assert seen[2] == VAL_MIX
    assert np.isnan(seen[3])


def test_classification_color_is_distinct_per_class():
    colors = {classification_color(v) for v in (VAL_TRUE, VAL_FALSE, VAL_MIX, float("nan"))}
    assert len(colors) == 4


def test_compute_roi_coms_centers_on_a_square_blob():
    roi = np.zeros((20, 20))
    roi[4:9, 10:15] = 1.0  # rows 4-8, cols 10-14 -> center (6, 12)
    coms, bounds = compute_roi_coms(roi)
    assert coms[0] == pytest.approx([12.0, 6.0])
    assert tuple(bounds[0]) == (4, 8, 10, 14)


def test_robust_std_matches_ordinary_std_for_gaussian_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(20000) * 2.0
    assert robust_std(data) == pytest.approx(2.0, rel=0.05)


def test_correlation_vector_matrix_recovers_known_correlations():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(200)
    same = v.copy()
    opposite = -v
    unrelated = rng.standard_normal(200)
    m = np.column_stack([same, opposite, unrelated])
    corrs = correlation_vector_matrix(v, m)
    assert corrs[0] == pytest.approx(1.0)
    assert corrs[1] == pytest.approx(-1.0)
    assert abs(corrs[2]) < 0.3


def test_make_seudo_blob_is_normalized_and_peaks_at_center():
    blob = make_seudo_blob(blob_radius=2.0)
    assert blob.shape[0] == blob.shape[1]
    assert blob.shape[0] % 2 == 1
    center = blob.shape[0] // 2
    assert blob[center, center] == blob.max()
    assert np.sum(blob**2) == pytest.approx(1.0)


def test_fista_nonneg_weighted_l1_recovers_a_sparse_nonneg_solution():
    rng = np.random.default_rng(0)
    n = 10
    A_mat = rng.standard_normal((20, n))

    def A(x):
        return A_mat @ x

    def At(v):
        return A_mat.T @ v

    x_true = np.zeros(n)
    x_true[[2, 5]] = [3.0, 1.5]
    b = A(x_true)
    lam = np.full(n, 0.01)  # tiny penalty -- recover x_true closely

    x_hat = fista_nonneg_weighted_l1(A, At, b, lam, np.zeros(n), tol=1e-6, max_iter=2000)

    assert (x_hat >= -1e-8).all()
    assert np.allclose(x_hat, x_true, atol=0.05)


def _overlapping_cells_scenario(height=30, width=30, n_frames=200, seed=0):
    """Two overlapping cells: cell 0 has its own real transient plus a
    "false transient" caused by cell 1's activity bleeding into its
    plain masked-mean trace through the overlap region; cell 1 has only
    its own real transient. Boolean masks are Gaussian-blurred into
    continuous profiles -- required for the correlation classifier to be
    well-defined (a hard-edged binary mask, cropped to its own support,
    has zero within-support variance)."""
    rng = np.random.default_rng(seed)
    mask_a = np.zeros((height, width))
    mask_a[6:14, 6:14] = 1.0
    mask_b = np.zeros((height, width))
    mask_b[10:18, 10:18] = 1.0
    profile_a = gaussian_filter(mask_a, sigma=1.5)
    profile_b = gaussian_filter(mask_b, sigma=1.5)
    profiles = np.stack([profile_a, profile_b], axis=2)

    trace_a_true = np.zeros(n_frames)
    trace_a_true[50:55] = 5.0
    trace_b_true = np.zeros(n_frames)
    trace_b_true[120:130] = 8.0

    movie = rng.standard_normal((height, width, n_frames)) * 0.05
    movie += mask_a[:, :, None] * trace_a_true[None, None, :]
    movie += mask_b[:, :, None] * trace_b_true[None, None, :]

    tc_a = (movie * mask_a[:, :, None]).sum(axis=(0, 1)) / mask_a.sum()
    tc_b = (movie * mask_b[:, :, None]).sum(axis=(0, 1)) / mask_b.sum()
    time_courses = np.stack([tc_a, tc_b], axis=1)

    return SeudoData(movie, profiles, time_courses=time_courses)


def test_identify_transients_finds_an_injected_event():
    n_frames = 300
    tc = np.zeros((n_frames, 1))
    tc[100:110, 0] = 10.0
    found = identify_transients(tc, min_duration=3)
    assert found[100:110, 0].all()
    assert not found[:90, 0].any()
    assert not found[120:, 0].any()


def test_seudo_data_rejects_mismatched_profile_shape():
    movie = np.zeros((10, 10, 5))
    bad_profiles = np.zeros((8, 8, 2))
    with pytest.raises(ValueError, match="does not match"):
        SeudoData(movie, bad_profiles)


def test_seudo_data_resolve_tc_struct_requires_default_tc():
    se = SeudoData(np.zeros((10, 10, 5)), np.zeros((10, 10, 1)))
    with pytest.raises(ValueError, match="tc_default"):
        se._resolve_tc_struct("default")


def test_compute_transient_info_finds_two_transients_for_the_contaminated_cell():
    se = _overlapping_cells_scenario()
    se.compute_transient_info("default")

    ti0 = se.tc_default["transient_info"][0]
    ti1 = se.tc_default["transient_info"][1]
    assert ti0["times"].shape[0] == 2  # its own event + the bleed-in from cell 1
    assert ti1["times"].shape[0] == 1
    assert ti0["shapes"] is not None
    assert ti0["shapes"].shape[2] == 2


def test_auto_classify_distinguishes_real_from_contaminated_transient():
    # Regression test for the core SEUDO use case: cell 0's own transient
    # should classify true, its contaminated (bleed-in from cell 1)
    # transient should classify false.
    se = _overlapping_cells_scenario()
    se.compute_transient_info("default")

    results = auto_classify_transients(se, "default", save_results=False)

    own_event_corr, bleed_in_corr = results[0]["corrs"]
    assert own_event_corr > 0.8
    assert bleed_in_corr < 0.5
    assert results[0]["classification"][0] == VAL_TRUE
    assert results[0]["classification"][1] == VAL_FALSE
    assert results[1]["classification"][0] == VAL_TRUE


def test_frame_blocks_for_cell_matches_transient_times():
    se = _overlapping_cells_scenario()
    se.compute_transient_info("default")
    blocks = frame_blocks_for_cell(se.tc_default["transient_info"], 1)
    assert blocks == [tuple(row) for row in se.tc_default["transient_info"][1]["times"].tolist()]


def test_estimate_time_courses_with_seudo_runs_on_a_restricted_frame_block():
    se = _overlapping_cells_scenario(height=20, width=20, n_frames=60)
    result = estimate_time_courses_with_seudo(
        se.movie, se.profiles, which_cells=[0], zero_level=0.0, frame_blocks=[(10, 20)],
    )
    tc = result["tc"][:, 0]
    assert np.isnan(tc[:10]).all()
    assert not np.isnan(tc[10:21]).any()
    assert np.isnan(tc[21:]).all()


def test_save_and_load_classification_round_trips(tmp_path):
    se = _overlapping_cells_scenario()
    se.compute_transient_info("default")
    se.tc_default["transient_info"][0]["classification"][:] = VAL_TRUE

    path = tmp_path / "classification.pkl"
    save_classification(se.tc_default, path)
    loaded = load_classification(path)

    assert np.array_equal(loaded["tc"], se.tc_default["tc"])
    assert loaded["transient_info"][0]["classification"][0] == VAL_TRUE
