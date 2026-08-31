import numpy as np
import pytest

from orbit.seudo.streaming import (
    DetectionParams,
    FitParams,
    PromotionParams,
    StreamingState,
    _should_merge_temp_profiles,
    realSEUDOfit,
)


def _single_blob_movie(mov_y=30, mov_x=30, n_frames=60, onset=10, amplitude=5.0, center=(15, 15), seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:mov_y, 0:mov_x]
    blob = np.exp(-((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * 2.0 ** 2))
    blob /= blob.max()

    activity = np.zeros(n_frames)
    activity[onset:] = amplitude

    movie = np.zeros((mov_y, mov_x, n_frames))
    for t in range(n_frames):
        movie[:, :, t] = blob * activity[t] + rng.normal(scale=0.05, size=(mov_y, mov_x))
    return movie, blob


def test_streaming_state_construction_starts_with_no_known_cells():
    state = StreamingState((30, 30))
    assert state.profiles.shape == (30, 30, 0)
    assert state.candidate_tracks == {}
    assert state.frame_index == 0


def test_realseudofit_discovers_a_synthetic_cell_and_recovers_its_location():
    movie, blob = _single_blob_movie()
    state = StreamingState(
        movie.shape[:2],
        fit=FitParams(sigma2=0.01, lambda_blob=5.0, blob_radius=2.0, pad_space=5, lookahead_frames=1),
        detection=DetectionParams(min_roi_size=5, cutoff_multiplier=3.0),
        promotion=PromotionParams(consecutive_frames_required=3),
    )
    new_cell_frames = []
    for t in range(movie.shape[-1]):
        result = realSEUDOfit(movie[:, :, t], state)
        if result.new_cells:
            new_cell_frames.append(result.frame_index)

    assert state.profiles.shape[2] == 1
    assert new_cell_frames  # promoted at some point after the cell turned on at frame 10
    assert new_cell_frames[0] >= 10

    discovered_mask = state.profiles[:, :, 0] > 0
    ys, xs = np.nonzero(discovered_mask)
    assert abs(ys.mean() - 15) < 3
    assert abs(xs.mean() - 15) < 3
    state.close()


def test_lookahead_frames_returns_none_during_warmup_then_reports_a_lagged_frame_index():
    movie, _blob = _single_blob_movie()
    with pytest.warns(UserWarning, match="lookahead"):
        state = StreamingState(
            movie.shape[:2],
            fit=FitParams(sigma2=0.01, lambda_blob=5.0, blob_radius=2.0, lookahead_frames=3),
        )

    result0 = realSEUDOfit(movie[:, :, 0], state)
    result1 = realSEUDOfit(movie[:, :, 1], state)
    assert result0 is None
    assert result1 is None  # not enough future context yet (lookahead_frames=3 needs 3 calls)

    result2 = realSEUDOfit(movie[:, :, 2], state)
    assert result2 is not None
    assert result2.frame_index == 0  # reports on the OLDEST buffered frame, lagged by lookahead_frames-1


def test_lookahead_frames_1_never_returns_none_and_does_not_warn(recwarn):
    movie, _blob = _single_blob_movie(n_frames=5)
    state = StreamingState(movie.shape[:2], fit=FitParams(lookahead_frames=1))
    assert not any("lookahead" in str(w.message) for w in recwarn.list)

    for t in range(movie.shape[-1]):
        result = realSEUDOfit(movie[:, :, t], state)
        assert result is not None
        assert result.frame_index == t  # immediate, no lag


def test_consecutive_frames_required_gates_promotion():
    movie, _blob = _single_blob_movie(n_frames=30, onset=5)
    state = StreamingState(
        movie.shape[:2],
        fit=FitParams(sigma2=0.01, lambda_blob=5.0, blob_radius=2.0, lookahead_frames=1),
        detection=DetectionParams(min_roi_size=5, cutoff_multiplier=3.0),
        promotion=PromotionParams(consecutive_frames_required=10),  # deliberately high
    )
    for t in range(8):  # fewer frames than consecutive_frames_required allows
        realSEUDOfit(movie[:, :, t], state)
    assert state.profiles.shape[2] == 0  # not promoted yet -- hasn't hit the requirement

    for t in range(8, movie.shape[-1]):
        realSEUDOfit(movie[:, :, t], state)
    assert state.profiles.shape[2] == 1  # eventually promoted once consecutive_frames_required is met


def test_should_merge_temp_profiles_eq8_merges_a_mostly_contained_pair():
    mask_a = np.zeros((10, 10), dtype=bool)
    mask_a[2:8, 2:8] = True
    mask_b = np.zeros((10, 10), dtype=bool)
    mask_b[3:7, 3:7] = True  # fully contained within mask_a

    assert _should_merge_temp_profiles(mask_a, (0, 9, 0, 9), mask_b, (0, 9, 0, 9), k_temp=0.75)


def test_should_merge_temp_profiles_eq8_keeps_genuinely_separate_masks_apart():
    mask_a = np.zeros((10, 10), dtype=bool)
    mask_a[0:3, 0:3] = True
    mask_b = np.zeros((10, 10), dtype=bool)
    mask_b[7:10, 7:10] = True  # no overlap at all

    assert not _should_merge_temp_profiles(mask_a, (0, 9, 0, 9), mask_b, (0, 9, 0, 9), k_temp=0.75)
