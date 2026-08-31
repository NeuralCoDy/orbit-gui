import warnings

import numpy as np

from orbit.roi_extraction_realseudo import RealSeudoResult, real_seudo_source_extraction
from orbitapp.io import load_movie


def _single_blob_movie(height=30, width=30, n_frames=60, onset=10, amplitude=5.0, center=(15, 15), seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    blob = np.exp(-((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * 2.0 ** 2))
    blob /= blob.max()

    activity = np.zeros(n_frames)
    activity[onset:] = amplitude

    movie = np.zeros((height, width, n_frames), dtype=np.float32)
    for t in range(n_frames):
        movie[:, :, t] = blob * activity[t] + rng.normal(scale=0.05, size=(height, width))
    return movie


def test_real_seudo_source_extraction_recovers_a_synthetic_blob():
    movie = _single_blob_movie()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # lookahead_frames default warning, expected
        result = real_seudo_source_extraction(movie, min_roi_size=5, consecutive_frames_required=3)

    assert isinstance(result, RealSeudoResult)
    assert len(result.masks) == 1
    assert len(result.traces) == 1

    mask = result.masks[0]
    assert mask.dtype == bool
    ys, xs = np.nonzero(mask)
    assert abs(ys.mean() - 15) < 3
    assert abs(xs.mean() - 15) < 3

    trace = result.traces[0]
    assert trace.shape == (movie.shape[-1],)
    # the trace is the masked-mean of the RAW movie (same convention as
    # every other extraction method) -- should read near-zero before onset
    # and clearly elevated after
    assert trace[:10].mean() < trace[20:].mean()


def test_real_seudo_source_extraction_progress_callback_reports_every_frame():
    movie = _single_blob_movie(n_frames=20)
    calls = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        real_seudo_source_extraction(movie, min_roi_size=5, progress_callback=lambda done, total: calls.append((done, total)))

    assert calls == [(t, 20) for t in range(1, 21)]


def test_real_seudo_source_extraction_works_unmodified_against_a_memmap_movie(tmp_path):
    # The method's own key differentiator vs. every other Source Extraction
    # method here: no preview cap, no patch-based mode needed -- it reads
    # and fits one frame at a time regardless of the backing storage.
    movie = _single_blob_movie(n_frames=40)
    path = tmp_path / "movie.npy"
    np.save(path, movie)
    mmap_movie = load_movie(path, mmap=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = real_seudo_source_extraction(mmap_movie, min_roi_size=5, consecutive_frames_required=3)

    assert len(result.masks) == 1
    ys, xs = np.nonzero(result.masks[0])
    assert abs(ys.mean() - 15) < 3
    assert abs(xs.mean() - 15) < 3
