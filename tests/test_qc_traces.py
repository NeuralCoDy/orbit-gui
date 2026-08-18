import numpy as np

from orbit.qc_traces import qc_trace_samples


def test_qc_trace_samples_finds_separated_peak_and_low_correlation_locations():
    height, width, n_frames = 40, 40, 60
    rng = np.random.default_rng(7)
    movie = rng.standard_normal((height, width, n_frames)) * 0.1
    # Two well-separated correlated "cell" blobs, each internally
    # coherent, against uncorrelated background noise -- an unambiguous
    # local-correlation peak in each blob, far apart in the FOV.
    movie[5:10, 5:10, :] += rng.standard_normal(n_frames)
    movie[30:35, 30:35, :] += rng.standard_normal(n_frames)
    denoised = movie * 0.5  # arbitrary stand-in for "after"

    samples = qc_trace_samples(movie, denoised, n_peaks=2, n_low=2, min_separation_frac=0.2)

    assert len(samples) == 4
    kinds = [s["kind"] for s in samples]
    assert kinds.count("peak") == 2
    assert kinds.count("low") == 2
    for sample in samples:
        assert sample["before"].shape == (n_frames,)
        assert sample["after"].shape == (n_frames,)

    min_dist = 0.2 * max(height, width)
    peak_locs = [(s["row"], s["col"]) for s in samples if s["kind"] == "peak"]
    assert np.hypot(peak_locs[0][0] - peak_locs[1][0], peak_locs[0][1] - peak_locs[1][1]) >= min_dist

    peak_corrs = [s["corr"] for s in samples if s["kind"] == "peak"]
    low_corrs = [s["corr"] for s in samples if s["kind"] == "low"]
    assert min(peak_corrs) > max(low_corrs)


def test_qc_trace_samples_traces_match_the_source_movies():
    height, width, n_frames = 20, 20, 15
    rng = np.random.default_rng(8)
    movie = rng.standard_normal((height, width, n_frames))
    denoised = movie + 1.0  # distinguishable from "before"

    samples = qc_trace_samples(movie, denoised, n_peaks=1, n_low=1, min_separation_frac=0.2)

    for sample in samples:
        r, c = sample["row"], sample["col"]
        assert np.allclose(sample["before"], movie[r, c, :])
        assert np.allclose(sample["after"], denoised[r, c, :])
