import numpy as np

from orbitapp.format import describe_source, format_header_summary, format_movie_summary


def test_describe_source_single_file(tmp_path):
    path = tmp_path / "movie.tif"
    path.write_bytes(b"0" * 2048)

    n_files, size_mb = describe_source(path)

    assert n_files == 1
    assert size_mb == 2048 / 1024**2


def test_describe_source_folder_of_tiffs(tmp_path):
    for name in ("a.tif", "b.tiff", "c.txt"):
        (tmp_path / name).write_bytes(b"0" * 1024)

    n_files, size_mb = describe_source(tmp_path)

    assert n_files == 2  # c.txt isn't a TIFF
    assert size_mb == 2 * 1024 / 1024**2


def test_format_movie_summary_includes_name_size_and_frame_info(tmp_path):
    path = tmp_path / "movie.tif"
    path.write_bytes(b"0" * (1024 * 1024))
    movie = np.zeros((64, 48, 100))

    summary = format_movie_summary(path, movie)

    assert "movie.tif" in summary
    assert "1 file" in summary
    assert "1.0 MB" in summary
    assert "64 x 48" in summary
    assert "100 time-steps" in summary


def test_format_header_summary_is_one_line():
    movie = np.zeros((64, 48, 100))
    summary = format_header_summary("/some/path/movie.tif", movie)

    assert "\n" not in summary
    assert "movie.tif" in summary
    assert "64 x 48" in summary
    assert "100 time-steps" in summary
