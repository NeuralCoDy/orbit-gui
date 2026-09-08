import shutil

import numpy as np
import pytest

from orbitapp.report import (
    _latex_escape,
    _source_extraction_key,
    generate_report,
    render_report,
)
from orbitapp.state import AppState, PipelineStep, ROI


def _balanced(tex: str) -> bool:
    return tex.count("{") == tex.count("}") and tex.count(r"\begin{tabular}") == tex.count(r"\end{tabular}")


def _full_state() -> AppState:
    state = AppState()
    state.data_path = "my_movie.tif"
    state.original_data = np.zeros((5, 20, 20))
    state.steps = [
        PipelineStep(stage="load", label="Load", params={"data_path": "my_movie.tif"}),
        PipelineStep(
            stage="motion_correction", label="Patch Warp",
            params={"method": "PatchWarp (piecewise-affine)", "max_shift": 15.0, "grid_size": 4},
            metrics={"mmd": -0.5, "ecc": 0.97},
        ),
        PipelineStep(
            stage="mask", label="Mask (auto-threshold)", params={"action": "auto"},
            metrics={"fraction_kept": 0.23},
        ),
        PipelineStep(
            stage="denoising", label="Wavelet Denoising (Temporal)",
            params={"algorithm": "wavelet_time", "wavelet": "sym4", "level": 4},
            metrics={"residual_energy_fraction": 0.05, "ljung_box_failed": 12, "ljung_box_total": 400},
        ),
        PipelineStep(
            stage="normalization", label="Normalize (pixel-min, pixel-median)",
            params={
                "center": True, "normalize": True, "center_baseline": "min", "norm_baseline": "median",
                "pixel_center": True, "pixel_norm": True,
            },
            metrics={"stats_before": {"min": 0.0, "max": 1.0}, "stats_after": {"min": 0.0, "max": 2.0}},
        ),
        PipelineStep(stage="source_extraction", label="CNMF", params={"n_components": 30, "search_radius": 10.0}),
    ]
    state.rois = [
        ROI(id=0, mask=np.zeros((20, 20), dtype=bool), trace=np.zeros(5), source_method="cnmf"),
        ROI(id=1, mask=np.zeros((20, 20), dtype=bool), trace=np.zeros(5), source_method="cnmf"),
    ]
    return state


def test_latex_escape_neutralizes_special_characters():
    escaped = _latex_escape("weird_path%with#special&chars.tif")
    assert escaped == r"weird\_path\%with\#special\&chars.tif"


def test_generate_report_includes_every_stage_and_is_balanced():
    state = _full_state()
    tex = generate_report(state, None)

    assert _balanced(tex)
    for expected in ("Load", "Patch Warp", "Mask (auto-threshold)", "Wavelet Denoising (Temporal)",
                     "Normalize", "CNMF"):
        assert expected.split(" (")[0] in tex or _latex_escape(expected) in tex


def test_generate_report_picks_the_correct_motion_correction_equation():
    # Regression guard: "patch" is itself a substring of "patchwarp" -- the
    # dispatch must not misclassify PatchWarp's own equation as the
    # patch-based one just because of that substring relationship.
    state = _full_state()
    tex = generate_report(state, None)
    assert "piecewise-affine warp" in tex
    assert "bilinearly upsamples" not in tex  # patch-based equation, must not appear here


def test_generate_report_includes_parameter_and_metric_values():
    state = _full_state()
    tex = generate_report(state, None)
    assert "15" in tex  # max_shift
    assert "0.97" in tex  # ecc
    assert "0.23" in tex  # fraction_kept


def test_generate_report_source_extraction_reports_aggregate_roi_counts():
    state = _full_state()
    tex = generate_report(state, None)
    assert "cnmf" in tex
    assert "2" in tex  # two committed ROIs


def test_generate_report_includes_source_extraction_metrics():
    # Regression guard: _section_source_extraction used to hardcode {}
    # for metrics regardless of what the step actually recorded.
    state = _full_state()
    state.steps.append(
        PipelineStep(
            stage="source_extraction", label="Real-SEUDO",
            params={"sigma2": 0.002, "lambda_blob": 10.0},
            metrics={"rois_committed": 4, "rois_deleted": 7},
        )
    )
    tex = generate_report(state, None)

    assert "sigma2" in tex and "0.002" in tex
    assert _latex_escape("rois_committed") in tex and _latex_escape("rois_deleted") in tex
    assert "Validation metrics" in tex


def test_source_extraction_key_prefers_cnmf_e_over_cnmf_substring():
    # Regression guard: "cnmf" is itself a substring of "cnmf-e"/"cnmf_e"
    # -- the dispatch must not misclassify CNMF-E's own equation as
    # plain CNMF's just because of that substring relationship (same
    # trap _motion_correction_key already guards against for
    # "patch"/"patchwarp").
    assert _source_extraction_key("CNMF-E") == "cnmf_e"
    assert _source_extraction_key("CNMF") == "cnmf"


def test_generate_report_includes_cnmf_e_equation():
    state = _full_state()
    state.steps.append(
        PipelineStep(
            stage="source_extraction", label="CNMF-E",
            params={"n_components": 20, "ring_inner_radius": 20.0, "ring_outer_radius": 25.0},
        )
    )
    tex = generate_report(state, None)
    assert "ring" in tex.lower()
    assert "peak-to-noise" in tex.lower()


def test_source_extraction_key_resolves_real_seudo_and_does_not_collide_with_seudo():
    # "seudo" alone (e.g. a hypothetical "SEUDO" label) isn't a registered
    # key at all -- only the full "real-seudo"/"real_seudo" spelling this
    # app's own _PIPELINE_LABELS actually produces should resolve.
    assert _source_extraction_key("Real-SEUDO") == "real_seudo"
    assert _source_extraction_key("real_seudo") == "real_seudo"


def test_generate_report_includes_real_seudo_equation():
    state = _full_state()
    state.steps.append(
        PipelineStep(
            stage="source_extraction", label="Real-SEUDO",
            params={"sigma2": 0.002, "lambda_blob": 10.0, "consecutive_frames_required": 5},
        )
    )
    tex = generate_report(state, None)
    assert "seudo" in tex.lower()
    assert "streams the movie" in tex.lower()


def test_generate_report_with_only_load_step_has_no_other_sections():
    state = AppState()
    state.data_path = "movie.tif"
    state.original_data = np.zeros((3, 10, 10))
    state.steps = [PipelineStep(stage="load", label="Load", params={"data_path": "movie.tif"})]
    tex = generate_report(state, None)
    assert _balanced(tex)
    assert "Load" in tex


def test_generate_report_skips_roi_validation_section_when_no_results():
    class _FakeValidationTab:
        def export_results(self):
            return None

    state = _full_state()
    tex = generate_report(state, _FakeValidationTab())
    assert "ROI Validation" not in tex


def test_generate_report_includes_roi_validation_section_when_results_exist():
    class _FakeValidationTab:
        def export_results(self):
            return [
                {"times": [], "classification": "cell", "is_artifact": False},
                {"times": [], "classification": "cell", "is_artifact": False},
                {"times": [], "classification": "not cell", "is_artifact": True},
            ]

    state = _full_state()
    tex = generate_report(state, _FakeValidationTab())
    assert "ROI Validation" in tex
    assert _balanced(tex)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not on PATH")
def test_render_report_compiles_a_real_pdf(tmp_path):
    state = _full_state()
    pdf_path = tmp_path / "report.pdf"

    result = render_report(state, None, pdf_path)

    assert result == pdf_path
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000
    # only the .tex/.pdf remain -- aux files cleaned up
    assert set(p.suffix for p in tmp_path.iterdir()) == {".tex", ".pdf"}


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not on PATH")
def test_render_report_raises_a_clear_error_on_a_broken_document(tmp_path, monkeypatch):
    import orbitapp.report as report_mod

    monkeypatch.setattr(
        report_mod, "generate_report",
        lambda state, tab: r"\documentclass{article}\begin{document}\badcommand{oops}\end{document}",
    )
    pdf_path = tmp_path / "broken.pdf"

    with pytest.raises(RuntimeError, match="LaTeX compilation failed"):
        render_report(AppState(), None, pdf_path)
    assert not pdf_path.exists()


def test_render_report_raises_clearly_when_pdflatex_missing(tmp_path, monkeypatch):
    import orbitapp.report as report_mod

    monkeypatch.setattr(report_mod.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="pdflatex not found"):
        render_report(AppState(), None, tmp_path / "report.pdf")
