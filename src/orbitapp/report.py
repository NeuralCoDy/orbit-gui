"""Generates a LaTeX report of the currently committed pipeline (steps,
governing equations, parameters, validation metrics) and compiles it to
PDF via a system LaTeX toolchain (pdflatex) -- see the header bar's
"Generate Report" button (orbitapp.app._on_generate_report_clicked).

Text/tables/equations only -- no embedded figures (a possible future
addition once this structure is settled).
"""

from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from .state import AppState, PipelineStep

if TYPE_CHECKING:
    from .tabs.roi_validation_tab import ROIValidationTab

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _latex_escape(text) -> str:
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in str(text))


def _format_value(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_format_value(v) for v in value) + ")"
    if isinstance(value, dict):
        return "; ".join(f"{k}={_format_value(v)}" for k, v in value.items())
    return _latex_escape(value)


def _dict_table(rows: dict, value_header: str = "Value") -> str:
    if not rows:
        return ""
    lines = [r"\begin{tabular}{@{}ll@{}}", r"\toprule", f"Name & {value_header} \\\\", r"\midrule"]
    for key, value in rows.items():
        lines.append(f"{_latex_escape(key)} & {_format_value(value)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


# -- Per-method equations/descriptions, quoted directly from the actual
# implementation (see orbit/motion_correction.py, orbit/denoising.py,
# orbit/masking.py, orbit/cnmf.py, orbit/roi_extraction_*.py) -- volumetric
# 3D variants aren't re-derived separately since the underlying math is
# the same kernel generalized to (L, W, D), just noted as such in prose.

_MOTION_CORRECTION_EQUATIONS = {
    "rigid": (
        r"Rigid registration estimates a subpixel translation $(\Delta y, \Delta x)$ via phase "
        r"correlation (Guizar-Sicairos et al., 2008):"
        "\n"
        r"\[ (\Delta y, \Delta x) = \arg\max_{\delta}\ \mathcal{F}^{-1}\!\left\{ \hat{F}(f)\,"
        r"\hat{G}^{*}(f)\, e^{2\pi i f \cdot \delta} \right\}, \]"
        "\n"
        r"where $F$ is the reference template and $G$ the frame being registered, then applies the "
        r"shift via the Fourier shift theorem, $\hat{I}'(f) = \hat{I}(f)\, e^{-2\pi i f \cdot "
        r"(\Delta y, \Delta x)}$. For a volumetric movie the same estimate/shift is computed over a "
        r"3D $(\Delta l, \Delta w, \Delta d)$ translation instead."
    ),
    "patch": (
        r"Patch-based (piecewise-rigid) registration estimates an independent shift per spatial "
        r"patch, clipped to within \texttt{max\_dev} of the whole-frame rigid shift, then bilinearly "
        r"upsamples the coarse per-patch shift grid to a smooth per-pixel displacement field "
        r"$(\delta_y(y,x), \delta_x(y,x))$ and warps by cubic spline interpolation: "
        r"$I'(y,x) = I(y - \delta_y(y,x),\, x - \delta_x(y,x))$."
    ),
    "patchwarp": (
        r"PatchWarp estimates a piecewise-affine warp per patch by maximizing the enhanced "
        r"correlation coefficient (ECC) between template and frame across a multi-resolution image "
        r"pyramid, then blends overlapping patches into the registered frame."
    ),
}

_DENOISING_EQUATIONS = {
    "wavelet_time": (
        r"Wavelet shrinkage decomposes each pixel's time-trace via a discrete wavelet transform and "
        r"soft-thresholds each detail subband,"
        "\n"
        r"\[ \hat{w} = \mathrm{sign}(w)\,\max(|w| - \lambda,\ 0), \]"
        "\n"
        r"with threshold $\lambda$ chosen either by the universal rule $\lambda = \sigma_n\sqrt{2\ln n}$ "
        r"or BayesShrink, $\lambda = \sigma_n^2 / \sigma_x$ with $\sigma_x = \sqrt{\max(\sigma_w^2 - "
        r"\sigma_n^2,\ 0)}$; the noise level $\sigma_n$ is estimated via the median absolute deviation "
        r"of the finest detail subband."
    ),
    "wavelet_space": (
        r"Wavelet shrinkage decomposes each frame (2D) via a discrete wavelet transform and "
        r"soft-thresholds each detail subband the same way as temporal wavelet denoising (see above), "
        r"applied per-frame instead of per-pixel-trace."
    ),
    "gaussian": (
        r"Gaussian filtering convolves the movie with a separable Gaussian kernel, with independent "
        r"spatial ($\sigma_{\text{space}}$) and temporal ($\sigma_{\text{time}}$) widths: "
        r"$I' = I * G_{\sigma_{\text{space}}, \sigma_{\text{space}}, \sigma_{\text{time}}}$. For a "
        r"volumetric movie $\sigma_{\text{space}}$ applies equally to all three spatial axes "
        r"$(L, W, D)$."
    ),
    "median": (
        r"Median filtering replaces each pixel/voxel with the median of its "
        r"(\texttt{space\_window} $\times$ \texttt{space\_window} $\times$ \texttt{time\_window}) "
        r"neighborhood (or the 3D-isotropic generalization for a volumetric movie) -- a rank-order "
        r"filter, robust to isolated outlier spikes."
    ),
    "pca": (
        r"PCA denoising truncates the singular value decomposition of each overlapping spatiotemporal "
        r"block,"
        "\n"
        r"\[ X \approx U_k \Sigma_k V_k^{\top}, \]"
        "\n"
        r"keeping only the top $k$ components, then blends overlapping blocks' reconstructions."
    ),
}

_TRIANGLE_EQUATION = (
    r"The triangle method (Zack, Rogers \& Latt, 1977) thresholds the field of view's "
    r"mean-intensity histogram by drawing a line from its peak bin to its farthest empty tail bin "
    r"and choosing the threshold at the point of maximum perpendicular distance from that line to "
    r"the histogram curve."
)

_MASK_EQUATIONS = {
    "triangle": _TRIANGLE_EQUATION,
    "auto": _TRIANGLE_EQUATION,  # legacy saved-session value, from before multiple methods existed
    "otsu": (
        r"Otsu's method thresholds by maximizing the between-class intensity variance of the field of "
        r"view's mean-intensity histogram, treating it as a mixture of two classes (foreground/"
        r"background)."
    ),
    "manual": r"A user-supplied, fixed intensity threshold: pixels above \texttt{threshold} are kept.",
    "percentile": (
        r"Keeps the brightest \texttt{percentile} percent of pixels: the threshold is the "
        r"$(100 - \texttt{percentile})$-th percentile of the field of view's mean-intensity "
        r"distribution."
    ),
    "clear": r"An all-True (no-op) mask -- every pixel/voxel is kept.",
}

_SOURCE_EXTRACTION_EQUATIONS = {
    "correlation": (
        r"Correlation-based seeding grows a region outward from a seed pixel, adding neighboring "
        r"pixels whose fluorescence trace correlation with the seed (or the growing region's mean "
        r"trace) exceeds a threshold."
    ),
    "pca_ica": (
        r"PCA-ICA reduces the movie's pixel-by-time matrix via PCA to a temporal basis, then applies "
        r"spatial ICA to the corresponding spatial basis to unmix statistically independent spatial "
        r"components (Mukamel et al., 2009)."
    ),
    "cnmf": (
        r"CNMF (constrained non-negative matrix factorization) models the movie as"
        "\n"
        r"\[ Y \approx A C + b f^{\top}, \]"
        "\n"
        r"with non-negative spatial footprints $A$, temporal traces $C$, and a rank-1 background "
        r"$b f^{\top}$; each trace's spikes are deconvolved via the OASIS AR($p$) generative model,"
        "\n"
        r"\[ c_t = \sum_{i=1}^{p} g_i\, c_{t-i} + s_t,\qquad s_t \geq 0. \]"
    ),
    "cnmf_e": (
        r"CNMF-E (Zhou et al., 2018), for one-photon/microendoscopic data, extends CNMF with two "
        r"changes suited to strong, spatially-varying out-of-focus background fluorescence: "
        r"components are seeded at peaks of correlation image $\times$ peak-to-noise ratio (PNR) "
        r"rather than plain intensity, and the background is modeled per-pixel from a local ring of "
        r"neighboring pixels rather than one global low-rank term,"
        "\n"
        r"\[ b_i(t) = \sum_{j \in \text{ring}(i)} w_{ij}\, \big(Y_j(t) - (AC)_j(t)\big), \]"
        "\n"
        r"with ring weights $w_{ij}$ fit once by per-pixel least-squares regression against the "
        r"neuron-subtracted residual. Spatial/temporal updates and OASIS deconvolution are otherwise "
        r"identical to plain CNMF above."
    ),
    "graft": (
        r"GraFT (Graph-Filtered Temporal dictionary learning) jointly learns a spatial dictionary $S$ "
        r"and temporal dictionary $D$ by solving"
        "\n"
        r"\[ \min_{D, S}\ \|Y - DS\|_F^2 + \lambda \|S\|_1 + \tau\, \mathrm{tr}(S^{\top} L S), \]"
        "\n"
        r"where $L$ is the graph Laplacian of a $k$-nearest-neighbor graph built from each pixel's own "
        r"activity (Charles et al.). On volumetric data, the masked region's voxels are reorganized "
        r"into a flat pixel-by-time array before this same optimization, and the result scattered back "
        r"into a $(L, W, D)$ mask."
    ),
    "real_seudo": (
        r"Real-SEUDO streams the movie one frame at a time, fitting each known cell's profile against "
        r"a nonnegative, $\ell_1$-penalized sparse regression (the SEUDO solver, Gauthier \& Charles, "
        r"2021) and detecting candidate new cells in the unexplained residual. A candidate is tracked "
        r"across consecutive frames and promoted into the known-cell set once it's been consistently "
        r"detected for long enough, rather than requiring the whole movie or a fixed initial guess "
        r"upfront -- so, unlike every other method here, it never needs the whole movie in memory at "
        r"once."
    ),
}

_BASELINE_NAMES = {
    "min": "minimum", "median": "median", "mean": "mean", "mode": "half-sample mode",
    "max": "maximum", "robuststd": "MAD-based robust standard deviation",
}

_METRIC_NOTES = {
    "ljung_box_failed": (
        r"Ljung-Box whiteness test: $Q = T(T+2)\sum_k \rho_k^2/(T-k)$, tested against a $\chi^2$ "
        r"distribution -- a pixel fails if $\alpha \le 0.05$."
    ),
    "residual_energy_fraction": (
        r"Residual energy fraction: $\|Y_{\text{before}} - Y_{\text{after}}\|^2 / \|Y_{\text{before}}\|^2$."
    ),
    "mmd": r"mMD: difference in mean max-intensity-projection intensity, after minus before.",
    "ecc": r"ECC: Pearson correlation between the initial and final registration templates.",
    "mcm_before": r"mCM: mean per-frame Pearson correlation to a reference (mean) image.",
}


def _motion_correction_key(label: str) -> str:
    lowered = label.lower()
    if "patchwarp" in lowered.replace(" ", "").replace("-", ""):
        return "patchwarp"  # checked before "patch" -- "patch" is itself a substring of "patchwarp"
    if "patch" in lowered:
        return "patch"
    return "rigid"


def _source_extraction_key(label: str) -> str | None:
    lowered = label.lower()
    normalized = lowered.replace(" ", "").replace("-", "").replace("_", "")
    if "correlation" in lowered:
        return "correlation"
    if "pcaica" in normalized:
        return "pca_ica"
    if "cnmfe" in normalized:  # checked before "cnmf" -- "cnmf" is itself a substring of "cnmf-e"/"cnmf_e"
        return "cnmf_e"
    if "cnmf" in lowered:
        return "cnmf"
    if "graft" in lowered:
        return "graft"
    if "realseudo" in normalized:
        return "real_seudo"
    return None


def _section(title: str, prose: str, params: dict, metrics: dict) -> str:
    parts = [f"\\subsection*{{{_latex_escape(title)}}}"]
    if prose:
        parts.append(prose)
    if params:
        parts.append(r"\textbf{Parameters}\\")
        parts.append(_dict_table(params))
    if metrics:
        parts.append(r"\textbf{Validation metrics}\\")
        parts.append(_dict_table(metrics))
        notes = [_METRIC_NOTES[key] for key in metrics if key in _METRIC_NOTES]
        for note in dict.fromkeys(notes):  # de-duplicate, preserve order
            parts.append(r"\textit{" + note + r"}\\")
    return "\n\n".join(parts)


def _section_load(state: AppState, step: PipelineStep) -> str:
    path = state.data_path or "(unknown)"
    shape = state.original_data.shape if state.original_data is not None else None
    modality = ", ".join(state.modality_modifiers()) or "none"
    prose = (
        f"Data loaded from \\texttt{{{_latex_escape(path)}}}"
        + (f", shape {_format_value(list(shape))}" if shape is not None else "")
        + f". Data modality: {_latex_escape(modality)}."
    )
    return _section("Load", prose, step.params, step.metrics)


def _section_motion_correction(step: PipelineStep) -> str:
    key = _motion_correction_key(step.label)
    return _section(step.label, _MOTION_CORRECTION_EQUATIONS[key], step.params, step.metrics)


def _section_mask(step: PipelineStep) -> str:
    action = step.params.get("action", "triangle")
    return _section(step.label, _MASK_EQUATIONS.get(action, ""), step.params, step.metrics)


def _section_denoising(step: PipelineStep) -> str:
    algorithm = step.params.get("algorithm")
    equation = _DENOISING_EQUATIONS.get(algorithm, "")
    return _section(step.label, equation, step.params, step.metrics)


def _section_normalization(step: PipelineStep) -> str:
    params = step.params
    parts = []
    if params.get("center", True):
        baseline = _BASELINE_NAMES.get(params.get("center_baseline", "min"), params.get("center_baseline"))
        scope = "per-pixel" if params.get("pixel_center") else "field-wide"
        parts.append(f"subtracting a {scope} {baseline} baseline")
    if params.get("normalize", True):
        baseline = _BASELINE_NAMES.get(params.get("norm_baseline", "median"), params.get("norm_baseline"))
        scope = "per-pixel" if params.get("pixel_norm") else "field-wide"
        parts.append(f"dividing by a {scope} {baseline} scale")
    description = " and ".join(parts) if parts else "neither centering nor scaling"
    prose = (
        r"Normalization computes $\hat{Y} = (Y - B_{\text{center}}) / B_{\text{scale}}$; here: "
        + description + ". The MAD-based robust scale option is "
        r"$B = \mathrm{median}(|x - \mathrm{median}(x)|) / 0.6745$."
    )
    return _section(step.label, prose, step.params, step.metrics)


def _section_source_extraction(step: PipelineStep, rois: list) -> str:
    key = _source_extraction_key(step.label)
    prose = _SOURCE_EXTRACTION_EQUATIONS.get(key, "")
    counts = Counter(roi.source_method for roi in rois)
    count_line = (
        "; ".join(f"{_latex_escape(k)}: {v}" for k, v in counts.items()) if counts else "none committed yet"
    )
    section = _section(step.label, prose, step.params, step.metrics)
    return (
        section + "\n\n"
        r"\textbf{Committed ROIs so far (all Source Extraction commits combined):} " + count_line + "."
    )


def _section_generic(step: PipelineStep) -> str:
    return _section(step.label, "", step.params, step.metrics)


def _section_roi_validation(roi_validation_tab: "ROIValidationTab") -> str | None:
    results = roi_validation_tab.export_results()
    if not results:
        return None
    classifications = Counter(r["classification"] for r in results)
    n_artifact = sum(1 for r in results if r["is_artifact"])
    prose = (
        f"Each committed ROI's fluorescence trace was inspected for calcium transients; "
        f"{len(results)} ROI(s) were classified, {n_artifact} flagged as artifact(s)."
    )
    return _section("ROI Validation", prose, {}, dict(classifications))


_STAGE_BUILDERS = {
    "motion_correction": lambda state, step: _section_motion_correction(step),
    "mask": lambda state, step: _section_mask(step),
    "denoising": lambda state, step: _section_denoising(step),
    "normalization": lambda state, step: _section_normalization(step),
    "source_extraction": lambda state, step: _section_source_extraction(step, state.rois),
}

_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{hyperref}
\title{ORBIT GUI Pipeline Report}
\author{}
\date{\today}
\begin{document}
\maketitle

"""

_POSTAMBLE = "\n\n\\end{document}\n"


def generate_report(state: AppState, roi_validation_tab: "ROIValidationTab | None" = None) -> str:
    """Assembles the full .tex source describing every committed step in
    state.steps (in commit order), plus a final ROI Validation section
    (not itself a PipelineStep -- see roi_validation_tab.export_results)
    if a validation session was actually run."""
    sections = []
    for step in state.steps:
        if step.stage == "load":
            sections.append(_section_load(state, step))
        else:
            builder = _STAGE_BUILDERS.get(step.stage, lambda state, step: _section_generic(step))
            sections.append(builder(state, step))

    if roi_validation_tab is not None:
        validation_section = _section_roi_validation(roi_validation_tab)
        if validation_section:
            sections.append(validation_section)

    body = "\n\n".join(sections)
    return _PREAMBLE + body + _POSTAMBLE


def _parse_latex_error(tex_path: Path) -> str:
    log_path = tex_path.with_suffix(".log")
    if not log_path.exists():
        return "pdflatex failed and produced no log file."
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("! "):
            return f"LaTeX compilation failed: {line}"
    return "LaTeX compilation failed (see the .log file for details)."


def render_report(state: AppState, roi_validation_tab: "ROIValidationTab | None", pdf_path: Path) -> Path:
    """Writes the .tex next to pdf_path (same stem) and compiles it to
    PDF via pdflatex (run twice, so any cross-references resolve),
    cleaning up the .aux/.log/.out/.toc files afterward. Raises
    RuntimeError with a short, specific message (pdflatex missing, or
    the first "! "-prefixed line of the compile log) rather than a raw
    subprocess traceback."""
    if shutil.which("pdflatex") is None:
        raise RuntimeError(
            "pdflatex not found on PATH -- install a LaTeX distribution (e.g. TeX Live) to generate a PDF report."
        )

    pdf_path = Path(pdf_path)
    tex_path = pdf_path.with_suffix(".tex")
    tex_path.write_text(generate_report(state, roi_validation_tab))

    for _ in range(2):
        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                 "-output-directory", str(tex_path.parent), str(tex_path)],
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("pdflatex timed out while compiling the report.") from exc
        if result.returncode != 0:
            raise RuntimeError(_parse_latex_error(tex_path))

    for ext in (".aux", ".log", ".out", ".toc"):
        tex_path.with_suffix(ext).unlink(missing_ok=True)

    return pdf_path
