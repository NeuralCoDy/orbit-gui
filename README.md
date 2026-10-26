# orbit-gui

An assessment-forward GUI for the analysis of functional neural optical imaging data:
validation metrics are computed and surfaced at every pipeline stage (motion correction,
source identification, demixing, ...), not bolted on at the end.

## Layout

- `src/orbit/` -- pure-Python algorithm and validation-metric core, no Qt. Every stage
  and every metric is a standalone function (e.g. `orbit.projections.mean_projection`),
  independently unit-tested, and called by the GUI rather than implemented inside it.
- `src/orbitapp/` -- the PySide6 GUI shell. One tab per pipeline stage; tabs only call
  `orbit` functions and display results. Mirrors the architecture of the sibling
  [pyGraFT](https://github.com/adamshch/GraFT-analysis) project (`AppState` +
  per-stage tabs), and reuses the `MovieSliderWidget` bundled with the
  [roiapp](https://pypi.org/project/roiapp/) distribution for movie playback.

## Install

```bash
pip install orbit-gui[gui]
```

or, for a development checkout:

```bash
pip install -e ".[test,gui]"
```

`orbit`'s native (C++) accelerator for the slower per-pixel/per-trace operations
(local correlation and mode projections, OASIS deconvolution, the per-pixel
Ljung-Box test) builds automatically whenever a C++ compiler is available (via
`setup.py`'s `ext_modules` -- see PyPI's published wheels, which ship it
prebuilt) -- falls back to pure numpy/Python if no compiler is found, so this
never blocks `pip install`. To rebuild it in place after editing the `.cpp`
source, without reinstalling:

```bash
src/orbit/_native/build_native.sh
```

Real-SEUDO's own per-cell FISTA solve has a separate optional accelerator (also
falls back to pure Python if skipped) -- needs FFTW3 in addition to a C++14
compiler and pybind11 (Debian/Ubuntu: `apt-get install libfftw3-dev`). It builds
automatically alongside the accelerator above wherever FFTW3 is found (including
in PyPI's published wheels, on the platforms where FFTW3 could be provisioned in
CI). To rebuild it in place after editing the `.cpp` source, without reinstalling:

```bash
src/orbit/seudo/_native/build_native.sh
```

## Run

```bash
python3 -m orbitapp
```

## Develop

```bash
pytest
```

## Status

- **Load**: a movie (TIFF/folder-of-TIFFs/NPY/H5/MAT), previewed via an embedded,
  scrubbable movie player.
- **Data Projections**: mean/median/mode/variance/Fano-factor/local-correlation
  projections, computed lazily and cached.
- **Motion Correction**: rigid, piecewise-rigid, or PatchWarp-style piecewise-affine
  registration, with quality metrics (mMD/mCM/ECC, singular-value-spectrum tightening,
  spatial PC maps) shown alongside the before/after images. Runs in the background;
  results are explicit candidates previewed in the tab until committed to the active
  dataset, which is then recorded in the header's pipeline breadcrumb.

Later phases add source identification and demixing/contamination assessment -- see
project memory for the full roadmap.

## Changelog

### 0.1.3

- Source Extraction: the trace plot now redraws when a click replaces the ROI
  preview. It used to keep showing the first preview's trace.
- Correlation auto-threshold: when no threshold in 0.6-0.9 grows a mask (low-SNR
  movies whose local correlation peaks below 0.6), the search falls back to
  0.05-0.59 instead of returning an empty ROI with an all-zero trace. If nothing
  grows at all, the status bar says so and Add ROI stays disabled.

### 0.1.2

- Fix rigid motion correction wrapping content around the frame edges: the Fourier
  sub-pixel shift is circular, so a shifted frame used to pull pixels from the
  opposite border. Frames are now edge-padded by the shift before shifting and
  cropped back, so uncovered borders are filled with the nearest edge values.

### 0.1.1

- Fix the splash screen and header logo not appearing after `pip install orbit-gui`
  -- the image lived outside `src/` and was never included in the published wheels
  or sdist, so it silently vanished on any real install (as opposed to a dev
  checkout, where the file happened to still be reachable).

### 0.1.0

- Initial PyPI release, with cross-platform wheel builds (Linux/macOS/Windows) for
  both optional native C++ extensions.

