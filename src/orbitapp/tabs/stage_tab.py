"""Shared skeleton for a pipeline-stage tab (Motion Correction, Denoising,
Normalization, ...): Apply runs the stage's algorithm in a background
worker and previews the result as a candidate; Commit makes it the
active dataset and records the step in the pipeline breadcrumb. Nothing
overwrites the active dataset until Commit is clicked.

Subclasses provide the algorithm-specific pieces: `_stage_name` and
`_result_key`/`_stage_key` class attributes, `_build_controls_row`/
`_build_metrics` to lay out their own widgets, `_current_fingerprint` to
snapshot their parameter widgets as a named dict (see below),
`restore_params` to set widgets back from a loaded dict (the inverse),
`_start_worker` to launch the Apply run, and `_render_result` to show a
finished candidate. `_extract_metrics` is optional -- headline QC
numbers worth recording alongside a commit (see session_io.py).

Clicking Apply again with the same input data and the same parameters as
the last successful run would just reproduce the same candidate, so
that's caught before spending real time on it -- confirmed once via a
Yes/No dialog rather than silently blocked, in case the algorithm is
non-deterministic or the user just wants to force a rerun.

When the active movie is memmap-backed (see orbitapp.io.is_memmap),
Apply and Commit split into two different-sized jobs rather than one:
Apply always previews just the first 5000 frames (bounded, in RAM,
using the existing algorithm unmodified -- see orbitapp.io.preview_slice)
while Commit re-runs the stage across the *whole* movie in time chunks,
writing straight to a new FITS-backed memmap (see orbitapp.fits_io) so
the full result never needs to fit in RAM either. Subclasses that
produce a movie (Motion Correction, Denoising, Normalization) implement
`_chunked_commit`; subclasses without one simply can't be committed
against a memmap input (StageTab's own default raises, which becomes a
Commit failure dialog rather than any silent wrong behavior). Non-memmap
input is completely unaffected -- Commit stays the free "promote
whatever Apply already computed" it's always been.

A volumetric movie (state.volumetric) runs an entirely separate,
dimension-specific algorithm (e.g. rigid_motion_correct_3d, not the 2D
rigid path) rather than a parameter variant of the same one, so
on_data_loaded/_apply/_commit each dispatch to a `*_volumetric`
counterpart instead of just branching internally. Subclasses that
support volumetric data set `_supports_volumetric = True` and implement
`_start_worker_volumetric` (parallel to `_start_worker`),
`_result_key_3d` (parallel to `_result_key`), and optionally
`_chunked_commit_volumetric` (parallel to `_chunked_commit`, for a
memmap-backed volumetric Commit).

`_supports_volumetric` defaults to False and must be opted into
explicitly, rather than every subclass automatically getting the
volumetric dispatch: every tab -- including ones with no volumetric
implementation at all, e.g. Detrending's single-trace-plot panel --
gets on_data_loaded() called on it whenever ANY tab commits (see
app.py's cross-tab data_changed wiring), regardless of state.volumetric
or whether that particular tab is even usable in volumetric mode.
Dispatching unconditionally would run _on_volumetric_data_loaded's
StagePanel-shaped preview code against a tab whose self.panel isn't one
(confirmed: an AttributeError on self.panel.before_view, not a graceful
no-op) the first time a volumetric commit happened to be followed by a
refresh of a non-supporting tab.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QVBoxLayout, QWidget

from ..io import is_memmap, preview_slice
from ..state import AppState
from ..volumetric_io import preview_slice_volumetric
from ..widgets import BusyBar, CommitControls, StagePanel, confirm_recompute
from ..workers import FunctionWorker, run_worker


class StageTab(QWidget):
    data_changed = Signal()  # emitted only on Commit, not on Apply

    _stage_name = "Stage"  # used in the failure dialog title
    _result_key = "result"  # key into the worker's result dict for the candidate array
    _result_key_3d = "result_3d"  # same, for the volumetric worker's result dict (see class docstring)
    _stage_key = "stage"  # lowercase identifier recorded in AppState.steps / session_io.py
    _chunk_frames = 500  # time-chunk size for a memmap input's chunked Commit
    _supports_volumetric = False  # set True by subclasses with a real *_volumetric implementation (see below)

    def __init__(
        self,
        state: AppState,
        apply_label: str,
        before_title: str = "Raw (mean projection)",
        after_title: str = "Candidate (mean projection)",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.worker: FunctionWorker | None = None
        self._input_movie: np.ndarray | None = None
        self._pending_result: dict | None = None
        self._pending_step_label: str | None = None
        self._pending_params: dict = {}
        self._pending_fingerprint: tuple | None = None
        self._last_run: tuple | None = None  # (id(movie), fingerprint of _current_fingerprint()) of the last successful Apply

        layout = QVBoxLayout(self)

        self.commit_controls = CommitControls(apply_label=apply_label)
        self.commit_controls.set_apply_enabled(False)
        self.commit_controls.apply_clicked.connect(self._apply)
        self.commit_controls.commit_clicked.connect(self._commit)

        layout.addLayout(self._build_controls_row())

        self.busy_bar = BusyBar()
        layout.addWidget(self.busy_bar)

        self.status_label = QLabel("No data loaded.")
        layout.addWidget(self.status_label)

        self.panel = self._build_panel(before_title, after_title)
        layout.addWidget(self.panel)

        self._build_metrics()

    def _build_panel(self, before_title: str, after_title: str) -> QWidget:
        """The stage's main content widget, stored as self.panel --
        defaults to the shared before/after-image StagePanel every
        other stage tab uses. Override for a stage whose main figure
        isn't a pair of images (e.g. a single trace plot) -- if you do,
        also override _show_before_preview (and, for a stage that
        supports volumetric data too, _show_before_preview_volumetric)
        below, since their defaults assume a StagePanel."""
        return StagePanel(before_title=before_title, after_title=after_title)

    def _show_before_preview(self, movie: np.ndarray) -> None:
        """Populates self.panel with a preview of the freshly-loaded
        movie, before any Apply has run -- called from on_data_loaded.
        Default assumes self.panel is a StagePanel; see _build_panel."""
        # preview_slice bounds this to the first 5000 frames for a
        # memmap movie -- otherwise this mean projection alone would
        # force a full read of an arbitrarily large movie just to
        # populate the "Raw" thumbnail.
        self.panel.before_view.setImage(preview_slice(movie).mean(axis=2))
        self.panel.set_before_movie(movie)

    def _build_controls_row(self) -> QHBoxLayout:
        """Returns the row above the busy bar: algorithm choice,
        Parameters button, self.commit_controls, etc."""
        raise NotImplementedError

    def _build_metrics(self) -> None:
        """Adds self.metrics_label and any other QC widgets to self.panel."""
        raise NotImplementedError

    def _on_data_reset(self) -> None:
        """Hook for subclass state that also needs clearing on reload
        (e.g. QC location markers). No-op by default."""

    def on_data_loaded(self) -> None:
        if self.state.volumetric and self._supports_volumetric:
            self._on_volumetric_data_loaded()
            return
        movie = self.state.active_data()
        self.commit_controls.set_apply_enabled(movie is not None)
        self.commit_controls.set_commit_enabled(False)
        self._pending_result = None
        self._pending_step_label = None
        self._last_run = None  # a new/changed movie invalidates any prior "already run" state
        self._clear_stale_candidate()
        self._on_data_reset()
        if hasattr(self.panel, "set_volumetric"):
            self.panel.set_volumetric(False)
        if movie is not None:
            self._show_before_preview(movie)
            self.status_label.setText(f"Ready. shape={movie.shape}")

    def _on_volumetric_data_loaded(self) -> None:
        """Volumetric counterpart of on_data_loaded."""
        movie = self.state.active_data()
        self.commit_controls.set_apply_enabled(movie is not None)
        self.commit_controls.set_commit_enabled(False)
        self._pending_result = None
        self._pending_step_label = None
        self._last_run = None
        self._clear_stale_candidate()
        self._on_data_reset()
        if hasattr(self.panel, "set_volumetric"):
            self.panel.set_volumetric(True)
        if movie is not None:
            self._show_before_preview_volumetric(movie)
            self.status_label.setText(f"Ready. shape={movie.shape} (volumetric)")

    def _clear_stale_candidate(self) -> None:
        """Drops references to this tab's own last Apply candidate:
        ``self._input_movie`` (already cleared by _finish_commit on a
        successful commit, but also needed here for an ABANDONED
        candidate -- Applied but never Committed before some OTHER tab's
        commit made it moot) and the panel's "after" movie (kept alive so
        its own Play Movie button stays usable right after Apply/Commit,
        only released once the pipeline has genuinely moved past this tab
        -- i.e. exactly when on_data_loaded fires due to a DIFFERENT
        tab's commit, since data_changed only wires to every OTHER tab's
        on_data_loaded, not this one's own).

        Without this, every StageTab-derived tab keeps two full-size
        movie-shaped arrays alive for the rest of the app's lifetime once
        Apply has been clicked on it even once -- confirmed via a real
        5-stage pipeline run (Motion Correction -> Mask -> Denoising ->
        Normalization -> Detrending) on the real default dataset: ~9.4GB
        of pure waste on top of a single ~1GB movie, since none of those
        stale _input_movie/after references were the current active
        dataset by the time the pipeline had moved on.

        self.panel might not support movies at all (e.g. Detrending's
        single-trace-plot panel -- see _build_panel), so this is guarded
        rather than assuming every subclass's panel is a StagePanel."""
        self._input_movie = None
        if hasattr(self.panel, "set_after_movie"):
            self.panel.set_after_movie(None)
        if hasattr(self.panel, "set_after_volume"):
            self.panel.set_after_volume(None)

    def _show_before_preview_volumetric(self, movie: np.ndarray) -> None:
        """Volumetric counterpart of _show_before_preview -- shows the
        (T, L, W, D) volume's time-mean in the StagePanel's 3D VolumeView
        (see StagePanel.set_volumetric). Default assumes self.panel is a
        StagePanel; override alongside _build_panel/_show_before_preview
        for a stage whose main figure isn't a pair of images (see
        DetrendingTab, whose volumetric preview is a 1D per-volume trace,
        not an image at all)."""
        self.panel.set_before_volume(preview_slice_volumetric(movie))

    def _current_fingerprint(self) -> dict:
        """Named snapshot of every widget value that affects the
        algorithm's output -- subclasses read their own parameter
        widgets into a dict. Doubles as (a) a hashable-once-sorted
        fingerprint to detect an unchanged rerun and (b) the exact params
        recorded against a commit for session_io.py's pipeline file --
        see restore_params for the inverse (loading a session back in)."""
        raise NotImplementedError

    def restore_params(self, params: dict) -> None:
        """Sets this stage's parameter widgets from a previously-saved
        _current_fingerprint() dict -- used when loading a session back
        in. Subclasses should tolerate missing/extra keys gracefully
        (a saved session may predate a newer parameter)."""
        raise NotImplementedError

    def _extract_metrics(self, result: dict) -> dict:
        """Headline scalar QC numbers worth recording alongside a commit
        (see session_io.py) -- e.g. a residual energy fraction or a
        before/after correlation. No-op by default; only the *small*,
        JSON-serializable summary numbers belong here, not full arrays
        (those already live in the candidate result itself)."""
        return {}

    def _start_worker(self, movie: np.ndarray) -> None:
        """Sets self._pending_step_label and launches self.worker via
        run_worker, with self._on_finished/self._on_failed as callbacks."""
        raise NotImplementedError

    def _start_worker_volumetric(self, movie: np.ndarray) -> None:
        """Volumetric counterpart of _start_worker -- ``movie`` is a
        (T, L, W, D) preview, not (H, W, T). Only needed by subclasses
        that support volumetric data; state.volumetric is only ever set
        when the data path supports it (see load_tab.py), so this
        default is unreachable in practice rather than a real gap."""
        raise NotImplementedError

    def _apply(self) -> None:
        if self.state.volumetric and self._supports_volumetric:
            self._apply_volumetric()
            return
        movie = self.state.active_data()
        if movie is None:
            QMessageBox.warning(self, "No data", "Load data on the Load tab first.")
            return

        params = self._current_fingerprint()
        fingerprint = (id(movie), tuple(sorted(params.items())))
        if fingerprint == self._last_run:
            message = f"{self._stage_name} was already run with these exact parameters on this data."
            if not confirm_recompute(self, message):
                return

        self._pending_fingerprint = fingerprint
        self._pending_params = params
        # self._input_movie stays the REAL (possibly memmap, possibly
        # much longer than 5000 frames) movie -- Commit needs it for the
        # chunked full-movie pass. Only the worker's own input is capped.
        self._input_movie = movie
        self.commit_controls.set_apply_enabled(False)
        self.commit_controls.set_commit_enabled(False)
        self._start_worker(preview_slice(movie))

    def _apply_volumetric(self) -> None:
        """Volumetric counterpart of _apply -- same fingerprint/confirm
        logic, against preview_slice_volumetric and _start_worker_volumetric
        instead."""
        movie = self.state.active_data()
        if movie is None:
            QMessageBox.warning(self, "No data", "Load data on the Load tab first.")
            return

        params = self._current_fingerprint()
        fingerprint = (id(movie), tuple(sorted(params.items())))
        if fingerprint == self._last_run:
            message = f"{self._stage_name} was already run with these exact parameters on this data."
            if not confirm_recompute(self, message):
                return

        self._pending_fingerprint = fingerprint
        self._pending_params = params
        self._input_movie = movie
        self.commit_controls.set_apply_enabled(False)
        self.commit_controls.set_commit_enabled(False)
        self._start_worker_volumetric(preview_slice_volumetric(movie))

    def _render_result(self, result: dict) -> None:
        """Updates the panel images/movies and self.metrics_label (plus
        any other QC widgets) for a finished candidate."""
        raise NotImplementedError

    def _on_finished(self, result: dict) -> None:
        self._last_run = self._pending_fingerprint
        self._pending_result = result
        self._render_result(result)

        self.busy_bar.stop("")
        self.status_label.setText(
            f"Candidate ready (shape={result[self._result_key].shape}). "
            "Click 'Commit to Active Dataset' to keep it, or Apply again to discard and retry."
        )
        self.commit_controls.set_apply_enabled(True)
        self.commit_controls.set_commit_enabled(True)

    def _on_failed(self, message: str) -> None:
        self.busy_bar.stop("Failed.")
        self.status_label.setText(f"Failed: {message}")
        QMessageBox.critical(self, f"{self._stage_name} failed", message)
        self.commit_controls.set_apply_enabled(True)

    def _chunked_commit(self, source: np.ndarray, output_path: Path) -> np.ndarray:
        """Runs this stage across the WHOLE ``source`` movie (a memmap,
        possibly far longer than the 5000-frame preview Apply ran
        against) in `self._chunk_frames`-sized time chunks, writing each
        chunk into a new FITS-backed memmap at ``output_path`` (see
        orbitapp.fits_io.create_fits_memmap) and returning it. Only
        needed by subclasses that produce a movie (Motion Correction,
        Denoising, Normalization); subclasses that don't override this
        (e.g. anything not chunkable) simply can't be committed against
        a memmap input -- surfaces as a Commit failure dialog via
        _on_chunked_commit_failed, not a silent full materialization."""
        raise NotImplementedError(f"{self._stage_name} doesn't support committing a memory-mapped movie.")

    def _chunked_commit_volumetric(self, source: np.ndarray, output_path: Path) -> np.ndarray:
        """Volumetric counterpart of _chunked_commit -- ``source`` is
        (T, L, W, D). See _run_chunked_commit for the dispatch between
        the two."""
        raise NotImplementedError(f"{self._stage_name} doesn't support committing a memory-mapped volumetric movie.")

    def _run_chunked_commit(self, source: np.ndarray, output_path: Path) -> np.ndarray:
        """The actual worker callable _start_chunked_commit launches --
        dispatches to _chunked_commit or _chunked_commit_volumetric so
        neither subclass override needs to repeat that check itself."""
        if self.state.volumetric and self._supports_volumetric:
            return self._chunked_commit_volumetric(source, output_path)
        return self._chunked_commit(source, output_path)

    def _commit(self) -> None:
        if self._pending_result is None:
            return
        if is_memmap(self._input_movie):
            self._start_chunked_commit()
            return
        if self.state.volumetric and self._supports_volumetric:
            # The worker's result dict carries both a depth-projected 2D
            # array (under _result_key, for _on_finished/_render_result's
            # shared display code) and the real 4D array (under
            # _result_key_3d) -- only Commit needs to tell them apart.
            self._finish_commit(self._pending_result[self._result_key_3d])
            return
        self._finish_commit(self._pending_result[self._result_key])

    def _finish_commit(self, data: np.ndarray) -> None:
        metrics = self._extract_metrics(self._pending_result)
        self.state.commit(
            data, self._pending_step_label, stage=self._stage_key, params=self._pending_params, metrics=metrics,
        )
        # _input_movie was only ever needed as _start_chunked_commit's
        # source (see its own docstring) -- nothing reads it again after
        # a successful commit until the next Apply overwrites it.
        # Dropped here immediately rather than waiting for some LATER
        # tab's own commit to eventually trigger this tab's own
        # on_data_loaded/_clear_stale_candidate -- see that method's
        # docstring for the full picture (this is the half of it that
        # doesn't need to wait for a cross-tab signal).
        self._input_movie = None
        self.status_label.setText(f"Committed as pipeline step '{self._pending_step_label}'.")
        self.commit_controls.set_commit_enabled(False)
        self.data_changed.emit()

    def _start_chunked_commit(self) -> None:
        self.commit_controls.set_apply_enabled(False)
        self.commit_controls.set_commit_enabled(False)
        fd, path = tempfile.mkstemp(suffix=".fits", prefix=f"orbit_{self._stage_key}_")
        os.close(fd)
        self.worker = run_worker(
            self.busy_bar, f"Committing {self._stage_name} across the full movie (this can take a while)...",
            self._run_chunked_commit, self._input_movie, Path(path),
            on_success=self._on_chunked_commit_finished, on_failure=self._on_chunked_commit_failed,
        )

    def _on_chunked_commit_finished(self, output: np.ndarray) -> None:
        self.busy_bar.stop("")
        self.commit_controls.set_apply_enabled(True)
        self._finish_commit(output)

    def _on_chunked_commit_failed(self, message: str) -> None:
        self.busy_bar.stop("Commit failed.")
        self.status_label.setText(f"Commit failed: {message}")
        QMessageBox.critical(self, f"{self._stage_name} commit failed", message)
        self.commit_controls.set_apply_enabled(True)
        self.commit_controls.set_commit_enabled(True)  # candidate is still there -- let them retry or Apply again
