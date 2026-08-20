"""Save tab: two HDF5 outputs (see orbitapp.session_io for the exact
schema) -- "Save Pipeline" always writes just the reproducibility recipe
(ordered steps + parameters, correlation seed/threshold per ROI, ROI
Validation's last-used auto-classify/Run-SEUDO parameters); "Full Save"
additionally writes an output file with the actual derived results (ROI
masks/traces/spike_traces/neuropil_traces, each stage's QC metrics, ROI
Validation's classification/transient times). Movies are never saved --
"Load Session" re-associates with the original file via its recorded
path instead.

This tab only handles gathering data to write and files to read; actually
reconstructing AppState and refreshing every other tab from a loaded
session is MainWindow's job (see its _on_session_loaded), the same way
MainWindow already orchestrates every other cross-tab refresh.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from .. import session_io
from ..state import AppState
from .roi_validation_tab import ROIValidationTab

_FILE_FILTER = "HDF5 files (*.h5 *.hdf5)"


def _derive_paths(base_path: str) -> tuple[Path, Path]:
    """From one user-chosen path, the paired (pipeline, output) paths a
    Full Save writes -- strips an existing _pipeline/_output suffix
    first so re-saving to a name you picked before doesn't stack them."""
    p = Path(base_path)
    stem = p.stem
    for suffix in ("_pipeline", "_output"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    suffix = p.suffix or ".h5"
    return p.with_name(f"{stem}_pipeline{suffix}"), p.with_name(f"{stem}_output{suffix}")


def _sibling_path(path: str, from_tag: str, to_tag: str) -> Path | None:
    p = Path(path)
    if from_tag not in p.stem:
        return None
    return p.with_name(p.stem.replace(from_tag, to_tag) + p.suffix)


def _load_with_sibling(path: str) -> tuple[dict | None, dict | None]:
    """Loads whichever of (pipeline, output) the chosen file actually is,
    then tries to also load its sibling (by the _pipeline/_output naming
    convention) so a Full Save's pair gets restored together from
    picking just one of the two files."""
    pipeline_data = output_data = None
    try:
        pipeline_data = session_io.load_pipeline(path)
    except KeyError:
        pass

    if pipeline_data is not None:
        sibling = _sibling_path(path, "_pipeline", "_output")
        if sibling is not None and sibling.exists():
            output_data = session_io.load_output(str(sibling))
    else:
        try:
            output_data = session_io.load_output(path)
        except KeyError:
            pass
        sibling = _sibling_path(path, "_output", "_pipeline")
        if sibling is not None and sibling.exists():
            pipeline_data = session_io.load_pipeline(str(sibling))

    return pipeline_data, output_data


class SaveTab(QWidget):
    session_loaded = Signal(dict)  # {"pipeline": dict|None, "output": dict|None}

    def __init__(self, state: AppState, roi_validation_tab: ROIValidationTab, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.roi_validation_tab = roi_validation_tab

        layout = QVBoxLayout(self)

        self.save_pipeline_btn = QPushButton("Save Pipeline...")
        self.save_pipeline_btn.setToolTip(
            "Saves the reproducibility recipe only: ordered steps and their parameters, "
            "correlation-based ROIs' seed location/threshold in the order added, and ROI "
            "Validation's last-used parameters. No movie or ROI mask/trace data."
        )
        self.save_pipeline_btn.clicked.connect(self._on_save_pipeline_clicked)
        layout.addWidget(self.save_pipeline_btn)

        self.full_save_btn = QPushButton("Full Save (Pipeline + Output)...")
        self.full_save_btn.setToolTip(
            "Saves the pipeline recipe AND the actual results: ROI masks/traces/spike traces/"
            "neuropil traces, each committed stage's QC metrics, and ROI Validation's actual "
            "classification and transient times. Two files, written together."
        )
        self.full_save_btn.clicked.connect(self._on_full_save_clicked)
        layout.addWidget(self.full_save_btn)

        self.load_btn = QPushButton("Load Session...")
        self.load_btn.clicked.connect(self._on_load_session_clicked)
        layout.addWidget(self.load_btn)

        self.status_label = QLabel(
            "Movies are never saved -- a session re-associates with the original file by its "
            "recorded path when loaded."
        )
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    def _on_save_pipeline_clicked(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Pipeline", "", _FILE_FILTER)
        if not path:
            return
        session_io.save_pipeline(self.state, self.roi_validation_tab.export_pipeline_params(), path)
        self.status_label.setText(f"Saved pipeline to {path}")

    def _on_full_save_clicked(self) -> None:
        base_path, _ = QFileDialog.getSaveFileName(self, "Full Save (Pipeline + Output)", "", _FILE_FILTER)
        if not base_path:
            return
        pipeline_path, output_path = _derive_paths(base_path)
        session_io.save_pipeline(self.state, self.roi_validation_tab.export_pipeline_params(), pipeline_path)
        session_io.save_output(self.state, self.roi_validation_tab.export_results(), output_path)
        self.status_label.setText(f"Saved pipeline to {pipeline_path} and output to {output_path}")

    def _on_load_session_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Session", "", _FILE_FILTER)
        if not path:
            return
        try:
            pipeline_data, output_data = _load_with_sibling(path)
        except OSError as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        if pipeline_data is None and output_data is None:
            QMessageBox.critical(self, "Load failed", "That file doesn't look like a saved pipeline or output session.")
            return

        self.session_loaded.emit({"pipeline": pipeline_data, "output": output_data})

        parts = []
        if pipeline_data is not None:
            parts.append("pipeline")
        if output_data is not None:
            parts.append("output")
        self.status_label.setText(f"Loaded {' + '.join(parts)} from {path}")
