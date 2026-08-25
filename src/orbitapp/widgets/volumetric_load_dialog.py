"""Popup shown when loading a folder of TIFF files as volumetric (Load
tab's "Volumetric" toggle) data -- resolves an ambiguity a folder of
TIFFs has that a single 4D FITS file doesn't: whether each file is its
own timepoint's whole volume, or the files together form one
continuous stream of depth-slices to be chopped up (see
orbitapp.volumetric_io.load_volumetric_tiff_folder, which this dialog's
mode()/depth() feed directly).
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QRadioButton, QSpinBox, QVBoxLayout

from ..volumetric_io import INTERLEAVED, ONE_VOLUME_PER_STACK


class VolumetricLoadDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load Volumetric TIFF Folder")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("How should the TIFF files in this folder be interpreted?"))

        self.one_per_stack_radio = QRadioButton("One volume per TIFF stack")
        self.one_per_stack_radio.setToolTip(
            "Each TIFF file is one full volume at one time point -- length, width, and depth are "
            "read directly from each file's own multi-page stack shape (every file must share the "
            "same shape), and the number of time points is the number of files."
        )
        self.one_per_stack_radio.setChecked(True)
        layout.addWidget(self.one_per_stack_radio)

        self.interleaved_radio = QRadioButton("Interleaved volumes")
        self.interleaved_radio.setToolTip(
            "Every page across every file, in order, is one continuous stream of depth-slices -- "
            "every N consecutive slices form one volume, where N (below) is the number of slices "
            "per volume."
        )
        layout.addWidget(self.interleaved_radio)

        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 100000)
        self.depth_spin.setValue(1)
        self.depth_spin.setEnabled(False)
        self.depth_spin.setPrefix("slices per volume: ")
        layout.addWidget(self.depth_spin)

        self.interleaved_radio.toggled.connect(self.depth_spin.setEnabled)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def mode(self) -> str:
        return INTERLEAVED if self.interleaved_radio.isChecked() else ONE_VOLUME_PER_STACK

    def depth(self) -> int | None:
        return self.depth_spin.value() if self.interleaved_radio.isChecked() else None
