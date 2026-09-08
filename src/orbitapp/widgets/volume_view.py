"""Drag-rotatable 3D view of a volumetric movie's time projection --
CPU-rendered (orbit.volume_render), no OpenGL, so it works headless and
over X-forwarding. Used by the Data Projections tab for volumetric data
(the "Volume (3D)" entry).

Click-drag rotates (horizontal = azimuth, vertical = elevation). A
heavily downsampled copy renders during the drag for immediate
feedback; the full-resolution render (~0.5-1s) runs when the drag
settles or a transfer-function control changes. Rendering is
synchronous -- the full render is short enough that a worker thread's
lifetime hassle isn't worth it here.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from orbit.volume_render import downsample_volume, normalize_volume, render_volume

_FULL_MAX_AXIS = 180
_PREVIEW_MAX_AXIS = 64
_DRAG_SENSITIVITY = 0.4  # degrees of rotation per pixel dragged
_DEFAULT_AZ, _DEFAULT_EL = 30.0, 20.0


def _colormapped(scalar: np.ndarray, cmap_name: str) -> QImage:
    lut = pg.colormap.get(cmap_name).getLookupTable(nPts=256, alpha=False)  # (256, 3) uint8
    idx = np.clip(scalar * 255.0, 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(lut[idx])  # (H, W, 3)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class VolumeView(QWidget):
    def __init__(self, parent=None, placeholder: str = "Load volumetric data to see the 3D view.") -> None:
        super().__init__(parent)
        self._placeholder = placeholder
        self._full: np.ndarray | None = None
        self._preview: np.ndarray | None = None
        self._pending_movie: np.ndarray | None = None  # set but not yet prepped (widget not visible)
        self._az, self._el = _DEFAULT_AZ, _DEFAULT_EL
        self._drag_origin: tuple[int, int, float, float] | None = None
        self._last_scalar: np.ndarray | None = None

        self.image = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(320, 320)
        self.image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image.setStyleSheet("background:#000;")
        self.image.installEventFilter(self)

        self.mode = QComboBox()
        self.mode.addItems(["emission-absorption", "MIP"])
        self.gamma = self._spin(0.2, 4.0, 1.6, 0.1)
        self.density = self._spin(0.2, 8.0, 3.0, 0.2)
        self.cmap = QComboBox()
        self.cmap.addItems(["inferno", "magma", "viridis", "gray"])
        reset = QPushButton("Reset view")
        reset.clicked.connect(self._reset_view)

        controls = QHBoxLayout()
        for label, w in (("mode", self.mode), ("gamma", self.gamma), ("density", self.density),
                         ("colormap", self.cmap)):
            controls.addWidget(QLabel(label))
            controls.addWidget(w)
        controls.addWidget(reset)
        controls.addStretch()

        self.hint = QLabel(self._placeholder)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.hint)

        self.mode.currentIndexChanged.connect(self._on_mode_changed)
        self.gamma.valueChanged.connect(self._render_full)
        self.density.valueChanged.connect(self._render_full)
        self.cmap.currentIndexChanged.connect(self._recolor)
        self._on_mode_changed()

    # -- public --------------------------------------------------------

    def set_volume(self, movie: np.ndarray) -> None:
        """``movie``: a volumetric (T, L, W, D) array (mean-projected
        over time here) or an already-3D (L, W, D) volume.

        The mean + downsample (a few seconds on a real movie) is
        deferred until the widget is actually visible -- several stage
        tabs each hold a VolumeView and all get set_volume on data load,
        but only the one the user is looking at needs to prep. See
        showEvent / _consume_pending."""
        self._pending_movie = movie
        if self.isVisible():
            self._consume_pending()

    def _consume_pending(self) -> None:
        movie, self._pending_movie = self._pending_movie, None
        if movie is None:
            return
        self.hint.setText("Preparing 3D view...")
        # Downsample in the native (L, W, D) layout -- C-contiguous --
        # *before* the moveaxis to (D, L, W); doing it after forces a
        # strided read of the whole full-res volume (~17s on a
        # 150x3200x530 movie vs a fraction of a second this way).
        lwd = movie.mean(axis=0, dtype=np.float32) if movie.ndim == 4 else np.asarray(movie, dtype=np.float32)
        lwd = downsample_volume(lwd, _FULL_MAX_AXIS)
        self._full = normalize_volume(np.moveaxis(lwd, -1, 0))  # (D', L', W'): depth is the view axis
        self._preview = downsample_volume(self._full, _PREVIEW_MAX_AXIS)
        self._az, self._el = _DEFAULT_AZ, _DEFAULT_EL
        self.hint.setText("Click-drag to rotate  ·  horizontal = azimuth, vertical = elevation")
        self._render_full()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if self._pending_movie is not None:
            self._consume_pending()

    def clear(self) -> None:
        self._full = self._preview = self._last_scalar = self._pending_movie = None
        self.image.clear()
        self.hint.setText(self._placeholder)

    # -- rotation via drag --------------------------------------------

    def eventFilter(self, obj, event):  # noqa: N802 (Qt signature)
        if obj is self.image and self._full is not None:
            et = event.type()
            if et == event.Type.MouseButtonPress:
                p = event.position().toPoint()
                self._drag_origin = (p.x(), p.y(), self._az, self._el)
                return True
            if et == event.Type.MouseMove and self._drag_origin is not None:
                x0, y0, az0, el0 = self._drag_origin
                p = event.position().toPoint()
                self._az = az0 + (p.x() - x0) * _DRAG_SENSITIVITY
                self._el = float(np.clip(el0 - (p.y() - y0) * _DRAG_SENSITIVITY, -89.0, 89.0))
                self._render_preview()
                return True
            if et == event.Type.MouseButtonRelease and self._drag_origin is not None:
                self._drag_origin = None
                self._render_full()
                return True
        return super().eventFilter(obj, event)

    def _reset_view(self) -> None:
        self._az, self._el = _DEFAULT_AZ, _DEFAULT_EL
        self._render_full()

    # -- rendering ---------------------------------------------------

    def _mode_str(self) -> str:
        return "mip" if self.mode.currentText() == "MIP" else "composite"

    def _on_mode_changed(self, *_) -> None:
        composite = self._mode_str() == "composite"
        self.gamma.setEnabled(composite)
        self.density.setEnabled(composite)
        self._render_full()

    def _render_preview(self) -> None:
        if self._preview is not None:
            self._show(self._render(self._preview, order=0))

    def _render_full(self, *_) -> None:
        if self._full is not None:
            self._show(self._render(self._full, order=1))

    def _render(self, vol: np.ndarray, order: int) -> np.ndarray:
        return render_volume(
            vol, self._az, self._el,
            gamma=self.gamma.value(), density=self.density.value(), mode=self._mode_str(), order=order,
        )

    def _recolor(self, *_) -> None:
        if self._last_scalar is not None:
            self._show(self._last_scalar)

    def _show(self, scalar: np.ndarray) -> None:
        self._last_scalar = scalar
        pix = QPixmap.fromImage(_colormapped(scalar, self.cmap.currentText()))
        self.image.setPixmap(
            pix.scaled(self.image.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )

    def _spin(self, lo, hi, val, step) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        s.setKeyboardTracking(False)
        return s

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self._last_scalar is not None:
            self._show(self._last_scalar)
