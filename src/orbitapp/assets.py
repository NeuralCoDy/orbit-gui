"""Shared asset paths (logo, ...) for the splash screen and header
branding -- one place to resolve the packaged img/ dir so both call
sites agree, rather than each re-deriving the path.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap

# Packaged alongside this module (src/orbitapp/img/) rather than a
# repo-root-relative imgs/ -- a path outside src/ is invisible to an
# actual `pip install orbit-gui` (only present in a dev/editable
# checkout), so the splash/logo silently vanished on any real install.
LOGO_PATH = Path(__file__).resolve().parent / "img" / "orbitlogoclean.png"


def load_logo(height_px: int) -> QPixmap:
    """The full logo image (background transparent, per the source PNG),
    scaled to ``height_px`` tall -- shared by the splash screen and the
    header's corner branding so both agree on how the image is loaded/
    scaled. Fine as-is in the header, which already sits on the dark
    theme's black background; see load_logo_on_black for the splash
    screen, which needs its own opaque backdrop."""
    return QPixmap(str(LOGO_PATH)).scaledToHeight(height_px, Qt.TransformationMode.SmoothTransformation)


def load_logo_on_black(height_px: int) -> QPixmap:
    """The logo composited onto an opaque black background, sized to
    exactly fit the (scaled) logo -- for the splash screen, whose own
    widget background would otherwise show through the logo's
    transparent areas instead of black."""
    logo = load_logo(height_px)
    canvas = QPixmap(logo.size())
    canvas.fill(Qt.GlobalColor.black)
    painter = QPainter(canvas)
    painter.drawPixmap(0, 0, logo)
    painter.end()
    return canvas


def load_app_icon(size_px: int = 1024) -> QIcon:
    """The logo centred on a square transparent canvas, as a QIcon --
    for QApplication.setWindowIcon, which is what macOS's Dock (and
    the window/taskbar icon elsewhere) shows for a bare `python -m
    orbitapp` process. The source PNG is portrait, so without the
    square padding the Dock would squash it to fit."""
    logo = QPixmap(str(LOGO_PATH)).scaled(
        size_px, size_px, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )
    canvas = QPixmap(size_px, size_px)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    painter.drawPixmap((size_px - logo.width()) // 2, (size_px - logo.height()) // 2, logo)
    painter.end()
    return QIcon(canvas)
