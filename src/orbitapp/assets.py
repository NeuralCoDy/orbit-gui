"""Shared asset paths (logo, ...) for the splash screen and header
branding -- one place to resolve the packaged img/ dir so both call
sites agree, rather than each re-deriving the path.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap

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
