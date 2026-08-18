"""Shared asset paths (logo, ...) for the splash screen and header
branding -- one place to resolve `imgs/` relative to the repo root so
both call sites agree, rather than each re-deriving the path.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QPixmap

# src/orbitapp/assets.py -> orbitapp/ -> src/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
LOGO_PATH = _REPO_ROOT / "imgs" / "orbitlogoclean.png"

# The full logo is a tall portrait image (axon + soma + ring, a small
# protein-structure motif, and long dendrite branches below) -- too tall
# to read as a small corner icon. This crop keeps just the recognizable
# ring-around-a-soma "orbit" mark for header use; the splash screen uses
# the full image instead, where the portrait aspect works fine.
_HEADER_ICON_CROP = QRect(0, 550, 1320, 1050)


def load_header_icon(height_px: int) -> QPixmap:
    """The logo, cropped to its ring+soma mark and scaled to
    ``height_px`` tall -- for the header's corner branding."""
    full = QPixmap(str(LOGO_PATH))
    cropped = full.copy(_HEADER_ICON_CROP)
    return cropped.scaledToHeight(height_px, Qt.TransformationMode.SmoothTransformation)
