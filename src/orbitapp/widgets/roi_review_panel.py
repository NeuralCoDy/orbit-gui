"""Shared review panel for Source Extraction, mirroring roiapp's own
click-a-preview / review-a-collection workflow -- with one twist: the
"field of view" view (left) does double duty. Clicking it always grows a
new ROI's *preview* (not saved anywhere yet); selecting an existing ROI
instead -- via the "Current ROIs" view (middle) or the table (right) --
switches that same panel to highlighting just that ROI's own mask plus
the ring of pixels feeding its neuropil trace, so its shape is easy to
inspect regardless of how many other ROIs are in the collection. The
table lists the current collection PLUS the current preview (shown as
"ROI -1" until it's added) with per-row/bulk accept, reject, and delete,
plus two stacked, x-linked plots for whichever row is selected: Trace -
Neuropil on top, the raw Trace/Neuropil/Spikes overlay below -- selecting
the preview row shows both immediately, same as any added ROI. Used
identically by every extraction method (interactive click, PCA-ICA,
CNMF) -- see SourceExtractionTab's module docstring for how each
method's results reach this same collection.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from orbit.neuropil import neuropil_ring_mask

from ..state import ROI
from .image_coords import scene_pos_to_pixel
from .roi_overlay import render_roi_overlay

_COLUMNS = ("ID", "Method", "Status", "Area (px)")
# Fixed (not per-ROI-cycling) colors for the FOV panel's highlight, so
# they read consistently regardless of which ROI is picked: green while
# it's still an unadded preview ("ROI -1", matching roiapp's own
# in-progress-mask color), cyan once it's a real, added candidate.
_PREVIEW_RGBA = (0, 255, 0, 160)
_SELECTED_ROI_RGBA = (0, 255, 255, 220)
_NEUROPIL_RGBA = (255, 140, 0, 170)  # orange, either way


def _project_mask_2d(mask: np.ndarray) -> np.ndarray:
    """A 2D silhouette of a >2D boolean mask -- "any voxel present along
    the trailing axes" (e.g. depth-collapsing a volumetric ROI's
    (L, W, D) mask), unchanged for an already-2D mask. This panel's
    rendering is inherently 2D (an image view); the projection is only
    ever used to build a display image, never to mutate the ROI's own
    mask, which stays the real (possibly 3D) array everywhere else."""
    return mask if mask.ndim == 2 else mask.any(axis=tuple(range(2, mask.ndim)))


def _for_display(roi: ROI) -> ROI:
    """A throwaway display copy with a 2D mask -- safe to build fresh on
    every render (unlike self._rois itself, which must stay the same
    list/objects SourceExtractionTab's _candidates holds, so accept/
    reject/delete keep working via that shared identity)."""
    return roi if roi.mask.ndim == 2 else replace(roi, mask=_project_mask_2d(roi.mask))


class ROIReviewPanel(QWidget):
    roi_status_changed = Signal(int, str)  # roi_id, new_status
    roi_selected = Signal(int)  # roi_id
    roi_deleted = Signal(int)  # roi_id
    pixel_clicked = Signal(int, int)  # row, col -- any click on the FOV (left) view

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rois: list[ROI] = []
        self._preview_roi: ROI | None = None
        self._selected_roi_id: int | None = None

        layout = QHBoxLayout(self)

        images_row = QHBoxLayout()

        fov_col = QVBoxLayout()
        fov_col.addWidget(QLabel("Field of view -- click to seed a new ROI, or select one to inspect it here"))
        self.fov_view = pg.ImageView()
        self._highlight_item = pg.ImageItem()
        self.fov_view.getView().addItem(self._highlight_item)
        self.fov_view.getView().scene().sigMouseClicked.connect(self._on_fov_clicked)
        fov_col.addWidget(self.fov_view)
        images_row.addLayout(fov_col)

        collection_col = QVBoxLayout()
        collection_col.addWidget(QLabel("Current ROIs (click one to select it)"))
        self.collection_view = pg.ImageView()
        self._overlay_item = pg.ImageItem()
        self.collection_view.getView().addItem(self._overlay_item)
        self.collection_view.getView().scene().sigMouseClicked.connect(self._on_collection_clicked)
        collection_col.addWidget(self.collection_view)
        images_row.addLayout(collection_col)

        layout.addLayout(images_row, stretch=3)

        table_col = QVBoxLayout()
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        table_col.addWidget(self.table)

        bulk_row = QHBoxLayout()
        accept_remainder_btn = QPushButton("Accept Remainder")
        accept_remainder_btn.setToolTip("Accepts every still-pending candidate -- already accepted/rejected ones are left as they are.")
        accept_remainder_btn.clicked.connect(lambda: self._set_pending_status("accepted"))
        reject_remainder_btn = QPushButton("Reject Remainder")
        reject_remainder_btn.setToolTip("Rejects every still-pending candidate -- already accepted/rejected ones are left as they are.")
        reject_remainder_btn.clicked.connect(lambda: self._set_pending_status("rejected"))
        delete_rejected_btn = QPushButton("Delete All Rejected")
        delete_rejected_btn.clicked.connect(self._delete_all_rejected)
        bulk_row.addWidget(accept_remainder_btn)
        bulk_row.addWidget(reject_remainder_btn)
        bulk_row.addWidget(delete_rejected_btn)
        table_col.addLayout(bulk_row)

        per_row = QHBoxLayout()
        self.accept_btn = QPushButton("Accept Selected")
        self.accept_btn.clicked.connect(lambda: self._set_selected_status("accepted"))
        self.reject_btn = QPushButton("Reject Selected")
        self.reject_btn.clicked.connect(lambda: self._set_selected_status("rejected"))
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self._delete_selected)
        per_row.addWidget(self.accept_btn)
        per_row.addWidget(self.reject_btn)
        per_row.addWidget(self.delete_btn)
        table_col.addLayout(per_row)

        self.diff_plot = pg.PlotWidget()
        self.diff_plot.setTitle("Trace - Neuropil")
        self.diff_plot.setLabel("left", "intensity")
        table_col.addWidget(self.diff_plot, stretch=1)

        self.trace_plot = pg.PlotWidget()
        self.trace_plot.addLegend()
        self.trace_plot.setLabel("bottom", "frame")
        self.trace_plot.setLabel("left", "intensity")
        table_col.addWidget(self.trace_plot, stretch=1)

        # Same x-axis (frame) for both -- panning/zooming one keeps them
        # in visual sync, matching the difference trace to whichever
        # stretch of the raw trace/neuropil it came from.
        self.diff_plot.setXLink(self.trace_plot)

        layout.addLayout(table_col, stretch=1)

    def set_base_image(self, image: np.ndarray) -> None:
        self.fov_view.setImage(image)
        self.collection_view.setImage(image)

    def set_preview_roi(self, roi: ROI | None) -> None:
        """The not-yet-added preview, listed in the table as "ROI -1" and
        highlighted in the FOV panel immediately -- distinct from
        set_candidates, which is the real, added collection shown in the
        Current ROIs panel's overlay."""
        self._preview_roi = roi
        self._refresh_table()
        if roi is not None:
            self.table.selectRow(len(self._rois))  # preview is always the last row
        self._refresh_selected_view(roi)

    def set_candidates(self, rois: list[ROI]) -> None:
        self._rois = rois
        self._refresh_table()
        self._refresh_overlay()
        self._refresh_selected_view(self._resolve_selected())

    def _all_rows(self) -> list[ROI]:
        return self._rois + ([self._preview_roi] if self._preview_roi is not None else [])

    def _resolve_selected(self) -> ROI | None:
        """The currently-highlighted ROI, re-looked-up by id -- used after
        the underlying collection changes (add/delete/recompute) so a
        still-selected ROI's neuropil ring stays fresh, and a since-
        deleted one's highlight clears."""
        if self._selected_roi_id is None:
            return None
        return next((roi for roi in self._all_rows() if roi.id == self._selected_roi_id), None)

    def _refresh_table(self) -> None:
        rows = self._all_rows()
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for row, roi in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(str(roi.id)))
            self.table.setItem(row, 1, QTableWidgetItem(roi.source_method))
            self.table.setItem(row, 2, QTableWidgetItem(roi.status))
            self.table.setItem(row, 3, QTableWidgetItem(str(int(roi.mask.sum()))))
        self.table.blockSignals(False)

    def _refresh_overlay(self) -> None:
        if not self._rois:
            self._overlay_item.clear()
            return
        display_rois = [_for_display(roi) for roi in self._rois]
        overlay = render_roi_overlay(display_rois, display_rois[0].mask.shape)
        if overlay is None:
            self._overlay_item.clear()
        else:
            self._overlay_item.setImage(overlay)

    def _selected_roi_overlay(self, roi: ROI) -> np.ndarray:
        """RGBA composite for the FOV panel's highlight: the ring of
        pixels used for ``roi``'s neuropil trace (same exclusion rule as
        orbit.neuropil.compute_neuropil_traces -- every other known ROI's
        pixels removed) underneath the ROI's own mask."""
        others = [_project_mask_2d(r.mask) for r in self._rois if r.id != roi.id]
        exclusion = np.logical_or.reduce(others) if others else None
        display_roi = _for_display(roi)
        ring = neuropil_ring_mask(display_roi.mask, exclusion_mask=exclusion)

        rgba = np.zeros((*display_roi.mask.shape, 4), dtype=np.uint8)
        rgba[ring] = _NEUROPIL_RGBA
        rgba[display_roi.mask] = _PREVIEW_RGBA if display_roi.status == "preview" else _SELECTED_ROI_RGBA
        return rgba

    def _refresh_selected_view(self, roi: ROI | None) -> None:
        self._selected_roi_id = roi.id if roi is not None else None
        if roi is None or not roi.mask.any():
            self._highlight_item.clear()
            return
        self._highlight_item.setImage(self._selected_roi_overlay(roi))

    def _click_pixel(self, view: pg.ImageView, event) -> tuple[int, int]:
        return scene_pos_to_pixel(view.getImageItem(), view.getView(), event.scenePos())

    def _on_fov_clicked(self, event) -> None:
        row, col = self._click_pixel(self.fov_view, event)
        self.pixel_clicked.emit(row, col)

    def _on_collection_clicked(self, event) -> None:
        row, col = self._click_pixel(self.collection_view, event)
        idx = self._find_roi_at_pixel(row, col)
        if idx is not None:
            self.table.selectRow(idx)

    def _find_roi_at_pixel(self, row: int, col: int) -> int | None:
        for i, roi in enumerate(self._rois):
            mask = _project_mask_2d(roi.mask)
            h, w = mask.shape
            if 0 <= row < h and 0 <= col < w and mask[row, col]:
                return i
        return None

    def _selected_ids(self) -> list[int]:
        """IDs of selected REAL (added) candidates only -- the preview row,
        if selected, is deliberately excluded here since Accept/Reject/
        Delete don't apply to something not yet added."""
        rows = {index.row() for index in self.table.selectedIndexes()}
        return [self._rois[row].id for row in rows if row < len(self._rois)]

    def _find(self, roi_id: int) -> ROI | None:
        return next((roi for roi in self._rois if roi.id == roi_id), None)

    def _set_status(self, roi_id: int, status: str) -> None:
        """Mutates one ROI's status and emits the change signal, but does
        NOT refresh the table/overlay -- callers that set several statuses
        at once (Accept Remainder, Reject Remainder) do that ONCE
        afterward instead of once per ROI, since _refresh_overlay
        recomposites every ROI's mask and doing that inside this per-ROI
        loop is quadratic in the collection size."""
        roi = self._find(roi_id)
        if roi is None:
            return
        roi.status = status
        self.roi_status_changed.emit(roi_id, status)

    def _apply_status(self, roi_ids: list[int], status: str) -> None:
        for roi_id in roi_ids:
            self._set_status(roi_id, status)
        self._refresh_table()
        self._refresh_overlay()

    def _set_selected_status(self, status: str) -> None:
        self._apply_status(self._selected_ids(), status)

    def _set_pending_status(self, status: str) -> None:
        """Accept/Reject Remainder: only ever touches still-"pending"
        candidates -- already accepted or rejected ones keep whatever
        status was explicitly given them, so this is safe to click again
        after reviewing some ROIs by hand without undoing that work."""
        self._apply_status([roi.id for roi in self._rois if roi.status == "pending"], status)

    def _delete_ids(self, ids: list[int]) -> None:
        """Removes the given ROI ids from the collection entirely --
        shared by Delete Selected and Delete All Rejected. Mutates
        self._rois in place, which is the same list object
        SourceExtractionTab's _candidates holds, so both stay in sync
        without a round-trip through set_candidates."""
        if not ids:
            return
        self._rois[:] = [roi for roi in self._rois if roi.id not in ids]
        for roi_id in ids:
            self.roi_deleted.emit(roi_id)
        self._refresh_table()
        self._refresh_overlay()
        self.trace_plot.clear()
        self.diff_plot.clear()
        self._refresh_selected_view(self._resolve_selected())

    def _delete_selected(self) -> None:
        """Removes the selected ROI(s) from the collection entirely --
        distinct from Reject, which only marks status and keeps them
        around for reconsideration."""
        self._delete_ids(self._selected_ids())

    def _delete_all_rejected(self) -> None:
        self._delete_ids([roi.id for roi in self._rois if roi.status == "rejected"])

    def _on_selection_changed(self) -> None:
        """Uses raw row index (not _selected_ids, which excludes the
        preview row) so selecting "ROI -1" also plots its trace."""
        selected_rows = {index.row() for index in self.table.selectedIndexes()}
        if not selected_rows:
            return
        row = next(iter(selected_rows))
        all_rows = self._all_rows()
        if row >= len(all_rows):
            return
        roi = all_rows[row]
        self.roi_selected.emit(roi.id)
        self.trace_plot.clear()
        self.trace_plot.plot(roi.trace, pen="c", name="Trace")
        self.diff_plot.clear()
        if roi.neuropil_trace is not None:
            self.trace_plot.plot(roi.neuropil_trace, pen="y", name="Neuropil")
            diff = roi.trace - roi.neuropil_trace
            self.diff_plot.plot(diff, pen="g")
            # setXLink keeps the two views in sync on later pan/zoom, but
            # its initial auto-range sync can lag a frame behind -- set it
            # explicitly here so the diff plot reads correctly right away.
            self.diff_plot.setXRange(0, len(diff) - 1, padding=0)
        if roi.spike_trace is not None:
            self.trace_plot.plot(roi.spike_trace, pen="m", name="Spikes")
        self._refresh_selected_view(roi)
