"""Reusable layout for pipeline-stage tabs: before/after images on top,
a metrics row (numbers/plots) underneath -- metrics get the full window
width rather than being squeezed into a sidebar. Every stage tab should
embed a StagePanel rather than rolling its own arrangement.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


class StagePanel(QWidget):
    def __init__(self, before_title: str = "Before", after_title: str = "After", parent=None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)

        images_row = QHBoxLayout()
        self.before_view = pg.ImageView()
        self.after_view = pg.ImageView()
        for view, title in ((self.before_view, before_title), (self.after_view, after_title)):
            col = QVBoxLayout()
            col.addWidget(QLabel(title))
            col.addWidget(view)
            images_row.addLayout(col)
        images_container = QWidget()
        images_container.setLayout(images_row)
        layout.addWidget(images_container, stretch=2)

        self.metrics_row = QHBoxLayout()
        metrics_container = QWidget()
        metrics_container.setLayout(self.metrics_row)
        layout.addWidget(metrics_container, stretch=1)

    def add_metric_widget(self, widget: QWidget) -> None:
        """Add one metric/plot widget to the row below the images."""
        self.metrics_row.addWidget(widget)
