"""Read-only popup showing one committed pipeline step's recorded
parameters -- opened by clicking that step's box in the header's
pipeline diagram (see widgets/header_bar.py's _StepBox/_PipelineDiagram).

Deliberately separate from ParametersDialog (widgets/parameters_dialog.py):
that one holds LIVE, editable widgets for a stage tab's own not-yet-run
settings; this one is a plain, immutable snapshot of what a step
already committed with, so it has no editable widgets at all.
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QFormLayout, QLabel, QPushButton, QVBoxLayout


def _format_value(value: object) -> str:
    """Plain-text (not LaTeX-escaped, unlike report.py's own
    _format_value) rendering of one parameter value -- floats get a
    reasonable fixed precision instead of Python's full repr."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_format_value(v) for v in value) + ")"
    if isinstance(value, dict):
        return "; ".join(f"{k}={_format_value(v)}" for k, v in value.items())
    return str(value)


class StepParamsDialog(QDialog):
    def __init__(self, step_label: str, params: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{step_label} parameters")
        self.setMinimumWidth(280)

        layout = QVBoxLayout(self)

        form = QFormLayout()
        if params:
            for key, value in params.items():
                form.addRow(f"{key}:", QLabel(_format_value(value)))
        else:
            no_params_label = QLabel("No parameters recorded for this step.")
            no_params_label.setWordWrap(True)
            form.addRow(no_params_label)
        layout.addLayout(form)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        layout.addWidget(self.close_btn)
