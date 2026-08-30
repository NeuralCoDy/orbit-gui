import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp.widgets.step_params_dialog import StepParamsDialog  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def test_title_includes_the_step_label():
    dialog = StepParamsDialog("Rigid", {"max_shift": 15.0})
    assert dialog.windowTitle() == "Rigid parameters"


def test_shows_one_row_per_parameter():
    dialog = StepParamsDialog("Rigid", {"max_shift": 15.0, "n_iter": 2, "method": "Rigid"})
    assert dialog.layout().itemAt(0).layout().rowCount() == 3


def test_formats_float_values_with_reasonable_precision():
    dialog = StepParamsDialog("Denoise", {"threshold": 0.123456789})
    form = dialog.layout().itemAt(0).layout()
    value_label = form.itemAt(1).widget()
    assert value_label.text() == "0.1235"


def test_formats_bool_list_and_dict_values():
    dialog = StepParamsDialog(
        "Normalize", {"center": True, "shape": (10, 10), "extra": {"a": 1, "b": 2.5}}
    )
    form = dialog.layout().itemAt(0).layout()
    values = [form.itemAt(i).widget().text() for i in range(1, form.count(), 2)]
    assert values == ["True", "(10, 10)", "a=1; b=2.5"]


def test_shows_a_message_when_no_parameters_were_recorded():
    dialog = StepParamsDialog("Load", {})
    form = dialog.layout().itemAt(0).layout()
    assert form.rowCount() == 1
    assert "No parameters recorded" in form.itemAt(0).widget().text()


def test_close_button_accepts_the_dialog():
    dialog = StepParamsDialog("Rigid", {"max_shift": 15.0})
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(1))

    dialog.close_btn.click()

    assert accepted == [1]
