import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from orbitapp.widgets.header_bar import HeaderBar  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QApplication.instance() or QApplication([])


def test_generate_report_clicked_fires_on_button_click():
    header = HeaderBar()
    clicked = []
    header.generate_report_clicked.connect(lambda: clicked.append(1))

    header.report_btn.click()

    assert clicked == [1]


def test_set_report_busy_disables_and_relabels_the_button():
    header = HeaderBar()
    assert header.report_btn.isEnabled()
    assert header.report_btn.text() == "Generate Report..."

    header.set_report_busy(True)
    assert not header.report_btn.isEnabled()
    assert header.report_btn.text() == "Generating..."

    header.set_report_busy(False)
    assert header.report_btn.isEnabled()
    assert header.report_btn.text() == "Generate Report..."
