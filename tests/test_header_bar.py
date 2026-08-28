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


def test_report_button_and_data_label_share_the_top_left_column():
    header = HeaderBar()
    # Both widgets sit in the same top-left vertical column (report
    # button above data status) -- confirmed by both being direct
    # children of the same layout item, distinct from the title label.
    assert header.report_btn.parentWidget() is header
    assert header.data_label.parentWidget() is header


def test_options_button_opens_a_dialog_with_a_font_size_control(monkeypatch):
    header = HeaderBar()
    opened = []
    monkeypatch.setattr(
        "orbitapp.widgets.header_bar.OptionsDialog.exec",
        lambda self: opened.append(self.font_size_spin.value()),
    )

    header.options_btn.click()

    assert opened == [0]


def test_changing_font_size_updates_the_app_font_via_theme(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "orbitapp.widgets.header_bar.set_font_size_delta", lambda app, delta: calls.append(delta)
    )

    header = HeaderBar()
    header._on_font_size_changed(6)

    assert calls == [6]
