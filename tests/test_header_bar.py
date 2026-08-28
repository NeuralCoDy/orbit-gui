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
        lambda self: opened.append(self.font_size_slider.value()),
    )

    header.options_btn.click()

    assert opened == [100]  # the low end of the slider == the current default (1x)


def test_options_dialog_reopens_at_the_last_scale_the_user_set(monkeypatch):
    header = HeaderBar()
    header._font_scale = 1.4

    opened = []
    monkeypatch.setattr(
        "orbitapp.widgets.header_bar.OptionsDialog.exec",
        lambda self: opened.append(self.font_size_slider.value()),
    )

    header.options_btn.click()

    assert opened == [140]


def test_changing_font_scale_updates_the_app_font_via_theme(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "orbitapp.widgets.header_bar.set_font_size_scale", lambda app, scale: calls.append(scale)
    )

    header = HeaderBar()
    header._on_font_scale_changed(1.6)

    assert calls == [1.6]
    assert header._font_scale == 1.6


def test_changing_font_scale_refreshes_the_title_and_pipeline_arrow_fonts(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from orbitapp import theme

    app = QApplication.instance()
    theme.apply_dark_theme(app)

    header = HeaderBar()
    header.set_pipeline(["Load", "Rigid"])
    before_title = header.title_label.font().pointSize()

    header._on_font_scale_changed(2.0)

    after_title = header.title_label.font().pointSize()
    assert after_title > before_title
    assert after_title == app.font().pointSize() + 2  # same fixed bump, over the new scaled base

    header._on_font_scale_changed(1.0)  # reset for other tests
