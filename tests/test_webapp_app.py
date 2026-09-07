from __future__ import annotations

from subprocess import CompletedProcess

import pytest

from envoi_webapp import app
from envoi_webapp.app import (
    ALL_DATASET_TYPES,
    DATASET_CATALOG_URL,
    _apply_pending_dataset_remove,
    _choose_output_directory,
    _dataset_display_name,
    _dataset_names_for_type,
    _dataset_option_label,
    _dataset_type,
    _escape_applescript_string,
    _escape_powershell_string,
    _ordered_dataset_types,
    _render_validation_issues,
    _type_option_label,
    _validate_dataset_rows,
    _validate_form,
)
from envoi_webapp.helpers import TABULAR_OUTPUT


class _FakeStreamlit:
    def __init__(self, session_state):
        self.session_state = session_state


class _FakeValidationStreamlit:
    def __init__(self):
        self.errors = []
        self.markdowns = []

    def error(self, message):
        self.errors.append(message)

    def markdown(self, body, **kwargs):
        self.markdowns.append((body, kwargs))


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _has_dataset_widget_keys(session_state) -> bool:
    return any(
        key.startswith(
            (
                "dataset_select_",
                "dataset_type_select_",
                "point_checkbox_",
                "window_checkbox_",
                "windows_input_",
                "stats_select_",
                "remove_dataset_button_",
                "windows_",
                "stats_",
            )
        )
        or (key.startswith("dataset_") and key[8:].isdigit())
        for key in session_state
    )


def test_escape_applescript_string_escapes_quotes_and_backslashes():
    assert _escape_applescript_string('/tmp/a "quoted" folder\\name') == (
        '/tmp/a \\"quoted\\" folder\\\\name'
    )


def test_escape_powershell_string_escapes_single_quotes():
    assert _escape_powershell_string("C:\\Users\\O'Brien") == "C:\\Users\\O''Brien"


def test_dataset_catalog_url_targets_formatted_catalog_on_main():
    assert DATASET_CATALOG_URL == (
        "https://github.com/BiodiversityDataLab/envoi/blob/main/docs/datasets.md"
    )


def test_dataset_display_name_uses_catalog_label_and_falls_back_to_key():
    catalog = {
        "dem_copernicus_glo30": {"display_name": "Copernicus DEM GLO-30"},
        "custom_raster": {"data_source": "local"},
        "blank_label": {"display_name": "  "},
    }

    assert _dataset_display_name("dem_copernicus_glo30", catalog) == "Copernicus DEM GLO-30"
    assert _dataset_display_name("custom_raster", catalog) == "custom_raster"
    assert _dataset_display_name("blank_label", catalog) == "blank_label"


def test_dataset_types_follow_documented_order_with_custom_and_missing_last():
    catalog = {
        "roads": {"category": "Human impact"},
        "custom": {"category": "A custom type"},
        "dem": {"category": "Terrain"},
        "uncategorised": {},
        "era5": {"category": "Climate"},
    }

    assert _ordered_dataset_types(catalog) == [
        "Terrain",
        "Climate",
        "Human impact",
        "A custom type",
        "Uncategorised",
    ]
    assert _dataset_type("uncategorised", catalog) == "Uncategorised"


def test_dataset_names_are_grouped_by_type_then_sorted_by_display_name():
    catalog = {
        "climate_z": {"display_name": "Zulu", "category": "Climate"},
        "terrain_b": {"display_name": "Beta", "category": "Terrain"},
        "terrain_a": {"display_name": "Alpha", "category": "Terrain"},
        "climate_a": {"display_name": "Alpha", "category": "Climate"},
    }

    assert _dataset_names_for_type(catalog, ALL_DATASET_TYPES) == [
        "terrain_a",
        "terrain_b",
        "climate_a",
        "climate_z",
    ]
    assert _dataset_names_for_type(catalog, "Climate") == [
        "climate_a",
        "climate_z",
    ]


def test_type_and_dataset_options_use_counts_and_contextual_labels():
    catalog = {
        "era5": {"display_name": "ERA5 Monthly", "category": "Climate"},
        "worldclim": {"display_name": "WorldClim BIO", "category": "Climate"},
    }

    assert _type_option_label(ALL_DATASET_TYPES, catalog) == "All types (2)"
    assert _type_option_label("Climate", catalog) == "Climate (2)"
    assert _dataset_option_label("era5", catalog, include_type=True) == ("Climate · ERA5 Monthly")
    assert _dataset_option_label("era5", catalog, include_type=False) == "ERA5 Monthly"


def test_validate_form_collects_errors_in_step_and_field_order():
    validation = _validate_form(
        points_df=None,
        points_error=None,
        input_crs="",
        credentials_bytes=None,
        output_type=None,
        output_dir="",
        dataset_rows=[],
        catalog={},
        widget_version=0,
    )

    assert [issue.message for issue in validation.issues] == [
        "Step 1 — Upload a valid location CSV.",
        "Step 1 — Enter the EPSG code for the uploaded location data.",
        "Step 2 — Upload an Earth Engine service-account JSON key.",
        "Step 3 — Choose between tabular or raster output.",
        "Step 3 — Enter an output directory.",
    ]


def test_validate_dataset_rows_names_products_and_checks_stats_before_windows():
    rows = [
        {
            "dataset": "dem_copernicus_glo30",
            "sample_point": False,
            "sample_window": True,
            "statistics": [],
            "window_sizes": "",
        },
        {
            "dataset": "",
            "sample_point": False,
            "sample_window": False,
            "statistics": [],
            "window_sizes": "",
        },
    ]
    catalog = {"dem_copernicus_glo30": {"display_name": "Copernicus DEM GLO-30"}}

    issues, selections = _validate_dataset_rows(rows, TABULAR_OUTPUT, catalog, 4)

    assert [issue.message for issue in issues] == [
        "Step 4 — Data product “Copernicus DEM GLO-30”: choose at least one spatial statistic.",
        "Step 4 — Data product “Copernicus DEM GLO-30”: enter at least one sampling-window size.",
        "Step 4 — Data product 2: choose a data product.",
    ]
    assert issues[0].widget_keys == ("stats_select_4_0_dem_copernicus_glo30",)
    assert issues[1].widget_keys == ("windows_input_4_0",)
    assert issues[2].widget_keys == ("dataset_select_4_1",)
    assert selections == []


def test_validate_dataset_rows_does_not_validate_sampling_for_blank_product():
    issues, selections = _validate_dataset_rows(
        [{"dataset": "", "sample_point": False, "sample_window": False}],
        TABULAR_OUTPUT,
        {},
        0,
    )

    assert [issue.message for issue in issues] == [
        "Step 4 — Data product 1: choose a data product."
    ]
    assert selections == []


def test_validate_dataset_rows_names_missing_point_or_window_choice():
    issues, selections = _validate_dataset_rows(
        [
            {
                "dataset": "dem",
                "sample_point": False,
                "sample_window": False,
            }
        ],
        TABULAR_OUTPUT,
        {"dem": {"display_name": "Elevation"}},
        2,
    )

    assert [issue.message for issue in issues] == [
        "Step 4 — Data product “Elevation”: choose Point, Window, or both."
    ]
    assert issues[0].widget_keys == ("point_checkbox_2_0", "window_checkbox_2_0")
    assert selections == []


def test_render_validation_issues_combines_messages_and_targets_widgets():
    st = _FakeValidationStreamlit()
    issues = (
        app._ValidationIssue("Step 1 — Upload a CSV.", ("location_csv",)),
        app._ValidationIssue("Step 2 — Upload a key.", ("credentials_json",)),
    )

    _render_validation_issues(st, issues)

    assert st.errors == [
        "Please fix the following:\n\n- Step 1 — Upload a CSV.\n- Step 2 — Upload a key."
    ]
    css, kwargs = st.markdowns[0]
    assert "st-key-location_csv" in css
    assert "st-key-credentials_json" in css
    assert app.VALIDATION_ERROR_COLOR in css
    assert kwargs == {"unsafe_allow_html": True}


def test_choose_output_directory_supports_windows(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(app.sys, "platform", "win32")
    monkeypatch.setattr(
        app.shutil,
        "which",
        lambda executable: (
            "C:\\Windows\\powershell.exe" if executable == "powershell.exe" else None
        ),
    )

    def fake_run(command, **kwargs):
        calls.append(command)
        return CompletedProcess(command, 0, stdout="C:\\output\n", stderr="")

    monkeypatch.setattr(app.subprocess, "run", fake_run)

    assert _choose_output_directory(str(tmp_path)) == "C:\\output"
    assert calls[0][0] == "C:\\Windows\\powershell.exe"


def test_choose_output_directory_uses_available_linux_chooser(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(app.sys, "platform", "linux")
    monkeypatch.setattr(
        app.shutil,
        "which",
        lambda executable: "/usr/bin/kdialog" if executable == "kdialog" else None,
    )

    def fake_run(command, **kwargs):
        calls.append(command)
        return CompletedProcess(command, 0, stdout="/tmp/output\n", stderr="")

    monkeypatch.setattr(app.subprocess, "run", fake_run)

    assert _choose_output_directory(str(tmp_path)) == "/tmp/output"
    assert calls[0][0] == "/usr/bin/kdialog"


def test_choose_output_directory_falls_back_to_tk(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(app.sys, "platform", "freebsd")

    def fake_run(command, **kwargs):
        calls.append(command)
        return CompletedProcess(command, 0, stdout="/tmp/output\n", stderr="")

    monkeypatch.setattr(app.subprocess, "run", fake_run)

    assert _choose_output_directory(str(tmp_path)) == "/tmp/output"
    assert calls[0][:2] == [app.sys.executable, "-c"]


def test_main_sets_cross_browser_primary_theme_color(monkeypatch):
    from streamlit.web import cli as stcli

    monkeypatch.setattr(app.sys, "argv", ["envoi-webapp"])
    monkeypatch.setattr(stcli, "main", lambda: 0)

    with pytest.raises(SystemExit, match="0"):
        app.main()

    theme_index = app.sys.argv.index("--theme.primaryColor")
    assert app.sys.argv[theme_index + 1] == app.THEME_PRIMARY_COLOR


def test_apply_pending_dataset_remove_removes_only_requested_row_and_clears_widget_cache():
    session_state = _SessionState(
        {
            "dataset_rows": [
                {"dataset": "dem", "window_sizes": "100", "statistics": ["mean"]},
                {"dataset": "agb", "window_sizes": "200", "statistics": ["mean"]},
                {"dataset": "lulc", "window_sizes": "300", "statistics": ["mode"]},
            ],
            "_pending_dataset_remove": 1,
            "_dataset_widget_version": 4,
            "dataset_select_4_0": "dem",
            "dataset_type_select_4_0": ALL_DATASET_TYPES,
            "dataset_select_4_1": "agb",
            "dataset_select_4_2": "lulc",
            "windows_input_4_1": "200",
            "stats_select_4_1_agb": ["mean"],
            "remove_dataset_button_4_1": True,
            "dataset_0": "dem",
            "dataset_1": "agb",
            "dataset_2": "lulc",
            "windows_1": "200",
            "stats_1_agb": ["mean"],
        }
    )

    _apply_pending_dataset_remove(_FakeStreamlit(session_state))

    assert session_state["dataset_rows"] == [
        {"dataset": "dem", "window_sizes": "100", "statistics": ["mean"]},
        {"dataset": "lulc", "window_sizes": "300", "statistics": ["mode"]},
    ]
    assert "_pending_dataset_remove" not in session_state
    assert session_state["_dataset_widget_version"] == 5
    assert not _has_dataset_widget_keys(session_state)


def test_apply_pending_dataset_remove_removes_first_row_from_two_rows():
    session_state = _SessionState(
        {
            "dataset_rows": [
                {"dataset": "dem", "window_sizes": "100", "statistics": ["mean"]},
                {"dataset": "lulc", "window_sizes": "300", "statistics": ["mode"]},
            ],
            "_pending_dataset_remove": 0,
            "_dataset_widget_version": 2,
            "dataset_select_2_0": "dem",
            "dataset_select_2_1": "lulc",
            "windows_input_2_0": "100",
            "windows_input_2_1": "300",
            "stats_select_2_0_dem": ["mean"],
            "stats_select_2_1_lulc": ["mode"],
        }
    )

    _apply_pending_dataset_remove(_FakeStreamlit(session_state))

    assert session_state["dataset_rows"] == [
        {"dataset": "lulc", "window_sizes": "300", "statistics": ["mode"]},
    ]
    assert "_pending_dataset_remove" not in session_state
    assert session_state["_dataset_widget_version"] == 3
    assert not _has_dataset_widget_keys(session_state)


def test_apply_pending_dataset_remove_clears_final_row_and_widget_cache():
    session_state = _SessionState(
        {
            "dataset_rows": [
                {"dataset": "agb", "window_sizes": "200", "statistics": ["mean"]},
            ],
            "_pending_dataset_remove": 0,
            "_dataset_widget_version": 7,
            "dataset_select_7_0": "agb",
            "windows_input_7_0": "200",
            "stats_select_7_0_agb": ["mean"],
            "remove_dataset_button_7_0": True,
            "dataset_0": "agb",
            "windows_0": "200",
            "stats_0_agb": ["mean"],
        }
    )

    _apply_pending_dataset_remove(_FakeStreamlit(session_state))

    assert session_state["dataset_rows"] == [
        {
            "dataset": "",
            "sample_point": False,
            "sample_window": False,
            "window_sizes": "",
            "statistics": [],
        }
    ]
    assert "_pending_dataset_remove" not in session_state
    assert session_state["_dataset_widget_version"] == 8
    assert not _has_dataset_widget_keys(session_state)
