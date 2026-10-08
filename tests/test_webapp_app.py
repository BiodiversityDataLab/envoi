from __future__ import annotations

from contextlib import nullcontext
from subprocess import CompletedProcess

import pytest

from envoi_webapp import app
from envoi_webapp.app import (
    ALL_DATASET_TYPES,
    DATASET_CATALOG_URL,
    _apply_pending_dataset_remove,
    _apply_row_output_type,
    _choose_output_directory,
    _dataset_display_name,
    _dataset_names_for_type,
    _dataset_option_label,
    _dataset_type,
    _escape_applescript_string,
    _escape_powershell_string,
    _ordered_dataset_types,
    _progress_segments,
    _render_dataset_rows,
    _render_validation_issues,
    _type_option_label,
    _validate_dataset_rows,
    _validate_form,
)
from envoi_webapp.helpers import (
    RASTER_OUTPUT,
    TABULAR_OUTPUT,
    DatasetSelection,
    build_run_config,
)


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


class _FakeRowStreamlit:
    """Record the widgets of ``_render_dataset_rows`` and return session-state values.

    Like Streamlit, a keyed widget returns its value from ``session_state`` when
    the key is present (a value from an earlier run or a user change), and
    otherwise its default. Columns, expanders, and containers return this
    object, so all widgets are recorded in one place.
    """

    def __init__(self, session_state):
        self.session_state = session_state
        self.widgets: dict[str, dict] = {}
        self.captions: list[str] = []

    def expander(self, label, expanded=False):
        return nullcontext()

    def columns(self, spec, **kwargs):
        return [self] * (spec if isinstance(spec, int) else len(spec))

    def container(self, **kwargs):
        return self

    def caption(self, body):
        self.captions.append(body)

    def error(self, message):
        raise AssertionError(f"Unexpected error message: {message}")

    def rerun(self):
        raise AssertionError("Unexpected rerun")

    def _widget(self, key, default, **kwargs):
        self.widgets[key] = kwargs
        if key not in self.session_state:
            self.session_state[key] = default
        return self.session_state[key]

    def selectbox(self, label, options, index=0, key=None, **kwargs):
        default = options[index] if index is not None else None
        return self._widget(key, default, label=label, options=list(options), index=index)

    def checkbox(self, label, value=False, key=None, **kwargs):
        return self._widget(key, value, label=label)

    def multiselect(self, label, options, default=None, key=None, **kwargs):
        return self._widget(key, list(default or []), label=label, options=list(options))

    def text_input(self, label, value="", key=None, **kwargs):
        return self._widget(key, value, label=label)

    def button(self, label, key=None, **kwargs):
        if key is not None:
            self.widgets[key] = {"label": label, **kwargs}
        return False


_ROW_CATALOG = {
    "dem": {"display_name": "Elevation", "category": "Terrain", "data_type": "continuous"},
    "lulc": {"display_name": "Land cover", "category": "Land cover", "data_type": "categorical"},
}

_TABULAR_GUIDANCE_FRAGMENT = "Coordinate point values as well as spatial statistics"
_RASTER_GUIDANCE_FRAGMENT = "The window size(s) determines the size of the extracted raster tiles"


def _render_rows_with_fake(monkeypatch, session_state) -> _FakeRowStreamlit:
    fake_st = _FakeRowStreamlit(session_state)
    monkeypatch.setattr(app, "_load_streamlit", lambda: fake_st)
    _render_dataset_rows(fake_st, _ROW_CATALOG)
    return fake_st


def _has_dataset_widget_keys(session_state) -> bool:
    return any(
        key.startswith(
            (
                "output_type_select_",
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

    assert _type_option_label(ALL_DATASET_TYPES, catalog) == "All categories (2)"
    assert _type_option_label("Climate", catalog) == "Climate (2)"
    assert _dataset_option_label("era5", catalog, include_type=True) == ("Climate · ERA5 Monthly")
    assert _dataset_option_label("era5", catalog, include_type=False) == "ERA5 Monthly"


def test_validate_form_collects_errors_in_step_and_field_order():
    validation = _validate_form(
        points_df=None,
        points_error=None,
        input_crs="",
        credentials_bytes=None,
        output_dir="",
        dataset_rows=[],
        catalog={},
        widget_version=0,
    )

    assert [issue.message for issue in validation.issues] == [
        "Step 1 — Upload a valid location CSV.",
        "Step 1 — Enter the EPSG code for the uploaded location data.",
        "Step 2 — Upload an Earth Engine service-account JSON key.",
        "Step 3 — Enter an output directory.",
        "Step 4 — Add at least one data product.",
    ]


def test_validate_form_names_new_row_without_output_type_or_product():
    """A new, empty row gets one issue for its output type and one for its product."""
    validation = _validate_form(
        points_df=None,
        points_error=None,
        input_crs="EPSG:4326",
        credentials_bytes=None,
        output_dir="out",
        dataset_rows=[app._empty_dataset_row()],
        catalog={},
        widget_version=0,
    )

    step_4_issues = [issue for issue in validation.issues if issue.message.startswith("Step 4")]
    assert [issue.message for issue in step_4_issues] == [
        "Step 4 — Data product 1: choose an output type (Tabular or Raster).",
        "Step 4 — Data product 1: choose a data product.",
    ]
    assert step_4_issues[0].widget_keys == ("output_type_select_0_0",)
    assert step_4_issues[1].widget_keys == ("dataset_select_0_0",)
    assert validation.selections == ()


def test_validate_dataset_rows_names_products_and_checks_stats_before_windows():
    rows = [
        {
            "output_type": TABULAR_OUTPUT,
            "dataset": "dem_copernicus_glo30",
            "sample_point": False,
            "sample_window": True,
            "statistics": [],
            "window_sizes": "",
        },
        {
            "output_type": TABULAR_OUTPUT,
            "dataset": "",
            "sample_point": False,
            "sample_window": False,
            "statistics": [],
            "window_sizes": "",
        },
    ]
    catalog = {"dem_copernicus_glo30": {"display_name": "Copernicus DEM GLO-30"}}

    issues, selections = _validate_dataset_rows(rows, catalog, 4)

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
        [
            {
                "output_type": TABULAR_OUTPUT,
                "dataset": "",
                "sample_point": False,
                "sample_window": False,
            }
        ],
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
                "output_type": TABULAR_OUTPUT,
                "dataset": "dem",
                "sample_point": False,
                "sample_window": False,
            }
        ],
        {"dem": {"display_name": "Elevation"}},
        2,
    )

    assert [issue.message for issue in issues] == [
        "Step 4 — Data product “Elevation”: choose Point, Window, or both."
    ]
    assert issues[0].widget_keys == ("point_checkbox_2_0", "window_checkbox_2_0")
    assert selections == []


def test_validate_dataset_rows_names_rows_without_output_type():
    """A row without an output type gets an issue that names it, and no sampling checks."""
    rows = [
        {
            "output_type": None,
            "dataset": "dem",
            "sample_point": False,
            "sample_window": True,
            "statistics": [],
            "window_sizes": "",
        },
        {"output_type": None, "dataset": ""},
    ]

    issues, selections = _validate_dataset_rows(rows, {"dem": {"display_name": "Elevation"}}, 3)

    assert [issue.message for issue in issues] == [
        "Step 4 — Data product “Elevation”: choose an output type (Tabular or Raster).",
        "Step 4 — Data product 2: choose an output type (Tabular or Raster).",
        "Step 4 — Data product 2: choose a data product.",
    ]
    assert [issue.widget_keys for issue in issues] == [
        ("output_type_select_3_0",),
        ("output_type_select_3_1",),
        ("dataset_select_3_1",),
    ]
    assert selections == []


def test_validate_dataset_rows_checks_each_row_with_its_own_output_type():
    """Tabular and raster rows in one form are validated and selected with their own type."""
    rows = [
        {
            "output_type": RASTER_OUTPUT,
            "dataset": "dem",
            "sample_point": False,
            "sample_window": False,
            "statistics": [],
            "window_sizes": "200, 500",
        },
        {
            "output_type": TABULAR_OUTPUT,
            "dataset": "dem",
            "sample_point": True,
            "sample_window": True,
            "statistics": ["mean"],
            "window_sizes": "100",
        },
        {
            "output_type": RASTER_OUTPUT,
            "dataset": "lulc",
            "window_sizes": "",
        },
    ]
    catalog = {"dem": {"display_name": "Elevation"}, "lulc": {"display_name": "Land cover"}}

    issues, selections = _validate_dataset_rows(rows, catalog, 1)

    assert [issue.message for issue in issues] == [
        "Step 4 — Data product “Land cover”: enter at least one sampling-window size."
    ]
    assert issues[0].widget_keys == ("windows_input_1_2",)
    assert selections == [
        DatasetSelection(
            dataset="dem",
            output_type=RASTER_OUTPUT,
            window_sizes=(200, 500),
            statistics=(),
        ),
        DatasetSelection(
            dataset="dem",
            output_type=TABULAR_OUTPUT,
            window_sizes=(100,),
            statistics=("mean", "point"),
        ),
    ]


def test_progress_segments_key_mixed_runs_by_each_run_output_type():
    """A run with tabular and raster rows gets one segment per window with its own mode."""
    config = build_run_config(
        [
            DatasetSelection(
                dataset="dem",
                output_type=TABULAR_OUTPUT,
                window_sizes=(0,),
                statistics=("point",),
            ),
            DatasetSelection(dataset="dem", output_type=RASTER_OUTPUT, window_sizes=(200, 500)),
        ]
    )

    assert _progress_segments(config, fallback_total=6) == {
        ("extract_01_dem", "dem", 0, TABULAR_OUTPUT): 6,
        ("extract_02_dem", "dem", 200, RASTER_OUTPUT): 6,
        ("extract_02_dem", "dem", 500, RASTER_OUTPUT): 6,
    }


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
    assert "FileOpenDialog" in calls[0][-1]
    assert "PickFolders" in calls[0][-1]


def test_choose_output_directory_supports_wsl(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(app.sys, "platform", "linux")
    monkeypatch.setattr(app, "_is_wsl", lambda: True)

    executables = {
        "powershell.exe": "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
        "wslpath": "/usr/bin/wslpath",
    }
    monkeypatch.setattr(app.shutil, "which", executables.get)

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["/usr/bin/wslpath", "-w"]:
            return CompletedProcess(
                command,
                0,
                stdout="\\\\wsl.localhost\\Ubuntu\\home\\user\\output\n",
                stderr="",
            )
        if command[0].endswith("powershell.exe"):
            return CompletedProcess(command, 0, stdout="C:\\Users\\user\\output\n", stderr="")
        if command[:2] == ["/usr/bin/wslpath", "-u"]:
            return CompletedProcess(command, 0, stdout="/mnt/c/Users/user/output\n", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(app.subprocess, "run", fake_run)

    assert _choose_output_directory(str(tmp_path)) == "/mnt/c/Users/user/output"
    assert calls[0][:2] == ["/usr/bin/wslpath", "-w"]
    assert calls[1][0].endswith("powershell.exe")
    assert calls[2][:2] == ["/usr/bin/wslpath", "-u"]


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
                {
                    "output_type": RASTER_OUTPUT,
                    "dataset": "dem",
                    "window_sizes": "100",
                    "statistics": [],
                },
                {"dataset": "agb", "window_sizes": "200", "statistics": ["mean"]},
                {
                    "output_type": TABULAR_OUTPUT,
                    "dataset": "lulc",
                    "window_sizes": "300",
                    "statistics": ["mode"],
                },
            ],
            "_pending_dataset_remove": 1,
            "_dataset_widget_version": 4,
            "output_type_select_4_0": RASTER_OUTPUT,
            "output_type_select_4_2": TABULAR_OUTPUT,
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
        {"output_type": RASTER_OUTPUT, "dataset": "dem", "window_sizes": "100", "statistics": []},
        {
            "output_type": TABULAR_OUTPUT,
            "dataset": "lulc",
            "window_sizes": "300",
            "statistics": ["mode"],
        },
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
            "output_type": None,
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


def _tabular_row(dataset: str) -> dict:
    return {
        "output_type": TABULAR_OUTPUT,
        "dataset": dataset,
        "sample_point": True,
        "sample_window": True,
        "window_sizes": "100",
        "statistics": ["mean"],
        "statistics_dataset": dataset,
    }


def test_apply_row_output_type_keeps_state_when_type_is_unchanged():
    """A rerun with the same output type keeps the row's statistics and widget state."""
    session_state = _SessionState(
        {
            "dataset_rows": [_tabular_row("dem")],
            "point_checkbox_2_0": True,
            "stats_select_2_0_dem": ["mean"],
        }
    )

    _apply_row_output_type(_FakeStreamlit(session_state), 0, "2_0", TABULAR_OUTPUT)

    assert session_state["dataset_rows"] == [_tabular_row("dem")]
    assert session_state["point_checkbox_2_0"] is True
    assert session_state["stats_select_2_0_dem"] == ["mean"]


def test_apply_row_output_type_change_clears_only_that_rows_statistics():
    """Changing one row's type resets that row's statistics state and nothing else."""
    session_state = _SessionState(
        {
            "dataset_rows": [_tabular_row("dem"), _tabular_row("lulc")],
            "output_type_select_2_0": TABULAR_OUTPUT,
            "output_type_select_2_1": RASTER_OUTPUT,
            "point_checkbox_2_0": True,
            "window_checkbox_2_0": True,
            "stats_select_2_0_dem": ["mean"],
            "point_checkbox_2_1": True,
            "window_checkbox_2_1": True,
            "stats_select_2_1_lulc": ["mode"],
            "stats_select_2_1_none": [],
            "dataset_select_2_1": "lulc",
            "windows_input_2_1": "100",
            "stats_select_2_10_dem": ["max"],
        }
    )

    _apply_row_output_type(_FakeStreamlit(session_state), 1, "2_1", RASTER_OUTPUT)

    assert session_state["dataset_rows"] == [
        _tabular_row("dem"),
        {
            "output_type": RASTER_OUTPUT,
            "dataset": "lulc",
            "sample_point": False,
            "sample_window": False,
            "window_sizes": "100",
            "statistics": [],
        },
    ]
    assert sorted(session_state) == [
        "dataset_rows",
        "dataset_select_2_1",
        "output_type_select_2_0",
        "output_type_select_2_1",
        "point_checkbox_2_0",
        "stats_select_2_0_dem",
        "stats_select_2_10_dem",
        "window_checkbox_2_0",
        "windows_input_2_1",
    ]


def test_render_dataset_rows_shows_only_type_and_product_before_type_is_chosen(monkeypatch):
    """A new row shows the output type, the product filter, and the product, and nothing more."""
    session_state = _SessionState({"dataset_rows": [app._empty_dataset_row()]})

    fake_st = _render_rows_with_fake(monkeypatch, session_state)

    assert sorted(fake_st.widgets) == [
        "dataset_select_0_0",
        "dataset_type_select_0_0",
        "output_type_select_0_0",
        "remove_dataset_button_0_0",
    ]
    output_type_widget = fake_st.widgets["output_type_select_0_0"]
    assert output_type_widget["label"] == "Output type"
    assert output_type_widget["options"] == [TABULAR_OUTPUT, RASTER_OUTPUT]
    assert output_type_widget["index"] is None
    assert fake_st.captions == []


def test_render_dataset_rows_shows_controls_and_guidance_for_each_rows_type(monkeypatch):
    """Tabular rows show statistics controls, raster rows only window sizes, each with its text."""
    raster_row = {**app._empty_dataset_row(), "output_type": RASTER_OUTPUT, "dataset": "lulc"}
    session_state = _SessionState({"dataset_rows": [_tabular_row("dem"), raster_row]})

    fake_st = _render_rows_with_fake(monkeypatch, session_state)

    row_0_widgets = sorted(key for key in fake_st.widgets if key.endswith(("_0_0", "_0_0_dem")))
    row_1_widgets = sorted(key for key in fake_st.widgets if key.endswith("_0_1"))
    assert row_0_widgets == [
        "dataset_select_0_0",
        "dataset_type_select_0_0",
        "output_type_select_0_0",
        "point_checkbox_0_0",
        "remove_dataset_button_0_0",
        "stats_select_0_0_dem",
        "window_checkbox_0_0",
        "windows_input_0_0",
    ]
    assert row_1_widgets == [
        "dataset_select_0_1",
        "dataset_type_select_0_1",
        "output_type_select_0_1",
        "remove_dataset_button_0_1",
        "windows_input_0_1",
    ]
    assert len(fake_st.captions) == 2
    assert fake_st.captions[0].startswith(_TABULAR_GUIDANCE_FRAGMENT)
    assert fake_st.captions[1].startswith(_RASTER_GUIDANCE_FRAGMENT)


def test_render_dataset_rows_rerun_keeps_types_and_clears_only_changed_row(monkeypatch):
    """A user change of one row's type clears that row's statistics, and later reruns keep it."""
    session_state = _SessionState(
        {
            "dataset_rows": [_tabular_row("dem"), _tabular_row("lulc")],
            "_dataset_widget_version": 0,
            # The user changed row 2 from tabular to raster before this rerun.
            "output_type_select_0_0": TABULAR_OUTPUT,
            "output_type_select_0_1": RASTER_OUTPUT,
            "point_checkbox_0_1": True,
            "window_checkbox_0_1": True,
            "stats_select_0_1_lulc": ["mode"],
        }
    )

    fake_st = _render_rows_with_fake(monkeypatch, session_state)

    assert session_state["dataset_rows"][0] == _tabular_row("dem")
    assert session_state["dataset_rows"][1] == {
        "output_type": RASTER_OUTPUT,
        "dataset": "lulc",
        "sample_point": False,
        "sample_window": False,
        "window_sizes": "100",
        "statistics": [],
    }
    assert "point_checkbox_0_1" not in fake_st.widgets
    assert "stats_select_0_1_lulc" not in session_state

    # A new widget version (as after "Remove data product") drops all widget
    # state. The rows alone must then restore each row's type.
    rerun_state = _SessionState(
        {"dataset_rows": session_state["dataset_rows"], "_dataset_widget_version": 1}
    )

    rerun_st = _render_rows_with_fake(monkeypatch, rerun_state)

    assert rerun_st.widgets["output_type_select_1_0"]["index"] == 0
    assert rerun_st.widgets["output_type_select_1_1"]["index"] == 1
    assert [row["output_type"] for row in rerun_state["dataset_rows"]] == [
        TABULAR_OUTPUT,
        RASTER_OUTPUT,
    ]
    assert rerun_state["dataset_rows"][0] == _tabular_row("dem")
    assert rerun_state["stats_select_1_0_dem"] == ["mean"]
