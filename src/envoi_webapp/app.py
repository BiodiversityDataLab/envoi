from __future__ import annotations

import shutil
import subprocess
import sys
from base64 import b64encode
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from envoi import list_datasets, list_reducers
from envoi.progress import ProgressEvent

try:
    from .helpers import (
        RASTER_OUTPUT,
        TABULAR_OUTPUT,
        DatasetSelection,
        build_run_config,
        normalize_crs,
        parse_window_sizes,
        permissible_statistics_for_dataset,
        read_points_csv,
        redact_credential_secrets,
        run_extraction,
        validate_points_dataframe,
        validate_service_account_json,
        validate_wgs84_ranges,
    )
except ImportError:
    from envoi_webapp.helpers import (
        RASTER_OUTPUT,
        TABULAR_OUTPUT,
        DatasetSelection,
        build_run_config,
        normalize_crs,
        parse_window_sizes,
        permissible_statistics_for_dataset,
        read_points_csv,
        redact_credential_secrets,
        run_extraction,
        validate_points_dataframe,
        validate_service_account_json,
        validate_wgs84_ranges,
    )

# Adjust these values to tune the Streamlit page gutters.
CONTENT_MARGIN_LEFT_REM = 5
CONTENT_MARGIN_RIGHT_REM = 32
MOBILE_CONTENT_MARGIN_REM = 3
LOGO_TOP_OFFSET_REM = -1.7
RIGHT_BORDER_COLOR = "#2f7f5f"
RIGHT_LIGHT_GAP_REM = 5
HEADER_BODY_FONT_SIZE_REM = 1.2
HEADER_KICKER_GAP_REM = -0.05
STAT_TAG_BACKGROUND = "#6fb488"
STAT_TAG_TEXT = "#17302b"
THEME_PRIMARY_COLOR = STAT_TAG_BACKGROUND
DATASET_CATALOG_URL = "https://github.com/BiodiversityDataLab/envoi/blob/main/docs/datasets.md"
VALIDATION_ERROR_COLOR = "#d32f2f"


@dataclass(frozen=True)
class _ValidationIssue:
    message: str
    widget_keys: tuple[str, ...]


@dataclass(frozen=True)
class _FormValidation:
    issues: tuple[_ValidationIssue, ...]
    selections: tuple[DatasetSelection, ...]
    normalized_crs: str


def _load_streamlit():
    import streamlit as st

    return st


def _logo_path() -> Path | None:
    try:
        path = resources.files("envoi_webapp").joinpath(
            "assets", "BDDL_icononly_clearspace_300x300.svg"
        )
    except ModuleNotFoundError:
        return None
    return Path(str(path)) if Path(str(path)).exists() else None


def _favicon_path() -> Path | str:
    try:
        path = resources.files("envoi_webapp").joinpath("assets", "bddl_logo.png")
    except ModuleNotFoundError:
        return "E"
    favicon_path = Path(str(path))
    return favicon_path if favicon_path.exists() else "E"


def _inject_css(st) -> None:
    st.markdown(
        """
        <style>
        :root {
          --envoi-green: #2f7f5f;
          --envoi-green-soft: #e7f1ea;
          --envoi-blue: #2f607f;
          --envoi-blue-soft: #e6eef4;
          --envoi-ink: #17302b;
        }
        """
        + f"""
        :root {{
          --envoi-right-border-color: {RIGHT_BORDER_COLOR};
          --envoi-right-panel-width: calc({CONTENT_MARGIN_RIGHT_REM}rem - {RIGHT_LIGHT_GAP_REM}rem);
          --envoi-header-body-font-size: {HEADER_BODY_FONT_SIZE_REM}rem;
          --envoi-header-kicker-gap: {HEADER_KICKER_GAP_REM}rem;
        }}
        .block-container {{
          padding-left: {CONTENT_MARGIN_LEFT_REM}rem;
          padding-right: {CONTENT_MARGIN_RIGHT_REM}rem;
        }}
        .envoi-right-panel {{
          background: {RIGHT_BORDER_COLOR};
          bottom: 0;
          pointer-events: none;
          position: fixed;
          right: 0;
          top: 0;
          width: var(--envoi-right-panel-width);
          z-index: 0;
        }}
        @media (max-width: 900px) {{
          .block-container {{
            padding-left: {MOBILE_CONTENT_MARGIN_REM}rem;
            padding-right: {MOBILE_CONTENT_MARGIN_REM}rem;
          }}
          .envoi-right-panel {{
            display: none;
          }}
        }}
        .envoi-logo {{
          transform: translateY({LOGO_TOP_OFFSET_REM}rem);
        }}
        .envoi-logo img {{
          max-width: 200px;
          width: 100%;
          height: auto;
          display: block;
        }}
        div[data-baseweb="select"] [data-baseweb="tag"] {{
          background-color: {STAT_TAG_BACKGROUND} !important;
          border-color: {STAT_TAG_BACKGROUND} !important;
          color: {STAT_TAG_TEXT} !important;
        }}
        div[data-baseweb="select"] [data-baseweb="tag"] svg {{
          color: {STAT_TAG_TEXT} !important;
          fill: {STAT_TAG_TEXT} !important;
        }}
        div[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:checked) > span {{
          background-color: {STAT_TAG_BACKGROUND} !important;
          border-color: {STAT_TAG_BACKGROUND} !important;
        }}
        """
        + """
        .stApp {
          background: linear-gradient(180deg, #f7faf8 0%, #ffffff 44%);
          color: var(--envoi-ink);
        }
        .envoi-header {
          border-bottom: 1px solid #d8e6dc;
          padding-bottom: 1.1rem;
          margin-bottom: 1.5rem;
        }
        .envoi-header h1 {
          margin-top: 0;
        }
        .envoi-header p {
          font-size: var(--envoi-header-body-font-size);
        }
        .envoi-kicker {
          color: var(--envoi-green);
          font-weight: 700;
          letter-spacing: 0;
          margin-bottom: var(--envoi-header-kicker-gap);
        }
        .envoi-footer {
          border-top: 1px solid #d8e6dc;
          color: #47645e;
          font-size: 0.9rem;
          line-height: 1.5;
          margin-top: 3rem;
          padding-top: 1rem;
        }
        div.stButton > button[kind="primary"] {
          background: var(--envoi-green);
          border-color: var(--envoi-green);
        }
        div.stButton > button[kind="primary"]:hover {
          background: #25694e;
          border-color: #25694e;
        }
        [data-testid="stExpander"] {
          border-color: #d8e6dc;
          border-radius: 8px;
        }
        [data-testid="stFileUploader"] button[aria-label="Add files"] {
          display: none;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] {
          display: none !important;
        }
        div[class*="st-key-remove_dataset_action"] button {
          background: #fdecec;
          border-color: #efb6b6;
          color: #8f2424;
        }
        div[class*="st-key-remove_dataset_action"] button:hover {
          background: #f9dede;
          border-color: #df8f8f;
          color: #711c1c;
        }
        </style>
        <div class="envoi-right-panel"></div>
        """,
        unsafe_allow_html=True,
    )


def _dataset_catalog() -> dict[str, dict[str, Any]]:
    entries = list_datasets("full")
    return {entry["name"]: entry for entry in entries}


def _dataset_display_name(dataset_name: str, catalog: dict[str, dict[str, Any]]) -> str:
    """Return the catalog label while keeping the dataset key as a safe fallback."""

    display_name = catalog.get(dataset_name, {}).get("display_name")
    if isinstance(display_name, str) and display_name.strip():
        return display_name.strip()
    return dataset_name


def _dataset_label(dataset_name: str, index: int, catalog: dict[str, dict[str, Any]]) -> str:
    if dataset_name:
        return f"Data product “{_dataset_display_name(dataset_name, catalog)}”"
    return f"Data product {index + 1}"


def _ensure_dataset_state() -> None:
    st = _load_streamlit()
    if "dataset_rows" not in st.session_state:
        st.session_state.dataset_rows = [_empty_dataset_row()]
    if "_dataset_widget_version" not in st.session_state:
        st.session_state._dataset_widget_version = 0


def _empty_dataset_row() -> dict:
    return {
        "dataset": "",
        "sample_point": False,
        "sample_window": False,
        "window_sizes": "",
        "statistics": [],
    }


def _logo_image_html(path: Path) -> str:
    media_type = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
    encoded = b64encode(path.read_bytes()).decode("ascii")
    return f'<div class="envoi-logo"><img src="data:{media_type};base64,{encoded}" /></div>'


def _render_header(st) -> None:
    logo = _logo_path()
    cols = st.columns([0.16, 0.84], vertical_alignment="center")
    if logo is not None:
        cols[0].markdown(_logo_image_html(logo), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(
            """
            <div class="envoi-header">
              <div class="envoi-kicker">Biodiversity Data Lab</div>
              <h1>envoi: Geospatial data extraction</h1>
              <p>
                A tool for downloading environmental data from Google Earth Engine
                for sampling points or occurrence records.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _read_uploaded_csv(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    return read_points_csv(uploaded_file)


def _choose_output_directory(initial_dir: str) -> str | None:
    """Open a native directory chooser on the machine running the local app.

    Prefer the operating system's own chooser, then fall back to Tk in a
    separate process. Keeping every GUI outside Streamlit's worker thread
    avoids platform-specific GUI event-loop failures.
    """

    initial_path = Path(initial_dir).expanduser()
    if not initial_path.exists():
        initial_path = Path.home()
    errors: list[str] = []

    if sys.platform == "darwin":
        # Streamlit runs app code outside the macOS main thread, so tkinter/Tk
        # can abort the whole process. AppleScript opens the chooser in a
        # separate process and returns the selected POSIX path.
        script = (
            'POSIX path of (choose folder with prompt "Select envoi output directory" '
            f'default location POSIX file "{_escape_applescript_string(str(initial_path))}")'
        )
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
        if result.returncode == 1 and "User canceled" in result.stderr:
            return None
        errors.append(result.stderr.strip() or "AppleScript folder chooser failed.")

    elif sys.platform.startswith("win"):
        powershell = (
            shutil.which("powershell.exe") or shutil.which("powershell") or shutil.which("pwsh")
        )
        if powershell:
            script = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
                f"$dialog.SelectedPath = '{_escape_powershell_string(str(initial_path))}'; "
                "$dialog.Description = 'Select envoi output directory'; "
                "if ($dialog.ShowDialog() -eq 'OK') { Write-Output $dialog.SelectedPath }"
            )
            result = subprocess.run(
                [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True,
                check=False,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip() or None
            errors.append(result.stderr.strip() or "Windows folder chooser failed.")

    elif sys.platform.startswith("linux"):
        linux_choosers = (
            (
                "zenity",
                [
                    "--file-selection",
                    "--directory",
                    "--title=Select envoi output directory",
                    f"--filename={initial_path}/",
                ],
            ),
            (
                "kdialog",
                [
                    "--getexistingdirectory",
                    str(initial_path),
                    "--title",
                    "Select envoi output directory",
                ],
            ),
            (
                "yad",
                [
                    "--file-selection",
                    "--directory",
                    "--title=Select envoi output directory",
                    f"--filename={initial_path}/",
                ],
            ),
        )
        for executable, arguments in linux_choosers:
            chooser = shutil.which(executable)
            if chooser is None:
                continue
            result = subprocess.run(
                [chooser, *arguments], capture_output=True, check=False, text=True
            )
            if result.returncode == 0:
                return result.stdout.strip() or None
            if result.returncode == 1:
                return None
            errors.append(result.stderr.strip() or f"{executable} folder chooser failed.")

    tkinter_script = """
import sys
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.update()
selected = filedialog.askdirectory(
    initialdir=sys.argv[1],
    title="Select envoi output directory",
    mustexist=True,
)
root.destroy()
if selected:
    print(selected)
""".strip()
    result = subprocess.run(
        [sys.executable, "-c", tkinter_script, str(initial_path)],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip() or None
    errors.append(result.stderr.strip() or "Tk folder chooser failed.")

    detail = next((error.splitlines()[-1] for error in reversed(errors) if error), "")
    suffix = f" Last error: {detail}" if detail else ""
    raise RuntimeError(
        "No graphical folder chooser is available. Enter the output directory path manually."
        + suffix
    )


def _escape_applescript_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _escape_powershell_string(value: str) -> str:
    """Escape a value embedded in a single-quoted PowerShell string."""

    return value.replace("'", "''")


def _clear_dataset_widget_state(st) -> None:
    for key in list(st.session_state.keys()):
        key_text = str(key)
        if key_text.startswith(
            (
                "dataset_select_",
                "point_checkbox_",
                "window_checkbox_",
                "windows_input_",
                "stats_select_",
                "remove_dataset_button_",
                "windows_",
                "stats_",
            )
        ) or (key_text.startswith("dataset_") and key_text[8:].isdigit()):
            st.session_state.pop(key, None)
    st.session_state._dataset_widget_version = (
        int(st.session_state.get("_dataset_widget_version", 0)) + 1
    )


def _apply_pending_dataset_remove(st) -> None:
    if "_pending_dataset_remove" not in st.session_state:
        return

    remove_index = st.session_state.pop("_pending_dataset_remove")
    rows = list(st.session_state.dataset_rows)
    if len(rows) <= 1:
        st.session_state.dataset_rows = [_empty_dataset_row()]
    elif 0 <= remove_index < len(rows):
        rows.pop(remove_index)
        st.session_state.dataset_rows = rows

    _clear_dataset_widget_state(st)


def _render_dataset_rows(st, catalog: dict[str, dict[str, Any]], output_type: str) -> None:
    dataset_names = sorted(
        catalog,
        key=lambda name: (_dataset_display_name(name, catalog).casefold(), name.casefold()),
    )
    reducers = list_reducers()
    if not dataset_names:
        st.error("No data products are available in the envoi catalog.")
        return

    _ensure_dataset_state()
    _apply_pending_dataset_remove(st)
    widget_version = int(st.session_state.get("_dataset_widget_version", 0))

    for index, row in enumerate(st.session_state.dataset_rows):
        with st.expander(f"Data product {index + 1}", expanded=True):
            row_widget_key = f"{widget_version}_{index}"
            if output_type == TABULAR_OUTPUT:
                top_cols = st.columns([0.78, 0.11, 0.11], vertical_alignment="bottom")
            else:
                top_cols = st.columns([0.65, 0.35], vertical_alignment="bottom")
            current_dataset = row.get("dataset") if row.get("dataset") in dataset_names else None
            selected_dataset = top_cols[0].selectbox(
                "Data product",
                dataset_names,
                index=dataset_names.index(current_dataset) if current_dataset else None,
                placeholder="Choose a data product",
                key=f"dataset_select_{row_widget_key}",
                format_func=lambda name: _dataset_display_name(name, catalog),
            )
            windows = str(row.get("window_sizes", ""))
            st.session_state.dataset_rows[index]["dataset"] = selected_dataset or ""

            if output_type == TABULAR_OUTPUT:
                sample_point = top_cols[1].checkbox(
                    "Point",
                    value=bool(row.get("sample_point", False)),
                    key=f"point_checkbox_{row_widget_key}",
                    help="Sample the raster pixel value at each coordinate.",
                )
                sample_window = top_cols[2].checkbox(
                    "Window",
                    value=bool(row.get("sample_window", False)),
                    key=f"window_checkbox_{row_widget_key}",
                    help="Calculate spatial statistics within one or more sampling windows.",
                )
                st.session_state.dataset_rows[index]["sample_point"] = sample_point
                st.session_state.dataset_rows[index]["sample_window"] = sample_window

                permitted_reducers = (
                    [
                        reducer
                        for reducer in permissible_statistics_for_dataset(
                            catalog[selected_dataset], reducers
                        )
                        if reducer != "point"
                    ]
                    if selected_dataset
                    else []
                )
                if selected_dataset and row.get("statistics_dataset") == selected_dataset:
                    defaults = list(row.get("statistics") or [])
                else:
                    defaults = []
                valid_defaults = [stat for stat in defaults if stat in permitted_reducers]
                if sample_window:
                    detail_cols = st.columns([0.65, 0.35], vertical_alignment="bottom")
                    selected_stats = detail_cols[0].multiselect(
                        "Spatial statistics",
                        permitted_reducers,
                        default=valid_defaults,
                        key=f"stats_select_{row_widget_key}_{selected_dataset or 'none'}",
                        placeholder=(
                            "Choose one or more statistics"
                            if selected_dataset
                            else "Choose a data product first"
                        ),
                        disabled=selected_dataset is None,
                    )
                    windows = detail_cols[1].text_input(
                        "Window size(s) in meters",
                        value=row.get("window_sizes", ""),
                        placeholder="e.g. 500, 1000",
                        key=f"windows_input_{row_widget_key}",
                    )
                    st.session_state.dataset_rows[index]["statistics"] = (
                        selected_stats if selected_dataset else []
                    )
                    st.session_state.dataset_rows[index]["window_sizes"] = windows
                    if selected_dataset:
                        st.session_state.dataset_rows[index]["statistics_dataset"] = (
                            selected_dataset
                        )
                    else:
                        st.session_state.dataset_rows[index].pop("statistics_dataset", None)
            else:
                windows = top_cols[1].text_input(
                    "Window size(s) in meters",
                    value=row.get("window_sizes", ""),
                    placeholder="e.g. 500, 1000",
                    key=f"windows_input_{row_widget_key}",
                )
                st.session_state.dataset_rows[index]["window_sizes"] = windows
                st.session_state.dataset_rows[index]["statistics"] = []
                st.session_state.dataset_rows[index].pop("statistics_dataset", None)

            remove_disabled = not (
                selected_dataset
                or st.session_state.dataset_rows[index].get("sample_point")
                or st.session_state.dataset_rows[index].get("sample_window")
                or windows.strip()
                or st.session_state.dataset_rows[index].get("statistics")
            )
            remove_container = st.container(key=f"remove_dataset_action_{row_widget_key}")
            if remove_container.button(
                "Remove data product",
                key=f"remove_dataset_button_{row_widget_key}",
                disabled=remove_disabled,
            ):
                st.session_state._pending_dataset_remove = index
                st.rerun()

    if st.button("Add another data product"):
        st.session_state.dataset_rows.append(_empty_dataset_row())
        st.rerun()


def _validate_dataset_rows(
    rows: list[dict],
    output_type: str,
    catalog: dict[str, dict[str, Any]],
    widget_version: int,
) -> tuple[list[_ValidationIssue], list[DatasetSelection]]:
    """Validate visible data-product fields in their left-to-right order."""

    issues: list[_ValidationIssue] = []
    selections: list[DatasetSelection] = []
    if not rows:
        return [_ValidationIssue("Step 4 — Add at least one data product.", ())], selections

    for index, row in enumerate(rows):
        row_widget_key = f"{widget_version}_{index}"
        dataset = str(row.get("dataset") or "")
        label = _dataset_label(dataset, index, catalog)
        if not dataset:
            issues.append(
                _ValidationIssue(
                    f"Step 4 — {label}: choose a data product.",
                    (f"dataset_select_{row_widget_key}",),
                )
            )
            # Point/window and their dependent fields are irrelevant until a
            # product has actually been selected.
            continue

        if output_type == TABULAR_OUTPUT:
            sample_point = bool(row.get("sample_point", False))
            sample_window = bool(row.get("sample_window", False))
            if not sample_point and not sample_window:
                issues.append(
                    _ValidationIssue(
                        f"Step 4 — {label}: choose Point, Window, or both.",
                        (
                            f"point_checkbox_{row_widget_key}",
                            f"window_checkbox_{row_widget_key}",
                        ),
                    )
                )
                continue

            statistics = tuple(row.get("statistics") or []) if sample_window else ()
            row_has_error = False
            if sample_window and not statistics:
                issues.append(
                    _ValidationIssue(
                        f"Step 4 — {label}: choose at least one spatial statistic.",
                        (f"stats_select_{row_widget_key}_{dataset}",),
                    )
                )
                row_has_error = True

            if sample_window:
                try:
                    window_sizes = parse_window_sizes(str(row.get("window_sizes") or ""))
                except ValueError as exc:
                    detail = str(exc)
                    if detail == "At least one window size is required.":
                        detail = "enter at least one sampling-window size."
                    else:
                        detail = f"enter valid sampling-window sizes. {detail}"
                    issues.append(
                        _ValidationIssue(
                            f"Step 4 — {label}: {detail}",
                            (f"windows_input_{row_widget_key}",),
                        )
                    )
                    row_has_error = True
                    window_sizes = ()
            else:
                window_sizes = (0,)

            if sample_point:
                statistics += ("point",)
            if row_has_error:
                continue
        else:
            try:
                window_sizes = parse_window_sizes(str(row.get("window_sizes") or ""))
            except ValueError as exc:
                detail = str(exc)
                if detail == "At least one window size is required.":
                    detail = "enter at least one sampling-window size."
                else:
                    detail = f"enter valid sampling-window sizes. {detail}"
                issues.append(
                    _ValidationIssue(
                        f"Step 4 — {label}: {detail}",
                        (f"windows_input_{row_widget_key}",),
                    )
                )
                continue
            statistics = ()

        selections.append(
            DatasetSelection(
                dataset=dataset,
                window_sizes=window_sizes,
                statistics=statistics,
            )
        )
    return issues, selections


def _validate_form(
    *,
    points_df: pd.DataFrame | None,
    points_error: str | None,
    input_crs: str,
    credentials_bytes: bytes | None,
    output_type: str | None,
    output_dir: str,
    dataset_rows: list[dict],
    catalog: dict[str, dict[str, Any]],
    widget_version: int,
) -> _FormValidation:
    """Collect form errors in step order without running the extraction."""

    issues: list[_ValidationIssue] = []
    normalized_crs = ""

    # Step 1: location file, then its coordinate reference system.
    if points_df is None:
        message = points_error or "Upload a valid location CSV."
        issues.append(_ValidationIssue(f"Step 1 — {message}", ("location_csv",)))

    if not input_crs.strip():
        issues.append(
            _ValidationIssue(
                "Step 1 — Enter the EPSG code for the uploaded location data.",
                ("custom_epsg",),
            )
        )
    else:
        try:
            normalized_crs = normalize_crs(input_crs)
        except ValueError as exc:
            issues.append(_ValidationIssue(f"Step 1 — {exc}", ("custom_epsg",)))
        else:
            if points_df is not None:
                try:
                    validate_wgs84_ranges(points_df, normalized_crs)
                except ValueError as exc:
                    issues.append(_ValidationIssue(f"Step 1 — {exc}", ("location_csv",)))

    # Step 2: credentials.
    if credentials_bytes is None:
        issues.append(
            _ValidationIssue(
                "Step 2 — Upload an Earth Engine service-account JSON key.",
                ("credentials_json",),
            )
        )
    else:
        try:
            validate_service_account_json(credentials_bytes)
        except ValueError as exc:
            issues.append(_ValidationIssue(f"Step 2 — {exc}", ("credentials_json",)))

    # Step 3: output type, then output directory.
    if output_type not in {TABULAR_OUTPUT, RASTER_OUTPUT}:
        issues.append(
            _ValidationIssue(
                "Step 3 — Choose between tabular or raster output.",
                ("output_type",),
            )
        )
    if not output_dir.strip():
        issues.append(
            _ValidationIssue(
                "Step 3 — Enter an output directory.",
                ("output_dir",),
            )
        )

    # Step 4 is hidden until the output type is known, so only validate fields
    # the user could actually interact with.
    selections: list[DatasetSelection] = []
    if output_type in {TABULAR_OUTPUT, RASTER_OUTPUT}:
        dataset_issues, selections = _validate_dataset_rows(
            dataset_rows, output_type, catalog, widget_version
        )
        issues.extend(dataset_issues)

    return _FormValidation(tuple(issues), tuple(selections), normalized_crs)


def _render_validation_issues(st, issues: tuple[_ValidationIssue, ...]) -> None:
    """Show one ordered summary and outline every implicated widget in red."""

    summary = "Please fix the following:\n\n" + "\n".join(f"- {issue.message}" for issue in issues)
    st.error(summary)

    widget_keys = dict.fromkeys(key for issue in issues for key in issue.widget_keys)
    selectors: list[str] = []
    for key in widget_keys:
        wrapper = f'div[class*="st-key-{key}"]'
        selectors.extend(
            (
                f'{wrapper} div[data-baseweb="select"] > div',
                f'{wrapper} div[data-baseweb="input"]',
                f'{wrapper} [data-testid="stFileUploaderDropzone"]',
                f'{wrapper} label[data-baseweb="checkbox"] > span',
            )
        )
    if selectors:
        st.markdown(
            "<style>\n"
            + ",\n".join(selectors)
            + f" {{ border-color: {VALIDATION_ERROR_COLOR} !important; "
            f"box-shadow: 0 0 0 1px {VALIDATION_ERROR_COLOR} !important; }}\n"
            "</style>",
            unsafe_allow_html=True,
        )


def _progress_segments(
    config: list[dict], fallback_total: int
) -> dict[tuple[str, str, int, str], int]:
    segments: dict[tuple[str, str, int, str], int] = {}
    for run_config in config:
        settings = run_config["settings"]
        raw_window_sizes = settings["window_size_m"]
        if isinstance(raw_window_sizes, list):
            window_sizes = raw_window_sizes
        else:
            window_sizes = [raw_window_sizes]
        for dataset in run_config["datasets"]:
            for window_size in window_sizes:
                key = (
                    run_config["batch_id"],
                    dataset,
                    int(window_size),
                    settings["output_type"],
                )
                segments[key] = fallback_total
    return segments


def _render_footer(st) -> None:
    st.markdown(
        """
        <div class="envoi-footer">
          envoi is developed at the <a href="https://www.biodiversity.se//" target="_blank">Biodiversity Data Lab</a> 
          at Uppsala University by Adrian Baggström and Jakob Nyström.
          The work is supported by the SciLifeLab and Wallenberg Data Driven Life
          Science Program and the Swedish Research Council. For advanced use, check out
          the envoi python package:
          <a href="https://pypi.org/project/envoi-geospatial/" target="_blank">PyPI</a>
          and
          <a href="https://github.com/BiodiversityDataLab/envoi" target="_blank">GitHub</a>.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_app() -> None:
    st = _load_streamlit()
    st.set_page_config(
        page_title="envoi: Geospatial data extraction",
        page_icon=_favicon_path(),
        layout="wide",
    )
    _inject_css(st)
    _render_header(st)

    catalog = _dataset_catalog()
    points_df: pd.DataFrame | None = None
    points_error: str | None = None

    st.subheader("1. Upload location data")
    st.write(
        "Upload a CSV file with occurrence records or sampling locations. It should contain the following columns, "
        "in Darwin Core format: occurrenceID (a unique identifier for the occurrence or location), decimalLatitude, and decimalLongitude."
        " Optionally, eventDate can be included to obtain date-specific information if available."
    )
    uploaded_csv = st.file_uploader(
        "Location CSV",
        type=["csv"],
        accept_multiple_files=False,
        key="location_csv",
    )
    if uploaded_csv is not None:
        try:
            points_df = _read_uploaded_csv(uploaded_csv)
            if points_df is not None:
                validation = validate_points_dataframe(points_df)
                st.success(
                    f"Loaded {validation.row_count} rows. "
                    f"Date column present: {'yes' if validation.has_date else 'no'}."
                )
                st.dataframe(points_df.head(20), width="stretch")
        except Exception as exc:
            points_error = str(exc)
            st.error(points_error)
            points_df = None

    crs_cols = st.columns([0.34, 0.66])
    crs_mode = crs_cols[0].selectbox(
        "Coordinate reference system of uploaded data", ["EPSG:4326", "Other EPSG"]
    )
    if crs_mode == "Other EPSG":
        input_crs = crs_cols[1].text_input(
            "EPSG code", placeholder="e.g. EPSG:3006", key="custom_epsg"
        )
    else:
        input_crs = "EPSG:4326"
    if input_crs:
        try:
            normalize_crs(input_crs)
        except ValueError as exc:
            st.error(str(exc))

    st.subheader("2. Add Earth Engine credentials")
    st.markdown(
        """
        Upload your Google Earth Engine service account JSON key. The key is only
        written to a temporary local file during extraction, then deleted when the
        run finishes. If you do not have a service account yet, follow the
        <a href="https://developers.google.com/earth-engine/guides/service_account" target="_blank">Earth Engine service account setup guide</a>.
        """,
        unsafe_allow_html=True,
    )
    credentials_file = st.file_uploader(
        "Earth Engine service account JSON",
        type=["json"],
        accept_multiple_files=False,
        key="credentials_json",
    )
    credentials_bytes = credentials_file.getvalue() if credentials_file is not None else None

    st.subheader("3. Choose output settings")
    output_type = st.selectbox(
        "Output type",
        [TABULAR_OUTPUT, RASTER_OUTPUT],
        index=None,
        placeholder="Choose between tabular or raster output",
        format_func=str.title,
        key="output_type",
    )
    if "output_dir" not in st.session_state:
        st.session_state.output_dir = str(Path("~/envoi_outputs").expanduser())
    if "_pending_output_dir" in st.session_state:
        st.session_state.output_dir = st.session_state.pop("_pending_output_dir")
    output_cols = st.columns([0.78, 0.22], vertical_alignment="bottom")
    output_dir = output_cols[0].text_input("Output directory", key="output_dir")
    if output_cols[1].button("Browse..."):
        try:
            selected_dir = _choose_output_directory(output_dir)
        except RuntimeError as exc:
            st.warning(str(exc))
        else:
            if selected_dir:
                st.session_state._pending_output_dir = selected_dir
                st.rerun()

    st.subheader("4. Select data products")
    window_guidance = ""
    if output_type == TABULAR_OUTPUT:
        window_guidance = "Coordinate point values as well as spatial statistics over sampling window(s) can be extracted. "
    elif output_type == RASTER_OUTPUT:
        window_guidance = (
            "The window size(s) determines the size of the extracted raster tiles. Note that raster "
            "outputs use 10 m resampling of source data by default, to ensure "
            "consistency in spatial resolution between data products."
        )
    st.markdown(
        f"""
        Add one entry per Earth Engine data product that should be downloaded. If a data product contains multiple bands, all of them will be processed and downloaded. For information about available data products, see the
        <a href="{DATASET_CATALOG_URL}" target="_blank">envoi catalog</a>. {window_guidance}
        """,
        unsafe_allow_html=True,
    )
    if output_type is None:
        st.info("Choose an output type before adding data products.")
    else:
        _render_dataset_rows(st, catalog, output_type)

    st.subheader("5. Run extraction")
    st.write(
        "The final outputs, data quality checks, and metadata are written to the output directory "
        "chosen in step 3."
    )
    run_button = st.button("Extract selected data", type="primary")
    if run_button:
        validation = _validate_form(
            points_df=points_df,
            points_error=points_error,
            input_crs=input_crs,
            credentials_bytes=credentials_bytes,
            output_type=output_type,
            output_dir=output_dir,
            dataset_rows=list(st.session_state.get("dataset_rows", [])),
            catalog=catalog,
            widget_version=int(st.session_state.get("_dataset_widget_version", 0)),
        )
        if validation.issues:
            _render_validation_issues(st, validation.issues)
        else:
            progress_bar = st.progress(0, text="Starting extraction")
            status = st.empty()
            completed_by_segment: dict[tuple[str, str, int, str], int] = {}

            try:
                selections = list(validation.selections)
                config = build_run_config(selections, output_type)
                expected_segments = _progress_segments(config, len(points_df))

                def handle_progress(event: ProgressEvent) -> None:
                    key = (event.batch_id, event.dataset, event.window_size_m, event.mode)
                    expected_segments[key] = max(event.total, 1)
                    completed_by_segment[key] = event.completed
                    total = sum(expected_segments.values()) or 1
                    completed = sum(
                        min(completed_by_segment.get(segment_key, 0), segment_total)
                        for segment_key, segment_total in expected_segments.items()
                    )
                    fraction = min(1.0, completed / total)
                    progress_bar.progress(
                        fraction,
                        text=(
                            f"{event.dataset} | "
                            f"{'point' if event.window_size_m == 0 else f'{event.window_size_m} m'} | "
                            f"{event.completed}/{event.total} {event.unit}"
                        ),
                    )
                    status.info(f"Current batch: {event.batch_id}")

                outputs = run_extraction(
                    points_df,
                    selections,
                    output_type,
                    output_dir,
                    validation.normalized_crs,
                    credentials_bytes,
                    progress_callback=handle_progress,
                )
                progress_bar.progress(1.0, text="Extraction complete")
                status.success("Extraction complete.")
                st.write("Outputs")
                for key, value in outputs.items():
                    st.code(f"{key}: {value}")
            except Exception as exc:
                safe_message = redact_credential_secrets(str(exc), credentials_bytes)
                st.error(safe_message)

    _render_footer(st)


def main() -> None:
    from streamlit.web import cli as stcli

    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Usage: envoi-webapp\nLaunches the local Envoi Streamlit web app.")
        return

    app_path = Path(__file__).resolve()
    sys.argv = [
        "streamlit",
        "run",
        "--server.address",
        "localhost",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--theme.primaryColor",
        THEME_PRIMARY_COLOR,
        str(app_path),
    ]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    render_app()
