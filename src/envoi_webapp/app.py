from __future__ import annotations

import sys
from base64 import b64encode
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
        read_points_csv,
        redact_credential_secrets,
        run_extraction,
        validate_points_dataframe,
    )
except ImportError:
    from envoi_webapp.helpers import (
        RASTER_OUTPUT,
        TABULAR_OUTPUT,
        DatasetSelection,
        build_run_config,
        normalize_crs,
        parse_window_sizes,
        read_points_csv,
        redact_credential_secrets,
        run_extraction,
        validate_points_dataframe,
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


def _ensure_dataset_state() -> None:
    st = _load_streamlit()
    if "dataset_rows" not in st.session_state:
        st.session_state.dataset_rows = [
            {
                "dataset": "",
                "window_sizes": "",
                "statistics": [],
            }
        ]


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


def _clear_dataset_widget_state(st) -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith(("dataset_", "windows_", "stats_")):
            st.session_state.pop(key, None)


def _render_dataset_rows(st, catalog: dict[str, dict[str, Any]], output_type: str) -> None:
    dataset_names = sorted(catalog)
    reducers = list_reducers()
    if not dataset_names:
        st.error("No datasets are available in the envoi catalog.")
        return

    _ensure_dataset_state()

    for index, row in enumerate(st.session_state.dataset_rows):
        with st.expander(f"Dataset {index + 1}", expanded=True):
            top_cols = st.columns([0.65, 0.35], vertical_alignment="bottom")
            current_dataset = row.get("dataset") if row.get("dataset") in dataset_names else None
            selected_dataset = top_cols[0].selectbox(
                "Dataset",
                dataset_names,
                index=dataset_names.index(current_dataset) if current_dataset else None,
                placeholder="Choose a dataset",
                key=f"dataset_{index}",
            )
            windows = top_cols[1].text_input(
                "Window size(s) in meters",
                value=row.get("window_sizes", ""),
                placeholder="500, 1000",
                key=f"windows_{index}",
            )

            st.session_state.dataset_rows[index]["dataset"] = selected_dataset or ""
            st.session_state.dataset_rows[index]["window_sizes"] = windows

            if output_type == TABULAR_OUTPUT:
                if selected_dataset and row.get("statistics_dataset") == selected_dataset:
                    defaults = list(row.get("statistics") or [])
                else:
                    defaults = []
                valid_defaults = [stat for stat in defaults if stat in reducers]
                selected_stats = st.multiselect(
                    "Spatial statistics",
                    reducers,
                    default=valid_defaults,
                    key=f"stats_{index}_{selected_dataset or 'none'}",
                    placeholder=(
                        "Choose one or more statistics"
                        if selected_dataset
                        else "Choose a dataset first"
                    ),
                    disabled=selected_dataset is None,
                )
                st.session_state.dataset_rows[index]["statistics"] = (
                    selected_stats if selected_dataset else []
                )
                if selected_dataset:
                    st.session_state.dataset_rows[index]["statistics_dataset"] = selected_dataset
                else:
                    st.session_state.dataset_rows[index].pop("statistics_dataset", None)
            else:
                st.session_state.dataset_rows[index]["statistics"] = []
                st.session_state.dataset_rows[index].pop("statistics_dataset", None)
                st.caption("Raster output uses 10 m resampling.")

            remove_disabled = not (
                selected_dataset
                or windows.strip()
                or st.session_state.dataset_rows[index].get("statistics")
            )
            remove_container = st.container(key=f"remove_dataset_action_{index}")
            if remove_container.button(
                "Remove dataset",
                key=f"remove_{index}",
                disabled=remove_disabled,
            ):
                if len(st.session_state.dataset_rows) == 1:
                    st.session_state.dataset_rows[index] = {
                        "dataset": "",
                        "window_sizes": "",
                        "statistics": [],
                    }
                else:
                    st.session_state.dataset_rows.pop(index)
                _clear_dataset_widget_state(st)
                st.rerun()

    if st.button("Add another dataset"):
        st.session_state.dataset_rows.append({"dataset": "", "window_sizes": "", "statistics": []})
        st.rerun()


def _dataset_selections(output_type: str) -> list[DatasetSelection]:
    st = _load_streamlit()
    selections: list[DatasetSelection] = []
    for row in st.session_state.dataset_rows:
        statistics = tuple(row.get("statistics") or []) if output_type == TABULAR_OUTPUT else ()
        selections.append(
            DatasetSelection(
                dataset=str(row.get("dataset") or ""),
                window_sizes=parse_window_sizes(str(row.get("window_sizes") or "")),
                statistics=statistics,
            )
        )
    return selections


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
        page_title="Envoi Earth Engine Extractor",
        page_icon="E",
        layout="wide",
    )
    _inject_css(st)
    _render_header(st)

    catalog = _dataset_catalog()
    points_df: pd.DataFrame | None = None

    st.subheader("1. Upload sampling locations")
    st.write(
        "Upload a CSV file with occurrence records or sampling locations. It should contain the following columns, "
        "in Darwin Core / GBIF format: occurrenceID (a unique identifier), decimalLatitude, and decimalLongitude."
        " Optionally, eventDate can be included to obtain date-specific information if available."
    )
    uploaded_csv = st.file_uploader("Sampling CSV", type=["csv"], accept_multiple_files=False)
    if uploaded_csv is not None:
        try:
            points_df = _read_uploaded_csv(uploaded_csv)
            if points_df is not None:
                validation = validate_points_dataframe(points_df)
                st.success(
                    f"Loaded {validation.row_count} rows. "
                    f"Date column present: {'yes' if validation.has_date else 'no'}."
                )
                st.dataframe(points_df.head(20), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))
            points_df = None

    crs_cols = st.columns([0.34, 0.66])
    crs_mode = crs_cols[0].selectbox("Coordinate reference system", ["EPSG:4326", "Other EPSG"])
    if crs_mode == "Other EPSG":
        input_crs = crs_cols[1].text_input("EPSG code", value="EPSG:3006")
    else:
        input_crs = "EPSG:4326"
    try:
        normalized_crs = normalize_crs(input_crs)
    except ValueError as exc:
        normalized_crs = input_crs
        st.error(str(exc))

    st.subheader("2. Choose output settings")
    output_type = st.selectbox(
        "Output type", [TABULAR_OUTPUT, RASTER_OUTPUT], format_func=str.title
    )
    output_dir = st.text_input("Output directory", value=str(Path("~/envoi_outputs").expanduser()))
    credentials_file = st.file_uploader(
        "Earth Engine service account key JSON",
        type=["json"],
        accept_multiple_files=False,
    )
    credentials_bytes = credentials_file.getvalue() if credentials_file is not None else None

    st.subheader("3. Select data products")
    st.markdown(
        """
        Add one entry per Earth Engine data product that should be downloaded. Each entry
        becomes one envoi output batch. If a data product contains multiple bands, all of
        them will be downloaded. For information about available data products, see the
        <a href="https://github.com/BiodiversityDataLab/envoi/blob/webapp/src/envoi/configs/ee_catalog.yml" target="_blank">envoi catalog</a>.
        """,
        unsafe_allow_html=True,
    )
    _render_dataset_rows(st, catalog, output_type)

    st.subheader("4. Run extraction")
    run_button = st.button("Extract selected data", type="primary")
    if run_button:
        if points_df is None:
            st.error("Upload a valid sampling CSV before running extraction.")
        elif credentials_bytes is None:
            st.error("Upload an Earth Engine service account key before running extraction.")
        else:
            progress_bar = st.progress(0, text="Starting extraction")
            status = st.empty()
            completed_by_segment: dict[tuple[str, str, int, str], int] = {}

            try:
                selections = _dataset_selections(output_type)
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
                            f"{event.dataset} | {event.window_size_m} m | "
                            f"{event.completed}/{event.total} {event.unit}"
                        ),
                    )
                    status.info(f"Current batch: {event.batch_id}")

                outputs = run_extraction(
                    points_df,
                    selections,
                    output_type,
                    output_dir,
                    normalized_crs,
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
        str(app_path),
    ]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    render_app()
