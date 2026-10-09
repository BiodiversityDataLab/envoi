from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
from base64 import b64encode
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from envoi import list_datasets, list_reducers
from envoi.catalog_docs import CATEGORY_ORDER as DATASET_TYPE_ORDER
from envoi.catalog_docs import UNCATEGORISED_LABEL

try:
    from .helpers import (
        RASTER_OUTPUT,
        TABULAR_OUTPUT,
        CsvValidationResult,
        DatasetSelection,
        build_run_config,
        normalize_crs,
        parse_window_sizes,
        permissible_statistics_for_dataset,
        read_points_csv,
        validate_output_dir,
        validate_points_dataframe,
        validate_service_account_json,
        validate_wgs84_ranges,
    )
    from .job_protocol import JobSnapshot, JobState, ProgressMessage
    from .jobs import JobManager, JobRejected, get_job_manager, key_hash
    from .settings import (
        HostedLimits,
        WebappSettings,
        check_hosted_limits,
        check_upload,
        format_bytes,
        format_minutes,
        load_settings,
    )
except ImportError:
    from envoi_webapp.helpers import (
        RASTER_OUTPUT,
        TABULAR_OUTPUT,
        CsvValidationResult,
        DatasetSelection,
        build_run_config,
        normalize_crs,
        parse_window_sizes,
        permissible_statistics_for_dataset,
        read_points_csv,
        validate_output_dir,
        validate_points_dataframe,
        validate_service_account_json,
        validate_wgs84_ranges,
    )
    from envoi_webapp.job_protocol import JobSnapshot, JobState, ProgressMessage
    from envoi_webapp.jobs import JobManager, JobRejected, get_job_manager, key_hash
    from envoi_webapp.settings import (
        HostedLimits,
        WebappSettings,
        check_hosted_limits,
        check_upload,
        format_bytes,
        format_minutes,
        load_settings,
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
ALL_DATASET_TYPES = "__all_dataset_types__"
# Installation of envoi for the local web app and the Python package. The
# hosted notice points users with larger runs here.
LOCAL_INSTALL_URL = "https://github.com/BiodiversityDataLab/envoi#browser-based-user-interface"

# Session-state keys of the extraction job of one browser session. The session
# keeps only the job ID and a random session token. The job manager keeps the
# job itself, so the job outlives a rerun of the page.
SESSION_TOKEN_KEY = "_job_session_token"
JOB_ID_KEY = "_job_id"
# The progress segments of the job (from _progress_segments), for the progress bar.
JOB_SEGMENTS_KEY = "_job_segments"
# Local mode: the absolute output folder of the job, for the output paths.
JOB_OUTPUT_DIR_KEY = "_job_output_dir"
# The job manager's rejection of the last run click: (message, reason, key
# hash). The key hash is set only for the reason "key". The rejection stays
# until the next run click, so that "Cancel the earlier job" can be clicked.
JOB_REJECTION_KEY = "_job_rejection"
# A message that the page shows once, after "Cancel the earlier job".
JOB_NOTICE_KEY = "_job_notice"
# The parsed upload of section 1: (file_id, points, validation, error message).
# A rerun then does not parse the same file again.
POINTS_CACHE_KEY = "_points_upload"

# The running-job fragment asks the job manager for the status this often, in
# seconds. Each request is also the heartbeat for the hosted abandon time-out.
JOB_POLL_INTERVAL_S = 2
JOB_LOST_MESSAGE = "The job state is lost. Start the extraction again."
RESULTS_DELETED_MESSAGE = "The results were deleted after the retention time."


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


def _dataset_type(dataset_name: str, catalog: dict[str, dict[str, Any]]) -> str:
    """Return a dataset's type with a safe label for missing catalog metadata."""

    category = catalog.get(dataset_name, {}).get("category")
    if isinstance(category, str) and category.strip():
        return category.strip()
    return UNCATEGORISED_LABEL


def _ordered_dataset_types(catalog: dict[str, dict[str, Any]]) -> list[str]:
    """Return present dataset types in the order used by the dataset docs."""

    present = {_dataset_type(name, catalog) for name in catalog}
    ordered = [dataset_type for dataset_type in DATASET_TYPE_ORDER if dataset_type in present]
    ordered.extend(
        sorted(
            present.difference(DATASET_TYPE_ORDER, {UNCATEGORISED_LABEL}),
            key=str.casefold,
        )
    )
    if UNCATEGORISED_LABEL in present:
        ordered.append(UNCATEGORISED_LABEL)
    return ordered


def _dataset_names_for_type(catalog: dict[str, dict[str, Any]], dataset_type: str) -> list[str]:
    """Return display-name-sorted dataset keys for one type or the full catalog."""

    dataset_types = _ordered_dataset_types(catalog)
    type_order = {name: index for index, name in enumerate(dataset_types)}
    names = [
        name
        for name in catalog
        if dataset_type == ALL_DATASET_TYPES or _dataset_type(name, catalog) == dataset_type
    ]
    return sorted(
        names,
        key=lambda name: (
            type_order[_dataset_type(name, catalog)] if dataset_type == ALL_DATASET_TYPES else 0,
            _dataset_display_name(name, catalog).casefold(),
            name.casefold(),
        ),
    )


def _type_option_label(dataset_type: str, catalog: dict[str, dict[str, Any]]) -> str:
    """Format a type filter option with its live dataset count."""

    if dataset_type == ALL_DATASET_TYPES:
        return f"All categories ({len(catalog)})"
    count = sum(_dataset_type(name, catalog) == dataset_type for name in catalog)
    return f"{dataset_type} ({count})"


def _dataset_option_label(
    dataset_name: str,
    catalog: dict[str, dict[str, Any]],
    *,
    include_type: bool,
) -> str:
    """Format a product option, including its type when viewing all products."""

    display_name = _dataset_display_name(dataset_name, catalog)
    if include_type:
        return f"{_dataset_type(dataset_name, catalog)} · {display_name}"
    return display_name


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
        "output_type": None,
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


def _load_uploaded_points(
    st, uploaded_file, limits: HostedLimits | None
) -> tuple[pd.DataFrame | None, CsvValidationResult | None, str | None]:
    """Parse and validate the uploaded points CSV once per uploaded file.

    In hosted mode, :func:`check_upload` checks the size and the line count of
    the raw bytes first, and a file that fails is not parsed. The outcome is
    kept in ``st.session_state`` under the file's ``file_id``, so a rerun of the
    page does not parse the same file again. Removing the upload removes the
    kept outcome.

    Args:
        st: The Streamlit module.
        uploaded_file: The value of the "Location CSV" uploader, or None.
        limits: The hosted limits, or None in local mode.

    Returns:
        ``(points, validation, error)``: the parsed points and their validation
        result, or None and None with a message for the user. All three are
        None when no file is uploaded.
    """
    if uploaded_file is None:
        st.session_state.pop(POINTS_CACHE_KEY, None)
        return None, None, None

    cached = st.session_state.get(POINTS_CACHE_KEY)
    if cached is not None and cached[0] == uploaded_file.file_id:
        return cached[1], cached[2], cached[3]

    points_df: pd.DataFrame | None = None
    validation: CsvValidationResult | None = None
    points_error: str | None = None
    upload_messages = (
        check_upload(uploaded_file.size, uploaded_file.getvalue(), limits)
        if limits is not None
        else []
    )
    if upload_messages:
        points_error = " ".join(upload_messages)
    else:
        try:
            points_df = read_points_csv(uploaded_file)
            validation = validate_points_dataframe(points_df)
        # Broad catch: pandas raises many error types for a file that is not a
        # readable CSV. Each one becomes a message for the user, not a crash.
        except Exception as exc:
            points_df, validation, points_error = None, None, str(exc)

    st.session_state[POINTS_CACHE_KEY] = (
        uploaded_file.file_id,
        points_df,
        validation,
        points_error,
    )
    return points_df, validation, points_error


def _is_wsl() -> bool:
    """Return whether the web app is running under Windows Subsystem for Linux."""

    return bool(os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"))


def _windows_folder_picker_script(initial_path: str) -> str:
    """Build a PowerShell script using Windows' modern folder picker."""

    escaped_initial_path = _escape_powershell_string(initial_path)
    return rf"""
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

namespace EnvoiWebApp {{
    [Flags]
    internal enum FileOpenOptions : uint {{
        PickFolders = 0x00000020,
        ForceFileSystem = 0x00000040,
        PathMustExist = 0x00000800
    }}

    internal enum DisplayName : uint {{
        FileSystemPath = 0x80058000
    }}

    [ComImport]
    [Guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")]
    internal class FileOpenDialog {{}}

    [ComImport]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    [Guid("42F85136-DB7E-439C-85F1-E4075D135FC8")]
    internal interface IFileOpenDialog {{
        [PreserveSig] int Show(IntPtr parent);
        void SetFileTypes(uint count, IntPtr filterSpec);
        void SetFileTypeIndex(uint index);
        void GetFileTypeIndex(out uint index);
        void Advise(IntPtr events, out uint cookie);
        void Unadvise(uint cookie);
        void SetOptions(FileOpenOptions options);
        void GetOptions(out FileOpenOptions options);
        void SetDefaultFolder(IShellItem item);
        void SetFolder(IShellItem item);
        void GetFolder(out IShellItem item);
        void GetCurrentSelection(out IShellItem item);
        void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
        void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string name);
        void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
        void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string text);
        void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
        void GetResult(out IShellItem item);
        void AddPlace(IShellItem item, int alignment);
        void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string extension);
        void Close(int result);
        void SetClientGuid(ref Guid guid);
        void ClearClientData();
        void SetFilter(IntPtr filter);
    }}

    [ComImport]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    [Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE")]
    internal interface IShellItem {{
        void BindToHandler(IntPtr bindContext, ref Guid handler, ref Guid iid, out IntPtr result);
        void GetParent(out IShellItem parent);
        void GetDisplayName(DisplayName displayName, out IntPtr name);
        void GetAttributes(uint mask, out uint attributes);
        void Compare(IShellItem other, uint hint, out int order);
    }}

    public static class FolderPicker {{
        [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
        private static extern void SHCreateItemFromParsingName(
            string path,
            IntPtr bindContext,
            ref Guid iid,
            [MarshalAs(UnmanagedType.Interface)] out IShellItem item
        );

        public static string Pick(string initialPath) {{
            IFileOpenDialog dialog = (IFileOpenDialog)new FileOpenDialog();
            IShellItem initialFolder = null;
            IShellItem selectedFolder = null;
            IntPtr selectedPath = IntPtr.Zero;
            try {{
                FileOpenOptions options;
                dialog.GetOptions(out options);
                dialog.SetOptions(
                    options
                    | FileOpenOptions.PickFolders
                    | FileOpenOptions.ForceFileSystem
                    | FileOpenOptions.PathMustExist
                );
                dialog.SetTitle("Select envoi output directory");

                if (!String.IsNullOrWhiteSpace(initialPath)) {{
                    try {{
                        Guid shellItemId = typeof(IShellItem).GUID;
                        SHCreateItemFromParsingName(
                            initialPath, IntPtr.Zero, ref shellItemId, out initialFolder
                        );
                        dialog.SetFolder(initialFolder);
                    }} catch {{
                        // The picker can still open at its default location.
                    }}
                }}

                int result = dialog.Show(IntPtr.Zero);
                if (result == unchecked((int)0x800704C7)) {{
                    return null;
                }}
                if (result != 0) {{
                    Marshal.ThrowExceptionForHR(result);
                }}

                dialog.GetResult(out selectedFolder);
                selectedFolder.GetDisplayName(DisplayName.FileSystemPath, out selectedPath);
                return Marshal.PtrToStringUni(selectedPath);
            }} finally {{
                if (selectedPath != IntPtr.Zero) Marshal.FreeCoTaskMem(selectedPath);
                if (selectedFolder != null) Marshal.FinalReleaseComObject(selectedFolder);
                if (initialFolder != null) Marshal.FinalReleaseComObject(initialFolder);
                Marshal.FinalReleaseComObject(dialog);
            }}
        }}
    }}
}}
'@

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$selected = [EnvoiWebApp.FolderPicker]::Pick('{escaped_initial_path}')
if ($null -ne $selected) {{ Write-Output $selected }}
""".strip()


def _choose_windows_output_directory(powershell: str, initial_path: str) -> str | None:
    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-STA",
            "-Command",
            _windows_folder_picker_script(initial_path),
        ],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Windows folder chooser failed.")
    return result.stdout.strip() or None


def _convert_wsl_path(path: str, direction: str) -> str:
    """Convert between WSL and Windows paths with the system's wslpath tool."""

    wslpath = shutil.which("wslpath")
    if wslpath is None:
        raise RuntimeError("The WSL path conversion tool (wslpath) is unavailable.")
    result = subprocess.run(
        [wslpath, direction, path],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"Could not convert WSL path: {path}")
    return result.stdout.strip()


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
            try:
                return _choose_windows_output_directory(powershell, str(initial_path))
            except RuntimeError as exc:
                errors.append(str(exc))

    elif sys.platform.startswith("linux"):
        if _is_wsl():
            powershell = shutil.which("powershell.exe")
            if powershell:
                try:
                    windows_initial_path = _convert_wsl_path(str(initial_path), "-w")
                    selected_path = _choose_windows_output_directory(
                        powershell, windows_initial_path
                    )
                    if selected_path is None:
                        return None
                    return _convert_wsl_path(selected_path, "-u")
                except RuntimeError as exc:
                    errors.append(str(exc))

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
                "output_type_select_",
                "dataset_type_select_",
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


def _apply_row_output_type(st, index: int, row_widget_key: str, output_type: str | None) -> None:
    """Store one row's output type, and clear its statistics state when the type changed.

    Point, window, and statistic choices exist only for tabular rows. When a
    row's type changes, this function resets these choices and removes their
    widget state for that row only. A later change back to tabular then starts
    with no choices, and does not restore choices that were hidden. Other rows,
    and the row's data product and window sizes, keep their values.

    Args:
        st: The Streamlit module.
        index: Position of the row in ``st.session_state.dataset_rows``.
        row_widget_key: The ``<widget version>_<index>`` suffix of the row's widget keys.
        output_type: The type that the row's "Output type" widget returned, or None.
    """

    row = st.session_state.dataset_rows[index]
    if row.get("output_type") == output_type:
        return

    row["output_type"] = output_type
    row["sample_point"] = False
    row["sample_window"] = False
    row["statistics"] = []
    row.pop("statistics_dataset", None)

    # The trailing "_" of the statistics prefix stops row 1 from matching row 10.
    checkbox_keys = {f"point_checkbox_{row_widget_key}", f"window_checkbox_{row_widget_key}"}
    statistics_prefix = f"stats_select_{row_widget_key}_"
    for key in list(st.session_state.keys()):
        key_text = str(key)
        if key_text in checkbox_keys or key_text.startswith(statistics_prefix):
            st.session_state.pop(key, None)


def _render_dataset_rows(st, catalog: dict[str, dict[str, Any]]) -> None:
    reducers = list_reducers()
    if not catalog:
        st.error("No data products are available in the envoi catalog.")
        return

    _ensure_dataset_state()
    _apply_pending_dataset_remove(st)
    widget_version = int(st.session_state.get("_dataset_widget_version", 0))
    type_options = [ALL_DATASET_TYPES, *_ordered_dataset_types(catalog)]
    output_type_options = [TABULAR_OUTPUT, RASTER_OUTPUT]

    for index, row in enumerate(st.session_state.dataset_rows):
        with st.expander(f"Data product {index + 1}", expanded=True):
            row_widget_key = f"{widget_version}_{index}"

            # First line: the output type and the data product. The other
            # controls depend on the output type, so they appear only after
            # the user chooses it.
            top_cols = st.columns([0.18, 0.29, 0.53], vertical_alignment="bottom")
            current_output_type = row.get("output_type")
            output_type = top_cols[0].selectbox(
                "Output type",
                output_type_options,
                index=(
                    output_type_options.index(current_output_type)
                    if current_output_type in output_type_options
                    else None
                ),
                placeholder="Choose output type",
                format_func=str.title,
                key=f"output_type_select_{row_widget_key}",
            )
            _apply_row_output_type(st, index, row_widget_key, output_type)
            selected_type = top_cols[1].selectbox(
                "Category",
                type_options,
                index=0,
                key=f"dataset_type_select_{row_widget_key}",
                format_func=lambda dataset_type: _type_option_label(dataset_type, catalog),
            )
            dataset_names = _dataset_names_for_type(catalog, selected_type)
            current_dataset = row.get("dataset") if row.get("dataset") in dataset_names else None
            include_type = selected_type == ALL_DATASET_TYPES
            selected_dataset = top_cols[2].selectbox(
                "Data product",
                dataset_names,
                index=dataset_names.index(current_dataset) if current_dataset else None,
                placeholder="Choose a data product",
                key=f"dataset_select_{row_widget_key}",
                # Bind include_type as a default so the label uses this row's value.
                format_func=lambda name, include_type=include_type: _dataset_option_label(
                    name,
                    catalog,
                    include_type=include_type,
                ),
            )
            windows = str(row.get("window_sizes", ""))
            st.session_state.dataset_rows[index]["dataset"] = selected_dataset or ""

            if output_type == TABULAR_OUTPUT:
                sampling_cols = st.columns([0.12, 0.12, 0.76], vertical_alignment="bottom")
                sample_point = sampling_cols[0].checkbox(
                    "Point",
                    value=bool(row.get("sample_point", False)),
                    key=f"point_checkbox_{row_widget_key}",
                    help="Sample the raster pixel value at each coordinate.",
                )
                sample_window = sampling_cols[1].checkbox(
                    "Window",
                    value=bool(row.get("sample_window", False)),
                    key=f"window_checkbox_{row_widget_key}",
                    help="Calculate spatial statistics within one or more sampling windows.",
                )
                st.caption(
                    "Coordinate point values as well as spatial statistics over sampling "
                    "window(s) can be extracted."
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
                        st.session_state.dataset_rows[index][
                            "statistics_dataset"
                        ] = selected_dataset
                    else:
                        st.session_state.dataset_rows[index].pop("statistics_dataset", None)
            elif output_type == RASTER_OUTPUT:
                window_cols = st.columns([0.35, 0.65], vertical_alignment="bottom")
                windows = window_cols[0].text_input(
                    "Window size(s) in meters",
                    value=row.get("window_sizes", ""),
                    placeholder="e.g. 500, 1000",
                    key=f"windows_input_{row_widget_key}",
                )
                st.caption(
                    "The window size(s) determines the size of the extracted raster tiles. "
                    "Note that raster outputs use 10 m resampling of source data by default, "
                    "to ensure consistency in spatial resolution between data products."
                )
                st.session_state.dataset_rows[index]["window_sizes"] = windows

            remove_disabled = not (
                output_type
                or selected_type != ALL_DATASET_TYPES
                or selected_dataset
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
    catalog: dict[str, dict[str, Any]],
    widget_version: int,
) -> tuple[list[_ValidationIssue], list[DatasetSelection]]:
    """Validate visible data-product fields in their left-to-right order.

    Each row is checked with its own output type. A row without an output type
    or without a data product gets an issue for each missing choice, and its
    other fields are not checked.
    """

    issues: list[_ValidationIssue] = []
    selections: list[DatasetSelection] = []
    if not rows:
        return [_ValidationIssue("Step 4 — Add at least one data product.", ())], selections

    for index, row in enumerate(rows):
        row_widget_key = f"{widget_version}_{index}"
        output_type = row.get("output_type")
        dataset = str(row.get("dataset") or "")
        label = _dataset_label(dataset, index, catalog)
        if output_type not in {TABULAR_OUTPUT, RASTER_OUTPUT}:
            issues.append(
                _ValidationIssue(
                    f"Step 4 — {label}: choose an output type (Tabular or Raster).",
                    (f"output_type_select_{row_widget_key}",),
                )
            )
        if not dataset:
            issues.append(
                _ValidationIssue(
                    f"Step 4 — {label}: choose a data product.",
                    (f"dataset_select_{row_widget_key}",),
                )
            )
        # Point/window and their dependent fields are hidden or irrelevant
        # until both the output type and the product are chosen.
        if output_type not in {TABULAR_OUTPUT, RASTER_OUTPUT} or not dataset:
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
                output_type=output_type,
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
    output_dir: str | None,
    dataset_rows: list[dict],
    catalog: dict[str, dict[str, Any]],
    widget_version: int,
    limits: HostedLimits | None = None,
) -> _FormValidation:
    """Collect form errors in step order without running the extraction.

    Args:
        points_df: The parsed points, or None when the upload is missing or bad.
        points_error: The message for a bad upload, or None.
        input_crs: The CRS text of step 1.
        credentials_bytes: The uploaded key, or None.
        output_dir: The output folder text of step 3. None in hosted mode, which
            has no output-folder field.
        dataset_rows: The data-product rows of step 4.
        catalog: The dataset catalog.
        widget_version: The widget version of the data-product rows.
        limits: The hosted limits, or None in local mode. The limit checks run
            only when every data-product row is valid, so that the row numbers
            in their messages match the rows of the form.

    Returns:
        The issues in step order, the valid data-product selections, and the
        normalized CRS.
    """

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

    # Step 3: output directory (local mode only).
    if output_dir is not None and not output_dir.strip():
        issues.append(
            _ValidationIssue(
                "Step 3 — Enter an output directory.",
                ("output_dir",),
            )
        )

    # Step 4: data-product rows, each with its own output type.
    dataset_issues, selections = _validate_dataset_rows(dataset_rows, catalog, widget_version)
    issues.extend(dataset_issues)

    # Hosted limits. check_hosted_limits() numbers the rows by their position
    # in ``selections``, which matches the form only when no row was skipped.
    if limits is not None and points_df is not None and not dataset_issues:
        issues.extend(
            _ValidationIssue(f"Server limit — {message}", ())
            for message in check_hosted_limits(points_df, selections, limits)
        )

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


def _job_progress(
    expected_segments: dict[tuple[str, str, int, str], int],
    progress: Sequence[ProgressMessage],
) -> tuple[float, str]:
    """Return the fraction done and the progress-bar text of a running job.

    Args:
        expected_segments: The segments of the job from :func:`_progress_segments`,
            each with the number of points as its expected total.
        progress: The latest progress message of each segment, in the order in
            which the segments first reported (``JobSnapshot.progress``). A
            message replaces the expected total of its segment with the real one.

    Returns:
        ``(fraction, text)``: the fraction of all segments that is done, from 0
        to 1, and a text about the segment that reported last.
    """
    segment_totals = dict(expected_segments)
    completed_by_segment: dict[tuple[str, str, int, str], int] = {}
    for message in progress:
        segment_key = (
            message["batch_id"],
            message["dataset"],
            message["window_size_m"],
            message["mode"],
        )
        segment_totals[segment_key] = max(message["total"], 1)
        completed_by_segment[segment_key] = message["completed"]

    total = sum(segment_totals.values()) or 1
    completed = sum(
        min(completed_by_segment.get(segment_key, 0), segment_total)
        for segment_key, segment_total in segment_totals.items()
    )
    fraction = min(1.0, completed / total)
    if not progress:
        return fraction, "Starting extraction"

    latest = progress[-1]
    window_text = "point" if latest["window_size_m"] == 0 else f"{latest['window_size_m']} m"
    return fraction, (
        f"{latest['dataset']} | {window_text} | "
        f"{latest['completed']}/{latest['total']} {latest['unit']}"
    )


def _warning_count_text(warning_count: int) -> str:
    """Return the number of run-log records as a text for the user.

    For example ``"3 warnings in the run log."``. envoi can report one warning
    per input row (for example for incomplete dates), so the count can be
    large. The text calls the records warnings, not errors.
    """
    if warning_count == 0:
        return "No warnings in the run log."
    noun = "warning" if warning_count == 1 else "warnings"
    return f"{warning_count:,} {noun} in the run log."


def _hosted_notice(limits: HostedLimits) -> str:
    """Return the hosted-mode notice about the key, the data, the limits, and downloads (R19).

    The numbers come from ``limits``, so the notice always states the limits
    that the web app applies.
    """
    data_handling = [
        (
            "**Your key** stays in the server's memory for this browser session only. "
            "It is never written to disk."
        ),
        (
            "**Your points and the results** stay on the server's disk only while they are "
            f"needed. The results are deleted {format_minutes(limits.retention_after_download_s)} "
            "after the first download, or at the latest "
            f"{format_minutes(limits.max_retention_s)} after the extraction ends. Like your "
            "key, the uploaded points file stays in the server's memory for this browser "
            "session, and the results contain the IDs and coordinates of your points."
        ),
        (
            "**Keep this page open and visible** while an extraction runs. If the page is "
            "closed or the computer sleeps, the extraction stops after "
            f"{format_minutes(limits.abandon_timeout_s)}."
        ),
        "**A server restart** stops running extractions and deletes all results.",
        (
            "**Downloads** go to the download folder of your browser. Most browsers can be "
            "set to ask where to save each download."
        ),
        (
            "**For larger runs**, or to write the results straight into a folder, "
            f"[install envoi]({LOCAL_INSTALL_URL}) and use the local web app or the Python "
            "package."
        ),
    ]
    service_limits = [
        (
            f"CSV file: at most {format_bytes(limits.max_upload_bytes)} and "
            f"{limits.max_input_rows:,} rows."
        ),
        f"At most {limits.max_dataset_rows} data products per extraction.",
        (
            f"Tabular rows: at most {limits.tabular_request_budget:,} points × window sizes "
            "(the point value counts as one window size), and windows of at most "
            f"{limits.max_tabular_window_m:,} m."
        ),
        (
            f"Raster rows: at most {limits.raster_tile_budget:,} points × window sizes, and "
            f"windows of at most {limits.max_raster_window_m:,} m."
        ),
        (
            f"Each extraction: at most {format_minutes(limits.max_run_time_s)} and "
            f"{format_bytes(limits.max_workspace_bytes)} of results. Raster tiles of data "
            "products with many bands (for example satellite embeddings) are large, so use "
            "fewer points or smaller windows for them."
        ),
        (
            "One extraction at a time per page and per key, and at most "
            f"{limits.max_concurrent_jobs} at a time on the server."
        ),
    ]
    return "\n".join(
        [
            "**How this service handles your data**",
            "",
            *(f"- {line}" for line in data_handling),
            "",
            "**Limits of this service**",
            "",
            *(f"- {line}" for line in service_limits),
        ]
    )


# ---------------------------------------------------------------------------
# Extraction jobs
# ---------------------------------------------------------------------------


def _archive_download(manager: JobManager, job_id: str, archive_path: Path) -> Callable[[], bytes]:
    """Return the function that "Download results" calls to get the ZIP archive.

    Streamlit calls the function when the user clicks the button, in a separate
    thread and without a rerun of the page. The function records the click with
    :meth:`JobManager.mark_downloaded`, which starts the retention time after
    the first download, and returns the archive bytes.

    When the archive cannot be read (the retention time ended between the page
    update and the click), the function raises ``RuntimeError`` with
    :data:`RESULTS_DELETED_MESSAGE`. Streamlit catches it, logs it, and shows
    "Failed to generate file for download" below the button. The user never
    gets an empty or partial file, and the next rerun of the page shows that
    the results were deleted.
    """

    def read_archive() -> bytes:
        manager.mark_downloaded(job_id)
        try:
            return archive_path.read_bytes()
        except OSError:
            pass
        # Raised outside the except block, so that the server log shows no
        # chained error with the path of the job workspace.
        raise RuntimeError(RESULTS_DELETED_MESSAGE)

    return read_archive


def _clear_job(st) -> None:
    """Remove the job of this session from ``st.session_state``."""
    for key in (JOB_ID_KEY, JOB_SEGMENTS_KEY, JOB_OUTPUT_DIR_KEY):
        st.session_state.pop(key, None)


def _submit_job(
    st,
    manager: JobManager,
    settings: WebappSettings,
    validation: _FormValidation,
    points_df: pd.DataFrame,
    credentials_bytes: bytes,
    output_dir: str | None,
) -> None:
    """Start a job for a valid form, keep its ID in the session, and rerun the page.

    The rerun shows the running job and disables the run button. A rejection
    by the job manager (:class:`JobRejected`) is kept in the session until the
    next run click. A bad output folder (local mode) or a key that the job
    manager cannot read is shown as a form error.

    Args:
        st: The Streamlit module.
        manager: The job manager.
        settings: The web-app settings.
        validation: The form validation, without issues.
        points_df: The parsed points.
        credentials_bytes: The uploaded key.
        output_dir: The output folder text in local mode, None in hosted mode.
    """
    run_configs = build_run_config(list(validation.selections))

    # Local mode: create the output folder now, and check that it is writable.
    # Hosted mode: the job manager chooses the folder.
    output_folder: Path | None = None
    if not settings.is_hosted:
        try:
            output_folder = validate_output_dir(output_dir)
        except (ValueError, OSError) as exc:
            _render_validation_issues(st, (_ValidationIssue(f"Step 3 — {exc}", ("output_dir",)),))
            return

    try:
        job_id = manager.submit(
            session_token=st.session_state[SESSION_TOKEN_KEY],
            points=points_df,
            run_configs=run_configs,
            input_crs=validation.normalized_crs,
            credentials_json=credentials_bytes,
            output_dir=output_folder,
        )
    except JobRejected as rejection:
        # For a running job with the same key, keep the hash of the key, so that
        # "Cancel the earlier job" can cancel it without the key itself.
        rejected_key_hash = key_hash(credentials_bytes) if rejection.reason == "key" else None
        st.session_state[JOB_REJECTION_KEY] = (str(rejection), rejection.reason, rejected_key_hash)
        return
    # The key is not UTF-8 text or not a JSON object. The message has no key material.
    except ValueError as exc:
        _render_validation_issues(st, (_ValidationIssue(f"Step 2 — {exc}", ("credentials_json",)),))
        return

    st.session_state[JOB_ID_KEY] = job_id
    st.session_state[JOB_SEGMENTS_KEY] = _progress_segments(run_configs, len(points_df))
    st.session_state[JOB_OUTPUT_DIR_KEY] = output_folder
    st.rerun()


def _render_job_rejection(st, manager: JobManager) -> None:
    """Show the kept job rejection, and "Cancel the earlier job" for a rejection by key.

    A session that uploaded the key of a running job may cancel that job, for
    example after a page reload lost the link to it. Before that, the page
    shows once the notice that the cancel left.
    """
    notice = st.session_state.pop(JOB_NOTICE_KEY, None)
    if notice is not None:
        st.info(notice)

    rejection = st.session_state.get(JOB_REJECTION_KEY)
    if rejection is None:
        return
    message, _reason, rejected_key_hash = rejection
    st.warning(message)
    if rejected_key_hash is not None and st.button(
        "Cancel the earlier job", key="cancel_earlier_job"
    ):
        with st.spinner("Stopping the earlier extraction..."):
            cancelled = manager.cancel_for_key(rejected_key_hash)
        st.session_state.pop(JOB_REJECTION_KEY, None)
        st.session_state[JOB_NOTICE_KEY] = (
            "The earlier extraction was cancelled. "
            if cancelled
            else "The earlier extraction had already ended. "
        ) + "Click “Extract selected data” to start your extraction."
        st.rerun()


def _render_running_job(st, manager: JobManager, job_id: str) -> None:
    """Show the progress of the running job and "Cancel", updated every 2 seconds.

    A fragment reruns only this part of the page. Each rerun asks the job
    manager for the status, which is also the job's heartbeat. When the job
    has ended, the fragment reruns the whole page, which then shows the result
    and enables the run button again.
    """
    expected_segments = st.session_state.get(JOB_SEGMENTS_KEY, {})

    @st.fragment(run_every=JOB_POLL_INTERVAL_S)
    def show_running_job() -> None:
        snapshot = manager.snapshot(job_id)
        if snapshot is None or snapshot.state.is_final:
            st.rerun(scope="app")
        fraction, text = _job_progress(expected_segments, snapshot.progress)
        st.progress(fraction, text=text)
        if snapshot.progress:
            st.caption(f"Current batch: {snapshot.progress[-1]['batch_id']}")
        if st.button("Cancel", key="cancel_job"):
            with st.spinner("Stopping the extraction..."):
                manager.cancel(job_id)
            st.rerun(scope="app")

    show_running_job()


def _render_succeeded_job(
    st, manager: JobManager, settings: WebappSettings, snapshot: JobSnapshot
) -> None:
    """Show a succeeded job: the warning count, and the download (hosted) or the paths (local)."""
    result = snapshot.result
    st.success("Extraction complete.")
    st.info(_warning_count_text(result["warning_count"]))
    if settings.is_hosted:
        if not snapshot.archive_available:
            st.info(RESULTS_DELETED_MESSAGE)
            return
        limits = settings.limits
        archive_path = Path(result["archive"])
        st.download_button(
            "Download results",
            data=_archive_download(manager, snapshot.job_id, archive_path),
            file_name=archive_path.name,
            mime="application/zip",
            on_click="ignore",
            type="primary",
            key="download_results",
        )
        st.caption(
            "The ZIP file holds the outputs and the run log. Results are deleted "
            f"{format_minutes(limits.retention_after_download_s)} after the first download, "
            f"or {format_minutes(limits.max_retention_s)} after the job ended."
        )
        return

    # Local mode: the worker gives the output paths relative to the output folder.
    output_folder = Path(st.session_state[JOB_OUTPUT_DIR_KEY])
    st.write("Outputs")
    for output_key, relative_path in result["outputs"].items():
        st.code(f"{output_key}: {output_folder / relative_path}")
    st.code(f"run log: {output_folder / result['run_log']}")


def _render_failed_job(st, snapshot: JobSnapshot) -> None:
    """Show a failed job: the message, the warning count, and the last lines of the run log.

    The worker removed key material from the message and the lines. In hosted
    mode the job workspace is already deleted, so the page names no run-log file.
    """
    error = snapshot.error
    st.error(f"The extraction failed. {error['message']}")
    if error["warning_count"]:
        st.info(_warning_count_text(error["warning_count"]))
    if error["run_log_tail"]:
        st.write("Last lines of the run log:")
        st.code("\n".join(error["run_log_tail"]), language=None)


def _render_job(
    st,
    manager: JobManager,
    settings: WebappSettings,
    job_id: str,
    snapshot: JobSnapshot | None,
) -> None:
    """Show the job of this session: its progress while it runs, or its end state.

    Each end state, and a job that the job manager does not know (for example
    after it forgot an old job), gets "Clear results". It discards the job (in
    hosted mode, the results are deleted) and removes it from the session.
    """
    if snapshot is not None and snapshot.state is JobState.RUNNING:
        _render_running_job(st, manager, job_id)
        return

    if snapshot is None:
        st.warning(JOB_LOST_MESSAGE)
    elif snapshot.state is JobState.SUCCEEDED:
        _render_succeeded_job(st, manager, settings, snapshot)
    elif snapshot.state is JobState.FAILED:
        _render_failed_job(st, snapshot)
    else:
        # Cancelled or stopped by a limit: the job manager stopped the worker
        # before it could write its run log, so only the reason is known.
        st.warning(snapshot.stop_reason or "The extraction was stopped.")

    if st.button("Clear results", key="clear_results"):
        manager.discard(job_id)
        _clear_job(st)
        st.rerun()


def _render_run_section(
    st,
    *,
    settings: WebappSettings,
    catalog: dict[str, dict[str, Any]],
    points_df: pd.DataFrame | None,
    points_error: str | None,
    input_crs: str,
    credentials_bytes: bytes | None,
    output_dir: str | None,
) -> None:
    """Render the run button, the form errors, and the job of this session (section 5).

    The run button is disabled while the session's job runs. A configuration
    error of the job manager replaces the whole section, so that no job starts.
    """
    # The one job manager of this server process (jobs.get_job_manager()). It
    # is not in the Streamlit cache, so a client's "Clear cache" cannot replace
    # it. Tests replace app.get_job_manager with a fake manager.
    try:
        manager = get_job_manager()
    except ValueError as exc:
        st.error(f"The web app cannot run extractions. {exc}")
        return

    # The status of the session's job decides whether the run button is active.
    # Each status request is also the job's heartbeat.
    job_id = st.session_state.get(JOB_ID_KEY)
    snapshot = manager.snapshot(job_id) if job_id is not None else None
    job_running = snapshot is not None and snapshot.state is JobState.RUNNING

    run_button = st.button(
        "Extract selected data",
        type="primary",
        disabled=job_running,
        key="run_extraction_button",
    )
    if run_button:
        st.session_state.pop(JOB_REJECTION_KEY, None)
        validation = _validate_form(
            points_df=points_df,
            points_error=points_error,
            input_crs=input_crs,
            credentials_bytes=credentials_bytes,
            output_dir=output_dir,
            dataset_rows=list(st.session_state.get("dataset_rows", [])),
            catalog=catalog,
            widget_version=int(st.session_state.get("_dataset_widget_version", 0)),
            limits=settings.limits,
        )
        if validation.issues:
            _render_validation_issues(st, validation.issues)
        else:
            _submit_job(st, manager, settings, validation, points_df, credentials_bytes, output_dir)
    _render_job_rejection(st, manager)

    if job_id is not None:
        _render_job(st, manager, settings, job_id, snapshot)


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

    # The mode and, in hosted mode, the limits. Invalid settings stop the page
    # here, so that no form appears in the wrong mode (for example an
    # output-folder field on a public server).
    try:
        settings = load_settings()
    except ValueError as exc:
        st.error(f"The web app is not configured correctly. {exc}")
        _render_footer(st)
        return
    # The job manager allows one running job per session token.
    if SESSION_TOKEN_KEY not in st.session_state:
        st.session_state[SESSION_TOKEN_KEY] = secrets.token_urlsafe(16)

    catalog = _dataset_catalog()

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
    points_df, points_validation, points_error = _load_uploaded_points(
        st, uploaded_csv, settings.limits
    )
    if points_validation is not None:
        st.success(
            f"Loaded {points_validation.row_count} rows. "
            f"Date column present: {'yes' if points_validation.has_date else 'no'}."
        )
        st.dataframe(points_df.head(20), width="stretch")
    elif points_error is not None:
        st.error(points_error)

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
        Upload your Google Earth Engine service account JSON key. The web app keeps
        the key in memory for this browser session only and never writes it to disk.
        Each extraction runs in a separate process that uses only your key. If you do
        not have a service account yet, follow the
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

    # Hosted mode has no output-folder field: the user downloads the results,
    # and no widget accepts a server path.
    output_dir: str | None = None
    if settings.is_hosted:
        st.subheader("3. Results and your data")
        st.info(_hosted_notice(settings.limits))
    else:
        st.subheader("3. Choose output settings")
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
    st.markdown(
        f"""
        Add one entry per Earth Engine data product that should be downloaded. If a data product contains multiple bands, all of them will be processed and downloaded. For information about available data products, see the
        <a href="{DATASET_CATALOG_URL}" target="_blank">envoi catalog</a>.
        """,
        unsafe_allow_html=True,
    )
    _render_dataset_rows(st, catalog)

    st.subheader("5. Run extraction")
    if settings.is_hosted:
        st.write(
            "The final outputs, data quality checks, metadata, and a run log are packed into "
            "one ZIP file. A download button appears here when the extraction is complete."
        )
    else:
        st.write(
            "The final outputs, data quality checks, metadata, and a run log are written to "
            "the output directory chosen in step 3."
        )
    _render_run_section(
        st,
        settings=settings,
        catalog=catalog,
        points_df=points_df,
        points_error=points_error,
        input_crs=input_crs,
        credentials_bytes=credentials_bytes,
        output_dir=output_dir,
    )

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
