"""Settings of the envoi web app: the mode, the job workspace folder, and the hosted limits.

The web app runs in one of two modes:

* ``local`` (the default): one user runs the app on their own computer through
  ``envoi-webapp``. No limits apply.
* ``hosted``: the app runs as a shared public service. The limits in
  :class:`HostedLimits` apply to each upload and each job.

:func:`load_settings` reads the mode and the other settings from environment
variables once, at the boundary, into a :class:`WebappSettings` object.
:func:`check_upload` and :func:`check_hosted_limits` check an upload and a form
against the hosted limits before the web app parses the file or starts a job.
They return messages for the user and raise nothing.

The module starts no processes, makes no Earth Engine calls, and creates no
folders.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from envoi_webapp.helpers import RASTER_OUTPUT, TABULAR_OUTPUT, DatasetSelection

# Size units. Streamlit's server.maxUploadSize counts a megabyte as 1024 * 1024
# bytes, so the app-level upload check uses the same unit and the two limits
# agree.
_BYTES_PER_MB = 1024 * 1024
_BYTES_PER_GB = 1024 * 1024 * 1024

# The job workspaces are in a folder with this name in the workspace parent
# folder, so that the clean-up at start-up in hosted mode can never touch other
# files in the parent folder. In local mode, the job manager adds a unique
# suffix for each server process (see WebappSettings).
WORKSPACE_SUBFOLDER_NAME = "envoi-webapp-jobs"

# Environment variables that load_settings() reads.
MODE_VARIABLE = "ENVOI_WEBAPP_MODE"
WORKSPACE_VARIABLE = "ENVOI_WEBAPP_WORKSPACE"
MAX_JOBS_VARIABLE = "ENVOI_WEBAPP_MAX_JOBS"

LOCAL_MODE = "local"
HOSTED_MODE = "hosted"
_DEFAULT_MAX_CONCURRENT_JOBS = 2


@dataclass(frozen=True)
class HostedLimits:
    """The limits of hosted mode (plan decision D8).

    The defaults are the decided starting values. Only ``max_concurrent_jobs``
    comes from an environment variable. Tests replace other values with
    ``dataclasses.replace()``. Sizes are in bytes and times in seconds.
    """

    # Upload checks, before the web app parses the file.
    max_upload_bytes: int = 50 * _BYTES_PER_MB
    # One header line and 10,000 data rows.
    max_upload_lines: int = 10_001

    # Form checks, before the web app starts a job.
    max_input_rows: int = 10_000
    max_dataset_rows: int = 10
    # Points x the number of window sizes over all tabular rows. The point
    # value (window size 0) counts as one window size.
    tabular_request_budget: int = 20_000
    # Points x the number of window sizes over all raster rows.
    raster_tile_budget: int = 1_000
    max_tabular_window_m: int = 10_000
    max_raster_window_m: int = 2_000

    # Job checks, applied by the job manager.
    max_concurrent_jobs: int = _DEFAULT_MAX_CONCURRENT_JOBS
    max_run_time_s: float = 60 * 60
    max_workspace_bytes: int = 1 * _BYTES_PER_GB
    # Admit a new job only if (the sizes of the finished workspaces) + (running
    # jobs + 1) x max_workspace_bytes <= disk_budget_bytes.
    disk_budget_bytes: int = 4 * _BYTES_PER_GB
    # The job manager stops a running job when no session asked for its status
    # for this time.
    abandon_timeout_s: float = 10 * 60
    # A succeeded job's workspace is deleted this long after the first click on
    # "Download results" ...
    retention_after_download_s: float = 10 * 60
    # ... and at the latest this long after the job ended.
    max_retention_s: float = 30 * 60


@dataclass(frozen=True)
class WebappSettings:
    """The settings of one web-app server process.

    ``workspace_parent`` is the folder that holds the web app's workspace root.
    The job manager (``envoi_webapp.jobs``) creates the root and one workspace
    folder per job in it:

    * hosted mode: the fixed subfolder ``envoi-webapp-jobs``, so that the next
      server process can delete the workspaces that this one left,
    * local mode: a new private folder ``envoi-webapp-jobs-<random>`` for each
      server process, so that other users of the computer cannot change it.

    ``limits`` is None in local mode, where no limits apply.
    """

    mode: Literal["local", "hosted"]
    workspace_parent: Path
    limits: HostedLimits | None

    @property
    def is_hosted(self) -> bool:
        """True in hosted mode."""
        return self.mode == HOSTED_MODE


def load_settings(environ: Mapping[str, str] = os.environ) -> WebappSettings:
    """Read the web-app settings from environment variables.

    Variables (a variable that is not set, or set to an empty value, takes its
    default):

    * ``ENVOI_WEBAPP_MODE``: ``local`` (default) or ``hosted``. Hosted mode
      runs only on Linux or macOS.
    * ``ENVOI_WEBAPP_WORKSPACE``: an absolute path of the parent folder of the
      job workspaces. The default is the system temporary folder. The job
      manager creates its workspace root in this folder (see
      :class:`WebappSettings`).
    * ``ENVOI_WEBAPP_MAX_JOBS``: the maximum number of jobs that run at the same
      time in hosted mode, a positive integer. The default is 2. Local mode
      does not read it, because no server-wide limit applies there.

    Args:
        environ: The environment variables. Tests give a dict.

    Returns:
        The settings. The function does not create the workspace folder.

    Raises:
        ValueError: when a variable has a value that the web app cannot use,
            or when hosted mode is requested on Windows.
    """
    mode = environ.get(MODE_VARIABLE, "").strip().lower() or LOCAL_MODE
    if mode not in (LOCAL_MODE, HOSTED_MODE):
        raise ValueError(
            f"{MODE_VARIABLE} must be '{LOCAL_MODE}' or '{HOSTED_MODE}', "
            f"not {environ[MODE_VARIABLE]!r}. Remove the variable to use local mode."
        )
    # Hosted mode runs in Linux containers. On Windows, the worker processes
    # would keep other credential locations of the server account (APPDATA,
    # USERPROFILE), and the job manager could not check the owner and the
    # permissions of the workspace folder.
    if mode == HOSTED_MODE and sys.platform == "win32":
        raise ValueError(
            f"{MODE_VARIABLE}={HOSTED_MODE} runs only on Linux or macOS. On Windows, remove "
            "the variable to use local mode."
        )

    workspace_text = environ.get(WORKSPACE_VARIABLE, "").strip()
    if workspace_text:
        workspace_parent = Path(workspace_text).expanduser()
        # The job workspaces become the working folders of the worker
        # processes, and the worker needs absolute output paths.
        if not workspace_parent.is_absolute():
            raise ValueError(
                f"{WORKSPACE_VARIABLE} must be an absolute folder path, not {workspace_text!r}."
            )
    else:
        workspace_parent = Path(tempfile.gettempdir())

    if mode == LOCAL_MODE:
        return WebappSettings(mode=LOCAL_MODE, workspace_parent=workspace_parent, limits=None)

    max_jobs_text = environ.get(MAX_JOBS_VARIABLE, "").strip()
    max_concurrent_jobs = _DEFAULT_MAX_CONCURRENT_JOBS
    if max_jobs_text:
        try:
            max_concurrent_jobs = int(max_jobs_text)
        except ValueError:
            max_concurrent_jobs = 0
        if max_concurrent_jobs < 1:
            raise ValueError(
                f"{MAX_JOBS_VARIABLE} must be a positive whole number, such as 2, "
                f"not {environ[MAX_JOBS_VARIABLE]!r}."
            )
    return WebappSettings(
        mode=HOSTED_MODE,
        workspace_parent=workspace_parent,
        limits=HostedLimits(max_concurrent_jobs=max_concurrent_jobs),
    )


# ---------------------------------------------------------------------------
# Admission checks
# ---------------------------------------------------------------------------


def format_bytes(size_bytes: int) -> str:
    """Return a size for the user, in MB below 1 GB and in GB from 1 GB, for example ``"50 MB"``.

    One MB is 1024 * 1024 bytes, as in Streamlit's upload limit. The value has
    at most one decimal.
    """
    if size_bytes >= _BYTES_PER_GB:
        value, unit = size_bytes / _BYTES_PER_GB, "GB"
    else:
        value, unit = size_bytes / _BYTES_PER_MB, "MB"
    value_text = f"{value:,.1f}".removesuffix(".0")
    return f"{value_text} {unit}"


def format_minutes(duration_s: float) -> str:
    """Return a duration for the user in whole minutes, for example ``"10 minutes"``."""
    minutes = round(duration_s / 60)
    return f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"


def check_upload(size_bytes: int, raw_bytes: bytes, limits: HostedLimits) -> list[str]:
    """Check an uploaded points file against the hosted limits, before the web app parses it.

    The line count is the number of lines in the raw bytes. ``\\n``,
    ``\\r\\n``, and ``\\r`` each end a line, as in the CSV parser, and a last
    line without a line break counts too. A quoted field with a line break in
    it therefore counts as two lines, so the check is strict near the limit.
    The row limit of :func:`check_hosted_limits` applies after the parse.

    Args:
        size_bytes: The size of the upload, as the upload widget reports it.
        raw_bytes: The content of the upload.
        limits: The hosted limits.

    Returns:
        One message for the user per exceeded limit. An empty list means that
        the file can be parsed.
    """
    messages: list[str] = []
    if size_bytes > limits.max_upload_bytes:
        messages.append(
            f"The file is {format_bytes(size_bytes)}. The limit is "
            f"{format_bytes(limits.max_upload_bytes)}. Remove columns that you do not need, "
            "or split the file into smaller files."
        )

    # bytes.splitlines() ends a line at \n, \r\n, and \r only. A file with old
    # Mac line ends (\r) has no \n, so a count of \n alone would let it through.
    line_count = len(raw_bytes.splitlines())
    if line_count > limits.max_upload_lines:
        messages.append(
            f"The file has {line_count:,} lines. The limit is {limits.max_upload_lines:,} "
            f"lines (one header line and {limits.max_upload_lines - 1:,} data rows). "
            "Split the file into smaller files and run each one separately."
        )
    return messages


def check_hosted_limits(
    points_df: pd.DataFrame,
    selections: Sequence[DatasetSelection],
    limits: HostedLimits,
) -> list[str]:
    """Check a validated form against the hosted limits, before the web app starts a job.

    The checks are: the number of input rows, the number of data-product rows,
    the window size of each row, the tabular request budget, and the raster
    tile budget. The budgets count window sizes, not metres. A row's "Point"
    choice is the window size 0, and it counts as one window size.

    Args:
        points_df: The parsed input points. One row is one point.
        selections: The data-product rows of the form.
        limits: The hosted limits.

    Returns:
        One message for the user per exceeded limit. Each message says what to
        change and states the limit. An empty list means that the job can be
        submitted.
    """
    messages: list[str] = []
    point_count = len(points_df)

    if point_count > limits.max_input_rows:
        messages.append(
            f"The CSV has {point_count:,} rows. The limit is {limits.max_input_rows:,} rows. "
            "Split the file into smaller files and run each one separately."
        )
    if len(selections) > limits.max_dataset_rows:
        messages.append(
            f"The form has {len(selections)} data-product rows. The limit is "
            f"{limits.max_dataset_rows} rows. Remove rows, or run them in separate jobs."
        )

    # Window sizes of each row, and the two budgets.
    tabular_window_count = 0
    raster_window_count = 0
    for row_number, selection in enumerate(selections, start=1):
        if selection.output_type == RASTER_OUTPUT:
            raster_window_count += len(selection.window_sizes)
            max_window_m = limits.max_raster_window_m
        else:
            tabular_window_count += len(selection.window_sizes)
            max_window_m = limits.max_tabular_window_m
        too_large = [size for size in selection.window_sizes if size > max_window_m]
        if too_large:
            sizes_text = ", ".join(f"{size:,} m" for size in too_large)
            subject = "window size" if len(too_large) == 1 else "window sizes"
            verb = "is" if len(too_large) == 1 else "are"
            messages.append(
                f"Data-product row {row_number} ({selection.dataset}): the {subject} "
                f"{sizes_text} {verb} larger than the limit of {max_window_m:,} m for "
                f"{selection.output_type} output. Use smaller window sizes."
            )

    tabular_requests = point_count * tabular_window_count
    if tabular_requests > limits.tabular_request_budget:
        messages.append(
            f"The {TABULAR_OUTPUT} rows need {point_count:,} points x "
            f"{_count_window_sizes(tabular_window_count)} = {tabular_requests:,} requests. "
            f"The limit is {limits.tabular_request_budget:,}. Use fewer points, or fewer "
            f"window sizes in the {TABULAR_OUTPUT} rows (the point value counts as one "
            "window size)."
        )
    raster_tiles = point_count * raster_window_count
    if raster_tiles > limits.raster_tile_budget:
        messages.append(
            f"The {RASTER_OUTPUT} rows need {point_count:,} points x "
            f"{_count_window_sizes(raster_window_count)} = {raster_tiles:,} tiles. "
            f"The limit is {limits.raster_tile_budget:,}. Use fewer points, or fewer "
            f"window sizes in the {RASTER_OUTPUT} rows."
        )
    return messages


def _count_window_sizes(count: int) -> str:
    """Return ``"1 window size"`` or ``"<count> window sizes"``."""
    return "1 window size" if count == 1 else f"{count} window sizes"
