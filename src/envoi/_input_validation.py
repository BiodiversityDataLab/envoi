# src/envoi/_input_validation.py
"""Input-DataFrame validation for :func:`envoi.extract`.

Six concerns, separately invokable:

* :func:`_validate_required_columns` — fail fast with an actionable message
  when id / latitude / longitude columns are missing under whichever names
  the user supplied.
* :func:`_build_reserved_column_renames` — give temporary names to unmapped
  input columns that have the name of a working column (``id``, ``lat``, ...),
  so they cannot collide with the columns extract() uses internally.
* :func:`_validate_sample_ids` — reject blank or duplicated sample IDs before
  raster mode tries to name one output file per point with them.
* :func:`_parse_and_validate_dates` — read only the listed date forms (GBIF /
  Darwin Core ISO 8601 dates with optional time of day and time zone,
  intervals, ``YYYYMMDD``, ``YYYY/MM/DD``, date and datetime values, and
  whole numbers as years or ``YYYYMMDD``) and collapse each entry to a
  YYYY-MM-DD day. Drops rows
  whose date is null; records per-batch warnings instead of raising. Raises
  for every other date value, such as ``15/06/2021`` or ``June 5, 2021``.
* :func:`_validate_and_reproject_crs` — reject missing, non-numeric, and
  infinite coordinates; reproject to WGS84 when the user declared a non-WGS84
  input CRS; raise if any coordinate then sits outside ±90° / ±180°.
* :func:`_validate_no_output_column_names` — reject input columns that have
  the name of a column that the same extract() call adds to its output
  tables, such as the columns of an earlier envoi run.

Each helper returns the warnings it generated so :func:`extract` can record
them in the metadata sidecar alongside its main run config.
"""

from __future__ import annotations

import datetime
import math
import numbers
import re
import warnings
from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pyproj import Transformer

from ._filenames import format_id_list

if TYPE_CHECKING:
    from ._config_parsing import RunSettings

# Column names that extract() uses in its working table: the user's mapped id,
# latitude, longitude, and date columns are renamed to "id", "lat", "lon", and
# "date", and _validate_and_reproject_crs adds "lat_original" and
# "lon_original". An unmapped input column with one of these names would give
# two columns with the same name, or envoi would read it as its own column
# (for example, an unmapped "date" column would become the sample date).
_WORKING_COLUMN_NAMES = frozenset({"id", "lat", "lon", "date", "lat_original", "lon_original"})

# Number of example IDs or values that an error message lists.
_MAX_IDS_IN_MESSAGE = 10

# A date written YYYY/MM/DD, such as 2021/06/15. It is the same date as
# YYYY-MM-DD. re.ASCII limits \d to the digits 0-9.
_SLASH_DATE_PATTERN = re.compile(r"\d{4}/\d{2}/\d{2}", re.ASCII)

# The start of an ISO 8601 interval ("start/end"): a year, a year-month, or a
# date, optionally with a time of day after "T" or a space, such as 2021,
# 2021-06, 2021-06-15, or 2021-06-15T13:00Z. A value such as 06/2021 has one
# "/" but its start is not a date, so it is not an interval.
_INTERVAL_START_PATTERN = re.compile(r"\d{4}(-\d{2}(-\d{2})?)?([T ].*)?", re.ASCII)

# The text dates that envoi reads. Every other text raises, so that envoi
# never guesses the order of day and month (05-06-2021, 05.06.2021).
# Date: YYYY; YYYY-M or YYYY-MM; YYYY-M-D up to YYYY-MM-DD; or YYYYMMDD.
# Time of day (optional): after "T", in the extended (08:00:00) or the basic
# (080000) form, also the hour alone (T08). After one space, at least hours
# and minutes with ":" (08:00), so that a value such as "2021-06 15" is not
# read as June 2021 at 15:00. Then an optional time zone: Z, +HH, +HH:MM, or
# +HHMM (also with "-").
_DATE_TEXT_PATTERN = re.compile(
    r"""
    (?P<year>\d{4})
    (?:
        (?P<basic_month>\d{2})(?P<basic_day>\d{2})
      | -(?P<month>\d{1,2})(?:-(?P<day>\d{1,2}))?
    )?
    (?P<time>
        (?:
            T\d{2}(?::\d{2}(?::\d{2}(?:[.,]\d+)?)?|\d{2}(?:\d{2}(?:[.,]\d+)?)?)?
          | [ ]\d{2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?
        )
        (?:Z|[+-]\d{2}(?::?\d{2})?)?
    )?
    """,
    re.ASCII | re.VERBOSE,
)


def _validate_required_columns(
    df: pd.DataFrame,
    id_column: str,
    latitude_column: str,
    longitude_column: str,
) -> None:
    """Raise ValueError if df is missing any of the required id/lat/lon columns.

    Uses the user-supplied column names so the error message points at the
    columns the caller is actually expecting to find — not the canonical
    internal names.
    """
    required_columns = {id_column, latitude_column, longitude_column}
    if not required_columns.issubset(df.columns):
        missing_columns = required_columns - set(df.columns)
        raise ValueError(
            f"Input DataFrame is missing required column(s): {sorted(missing_columns)}.\n"
            f"Expected columns: {id_column}, {latitude_column}, {longitude_column} "
            f"(and optionally a date column).\n"
            f"Found columns: {sorted(df.columns.tolist())}"
        )


def _build_reserved_column_renames(
    input_columns: Iterable[Hashable],
    mapped_columns: Iterable[str],
) -> dict[Hashable, str]:
    """Return temporary names for unmapped input columns that have a working-column name.

    ``extract()`` renames the user's id, latitude, longitude, and date columns to
    ``id``, ``lat``, ``lon``, and ``date``, and can add ``lat_original`` and
    ``lon_original`` (see ``_WORKING_COLUMN_NAMES``). An unmapped input column
    with one of these names, such as the ``id`` column of a Darwin Core file,
    must get a different name before that rename. ``extract()`` adds the returned
    mapping to its column rename, so ``_restore_user_column_names`` gives the
    column its original name again in the output tables.

    Args:
        input_columns: Column names of the input dataframe.
        mapped_columns: The user's id, latitude, longitude, and date column
            names. These columns become working columns, so they are not moved.

    Returns:
        A mapping from each colliding input column name to a temporary name
        that is not an input column name and not a working column name.
        The mapping is empty when no input column collides.
    """
    mapped_column_set = set(mapped_columns)
    taken_names = set(input_columns) | _WORKING_COLUMN_NAMES
    temporary_names: dict[Hashable, str] = {}
    for column in input_columns:
        if column not in _WORKING_COLUMN_NAMES or column in mapped_column_set:
            continue
        # Add a number when the input already has a column with the temporary name.
        temporary_name = f"_envoi_input_{column}"
        number = 1
        while temporary_name in taken_names:
            number += 1
            temporary_name = f"_envoi_input_{column}_{number}"
        taken_names.add(temporary_name)
        temporary_names[column] = temporary_name
    return temporary_names


def _validate_sample_ids(
    id_values: pd.Series,
    id_column_name: str,
    *,
    context: str = "",
) -> None:
    """Raise ValueError if sample IDs cannot safely name one file per point.

    Raster mode writes one GeoTIFF per point, named after that point's ID. A
    blank ID has no name to use, and two points sharing an ID would have their
    tiles silently overwrite one another — the user would get fewer tiles than
    points with nothing to indicate why. Both are real data errors, so we fail
    loudly and early (before any adapter is built or any Earth Engine quota is
    spent) rather than producing a quietly incomplete result.

    Args:
        id_values: The ID column of the input DataFrame.
        id_column_name: The user-facing column name, used in the error message
            so it points at the column the caller actually has — matching the
            convention used by :func:`_parse_and_validate_dates`.
        context: Optional prefix identifying which output config triggered the
            check, e.g. ``"Output 'terrain'"``.
    """
    # Prefix every message with the batch context when we have one, so a
    # multi-output run says which config the bad IDs belong to.
    message_prefix = f"{context}: " if context else ""

    # Treat NaN/None and whitespace-only strings alike — neither can name a
    # file. Coerce to str first so numeric ID columns compare cleanly.
    normalized_ids = id_values.astype("string").str.strip()
    blank_mask = normalized_ids.isna() | (normalized_ids == "")
    if blank_mask.any():
        # Report 1-based row positions, matching how a user reads their CSV.
        blank_row_numbers = [int(position) + 1 for position in blank_mask.to_numpy().nonzero()[0]]
        raise ValueError(
            f"{message_prefix}column '{id_column_name}' has missing or blank values at "
            f"row(s) {format_id_list(blank_row_numbers)}. Raster output names one GeoTIFF "
            f"per point using this column, so every row needs an identifier."
        )

    duplicate_mask = normalized_ids.duplicated(keep=False)
    if duplicate_mask.any():
        duplicated_values = sorted(set(normalized_ids[duplicate_mask].tolist()))
        raise ValueError(
            f"{message_prefix}column '{id_column_name}' has duplicate values: "
            f"{format_id_list(duplicated_values)}. Raster output names one GeoTIFF per "
            f"point using this column, so duplicated IDs would overwrite each other's "
            f"tiles. Give every point a unique identifier."
        )


def _parse_and_validate_dates(
    df: pd.DataFrame,
    date_column_name: str = "eventDate",
) -> tuple[pd.DataFrame, list | None, list[str]]:
    """Parse and validate the 'date' column from the input DataFrame.

    Returns a tuple of (df, dates, date_warnings) where df has rows with missing
    dates removed, dates is a list of YYYY-MM-DD strings (or None if no 'date'
    column is present), and date_warnings is a list of messages the caller can
    print and record in the output metadata.

    ``date_column_name`` is the user-facing column name used purely for
    diagnostic messages — by the time this function runs the column has
    already been renamed to the canonical ``"date"``, so look-ups inside
    use the canonical name regardless.

    Accepted values (every other value raises, so envoi never guesses the
    order of day and month):

    - Text ``YYYY``, ``YYYY-M`` or ``YYYY-MM``, ``YYYY-M-D`` up to
      ``YYYY-MM-DD``, and ``YYYYMMDD``, each optionally with a time of day
      after ``T`` or one space and a time zone (see ``_DATE_TEXT_PATTERN``).
      A year or a year-month is read as its first day, with a warning. The
      time of day and the time zone are dropped, with a warning: the day is
      the day as written.
    - An ISO 8601 interval: a value with exactly one ``/`` whose start is
      ``YYYY``, ``YYYY-MM``, or ``YYYY-MM-DD`` with a two-digit month and day
      (``_INTERVAL_START_PATTERN``), optionally with a time of day
      (``2021-06-15/2021-06-20``). Its start is read as above, with a warning.
    - Text ``YYYY/MM/DD``, the date ``YYYY-MM-DD``.
    - Date and datetime values (also in an object column), and the values
      of a column of a datetime type. The day is the day in the value's own
      time zone.
    - A whole number with four digits, read as a year (``2021`` or
      ``2021.0``), or with eight digits, read as ``YYYYMMDD`` (``20210615``
      or ``20210615.0``): a CSV reader stores such columns as numbers.

    Raises:
        ValueError: If a date value is not in an accepted form, or if it is
            not a valid date (such as ``2021-02-30``). The message names the
            user's date column and lists up to 10 different values.
    """
    date_warnings: list[str] = []

    if "date" not in df.columns:
        # A missing date column is a normal, expected case (static datasets, or
        # users who simply don't have dates), so this is informational rather
        # than a Python warning — a raised UserWarning here is too aggressive
        # and clutters every date-less run. We still record it in date_warnings
        # so it lands in the metadata sidecar for auditing.
        message = (
            f"No '{date_column_name}' column found in input DataFrame; proceeding without dates."
        )
        print(message)
        date_warnings.append(message)
        return df, None, date_warnings

    # Drop rows with missing dates rather than raising, so the user still gets
    # results for the valid rows. The dropped ids are recorded in the warnings
    # so the user can see exactly which points were skipped.
    null_date_mask = df["date"].isna()
    if null_date_mask.any():
        null_ids = df.loc[null_date_mask, "id"].tolist()
        message = (
            f"Skipping {len(null_ids)} row(s) with missing dates "
            f"(ids: {null_ids}). Provide a date for every row to include them."
        )
        warnings.warn(message, stacklevel=2)
        date_warnings.append(message)
        df = df.loc[~null_date_mask].copy()

    raw_dates = df["date"].tolist()

    # Read each value as one day, "YYYY-MM-DD". GBIF / Darwin Core
    # ``eventDate`` follows ISO 8601 and can be an interval like
    # ``"2026-05-12T13:00/2026-05-12T15:45"`` (start/end), a date with a time
    # of day, and a time zone (``Z``, ``+02:00``). Nearest-image lookup
    # downstream only uses day precision, so keep the start of an interval and
    # drop the time of day. Rows with a missing date were dropped above.
    days: list[str] = []
    # The date as written of each year or year-month value, which the
    # incomplete-date warning quotes, or None for a full date. Aligned with days.
    incomplete_date_texts: list[str | None] = []
    # The values (as text) that are not in an accepted form, for one error.
    unreadable_values: list[str] = []
    # True when one of them is a number, for a hint in the error.
    has_unreadable_number = False
    interval_truncation_count = 0
    time_truncation_count = 0
    for raw_date in raw_dates:
        # ---- date and datetime values ----
        # A datetime column gives pandas Timestamps, which are datetime values.
        # Use the day as it is, in the value's own time zone, as for text with
        # a time zone. An object column can also hold NumPy datetime64 values,
        # which have no time zone.
        if isinstance(raw_date, np.datetime64):
            days.append(str(np.datetime_as_string(raw_date, unit="D")))
            incomplete_date_texts.append(None)
            continue
        if isinstance(raw_date, datetime.date):
            days.append(f"{raw_date.year:04d}-{raw_date.month:02d}-{raw_date.day:02d}")
            incomplete_date_texts.append(None)
            continue

        # ---- numbers: a year or YYYYMMDD ----
        # A CSV reader stores a column of years (2021) or of YYYYMMDD dates
        # (20210615) as numbers, and as floats (2021.0) when a value is
        # missing. A whole number with four digits is read as the text YYYY,
        # one with eight digits as the text YYYYMMDD, below. Every other
        # number raises. bool is a subclass of int, but True is not a date.
        if isinstance(raw_date, numbers.Real) and not isinstance(raw_date, bool):
            is_whole_number = math.isfinite(raw_date) and float(raw_date).is_integer()
            digit_text = str(int(raw_date)) if is_whole_number else ""
            if len(digit_text) not in (4, 8):
                unreadable_values.append(str(raw_date))
                has_unreadable_number = True
                continue
            raw_date = digit_text

        # ---- text ----
        if not isinstance(raw_date, str):
            unreadable_values.append(str(raw_date))
            continue
        raw_date_str = raw_date.strip()
        date_text = raw_date_str
        slash_count = date_text.count("/")
        interval_start = date_text.split("/", 1)[0].strip()
        if slash_count == 1 and _INTERVAL_START_PATTERN.fullmatch(interval_start):
            # An ISO 8601 interval: keep its start.
            interval_truncation_count += 1
            date_text = interval_start
        elif _SLASH_DATE_PATTERN.fullmatch(date_text):
            # A date written YYYY/MM/DD. Cutting it at the first "/" would
            # keep only the year, which would be read as January 1.
            date_text = date_text.replace("/", "-")

        # Any other form, such as 15/06/2021, 06/2021, 05-06-2021, 05.06.2021,
        # or "June 5, 2021", does not match. The order of day and month is
        # unknown there, so do not guess. Collect the values for one error.
        date_match = _DATE_TEXT_PATTERN.fullmatch(date_text)
        if date_match is None:
            unreadable_values.append(raw_date_str)
            continue

        month = date_match.group("month") or date_match.group("basic_month")
        day = date_match.group("day") or date_match.group("basic_day")
        days.append(f"{date_match.group('year')}-{int(month or 1):02d}-{int(day or 1):02d}")
        # Drop the time of day and the time zone: the day is the day as written.
        time_start = date_match.start("time")
        if time_start != -1:
            time_truncation_count += 1
            date_text = date_text[:time_start]
        incomplete_date_texts.append(None if day else date_text)

    if unreadable_values:
        # List each different value once, so a column with many copies of the
        # same value gives a short message.
        different_values = list(dict.fromkeys(unreadable_values))
        number_hint = (
            " A number is read only when it is a whole number with four digits (a year, "
            "such as 2021) or eight digits (YYYYMMDD, such as 20210615)."
            if has_unreadable_number
            else ""
        )
        raise ValueError(
            f"Column '{date_column_name}' has {len(unreadable_values)} date value(s) that "
            f"envoi cannot read: {format_id_list(different_values, limit=_MAX_IDS_IN_MESSAGE)}. "
            f"envoi reads text dates in the forms YYYY, YYYY-MM, YYYY-MM-DD (the month and "
            f"the day can have one digit, as in 2021-6-5), YYYYMMDD, and YYYY/MM/DD. Each "
            f"form except YYYY/MM/DD can have a time of day after 'T' or a space, and a time "
            f"zone (for example 2021-06-15T08:00Z). envoi also reads ISO 8601 intervals "
            f"'start/end' (for example 2021-06-15/2021-06-20), of which it uses the start, "
            f"date and datetime values, and whole numbers as years (2021) or as YYYYMMDD "
            f"(20210615). It does not guess the order of day and month in other forms, such "
            f"as 15/06/2021, 05-06-2021, or 05.06.2021. Write these dates as "
            f"YYYY-MM-DD.{number_hint}"
        )

    # Every day is now "YYYY-MM-DD", so pandas reads it with this one format
    # and guesses nothing. A day that does not exist (2021-02-30, 2021-13-01),
    # or that pandas cannot store, becomes NaT. Collect those for one error.
    parsed_dates = pd.to_datetime(days, format="%Y-%m-%d", errors="coerce")
    invalid_date_values = [
        str(raw_date).strip()
        for raw_date, parsed_date in zip(raw_dates, parsed_dates)
        if pd.isna(parsed_date)
    ]
    if invalid_date_values:
        different_values = list(dict.fromkeys(invalid_date_values))
        raise ValueError(
            f"Column '{date_column_name}' has {len(invalid_date_values)} date value(s) that "
            f"are not valid dates: "
            f"{format_id_list(different_values, limit=_MAX_IDS_IN_MESSAGE)}. Check that each "
            f"month is 1 to 12, that each day exists in its month (2021-02-30 does not), "
            f"and that each year is correct."
        )

    # One aggregated warning per kind of truncation, not one per row — GBIF
    # downloads routinely carry these formats and a per-row warning floods the
    # console for no extra information.
    if interval_truncation_count:
        message = (
            f"Truncated {interval_truncation_count} ISO 8601 date interval(s) "
            f"('start/end') to their start day; nearest-image lookup only uses "
            f"day precision."
        )
        warnings.warn(message, stacklevel=2)
        date_warnings.append(message)
    if time_truncation_count:
        message = (
            f"Dropped time-of-day from {time_truncation_count} date value(s); "
            f"nearest-image lookup only uses day precision."
        )
        warnings.warn(message, stacklevel=2)
        date_warnings.append(message)

    # A year ("2002") or a year-month ("2002-02") is read as its first day.
    for incomplete_date_text, day_text in zip(incomplete_date_texts, days):
        if incomplete_date_text is not None:
            message = (
                f"Date '{incomplete_date_text}' interpreted as {day_text}. "
                f"Provide a full YYYY-MM-DD date if you want a specific day."
            )
            warnings.warn(message, stacklevel=2)
            date_warnings.append(message)

    return df, days, date_warnings


def _validate_and_reproject_crs(
    df: pd.DataFrame,
    input_crs: str | None,
    *,
    id_column: str,
    latitude_column: str,
    longitude_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Check the coordinates, reproject them to WGS84 if needed, and check their range.

    Expects the canonical ``id``, ``lat``, and ``lon`` columns. Changes ``df`` in
    place: ``lat`` and ``lon`` become numeric (text that holds a number, such as
    ``"62.5"``, is converted), and with a non-WGS84 ``input_crs`` they hold the
    WGS84 values and ``lat_original`` / ``lon_original`` keep the input values.

    Args:
        df: The working dataframe, with the canonical column names.
        input_crs: The CRS of the input coordinates, or None for WGS84.
        id_column: The user's ID column name, used in error messages.
        latitude_column: The user's latitude column name, used in error messages.
        longitude_column: The user's longitude column name, used in error messages.

    Returns:
        ``(df, crs_warnings)``, where ``crs_warnings`` lists messages about CRS
        handling (e.g. a reprojection notice) that the caller records in the
        metadata sidecar.

    Raises:
        ValueError: If a latitude or longitude is missing, not a number, or not
            finite, or if a WGS84 coordinate is outside ±90° / ±180°.
    """
    crs_warnings: list[str] = []

    # Every coordinate must be a finite number before the range check and the
    # reprojection. Text that is not a number becomes NaN here, so one check
    # finds missing, non-numeric, and infinite values. Without this check, text
    # gives an unclear TypeError, infinity gives a misleading "not in WGS84"
    # error, and NaN fails later inside an adapter or gives empty statistics.
    latitudes = pd.to_numeric(df["lat"], errors="coerce")
    longitudes = pd.to_numeric(df["lon"], errors="coerce")
    invalid_latitude_mask = ~np.isfinite(latitudes.astype("float64"))
    invalid_longitude_mask = ~np.isfinite(longitudes.astype("float64"))
    invalid_row_mask = invalid_latitude_mask | invalid_longitude_mask
    if invalid_row_mask.any():
        invalid_columns = []
        if invalid_latitude_mask.any():
            invalid_columns.append(f"'{latitude_column}'")
        if invalid_longitude_mask.any():
            invalid_columns.append(f"'{longitude_column}'")
        invalid_ids = df.loc[invalid_row_mask, "id"].tolist()
        raise ValueError(
            f"Column(s) {' and '.join(invalid_columns)} have a missing, non-numeric, or "
            f"infinite value in {len(invalid_ids)} row(s). '{id_column}' of these rows: "
            f"{format_id_list(invalid_ids, limit=_MAX_IDS_IN_MESSAGE)}. Every row needs a "
            f"finite number in '{latitude_column}' and '{longitude_column}'. Correct or "
            f"remove these rows."
        )
    df["lat"] = latitudes
    df["lon"] = longitudes

    if input_crs is not None:
        input_crs_upper = input_crs.upper()
        if input_crs_upper != "EPSG:4326" and input_crs_upper != "WGS84":
            message = f"Reprojecting coordinates from {input_crs} to EPSG:4326 (WGS84)."
            warnings.warn(message, stacklevel=2)
            crs_warnings.append(message)
            # Stash the caller's original coordinates before overwriting the
            # canonical lat/lon with their WGS84 reprojection. The rest of the
            # pipeline (the adapters) needs WGS84, but the output table should
            # still surface the coordinates the user actually supplied — these
            # helper columns ride along in the DataFrame and are read back when
            # the tabular output is assembled.
            df["lat_original"] = df["lat"]
            df["lon_original"] = df["lon"]
            transformer = Transformer.from_crs(input_crs, "EPSG:4326", always_xy=True)
            lons, lats = transformer.transform(df["lon"].values, df["lat"].values)
            df["lon"] = lons
            df["lat"] = lats

    bad_lat_mask = df["lat"].abs() > 90
    bad_lon_mask = df["lon"].abs() > 180
    bad_mask = bad_lat_mask | bad_lon_mask
    if bad_mask.any():
        bad_rows = df.loc[bad_mask, ["id", "lat", "lon"]]
        problems = []
        if bad_lat_mask.any():
            problems.append("latitude values outside ±90°")
        if bad_lon_mask.any():
            problems.append("longitude values outside ±180°")
        raise ValueError(
            f"Coordinates appear to not be in WGS84 (EPSG:4326): {'; '.join(problems)}.\n"
            f"Rows with invalid coordinates:\n{bad_rows.to_string(index=False)}\n"
            f"If your coordinates are in a different CRS, pass input_crs='EPSG:XXXX'"
        )

    return df, crs_warnings


def _validate_no_output_column_names(
    input_columns: Iterable[Hashable],
    all_run_settings: Sequence[RunSettings],
    catalog_dataset_names: Iterable[str],
    *,
    latitude_column: str,
    longitude_column: str,
    coordinates_reprojected: bool,
) -> None:
    """Raise ValueError if an input column has the name of an output column of this call.

    A tabular run config adds statistic and QC columns named after each
    dataset and window to the input table, and, when the coordinates are
    reprojected, ``<latitude column>_wgs84`` and ``<longitude column>_wgs84``.
    An input column with such a name, for example from the output of an
    earlier envoi run, would be overwritten without notice, or would stop the
    run after the extraction. Raster run configs add no columns.

    For a dataset ``d`` of a tabular run config with a window of ``N`` m, an
    input column matches when it is ``d_point``, or when it starts with
    ``d_`` and its remaining part ends with ``_Nm`` or ``_point``. A column
    that starts with ``e_`` for another catalog dataset ``e`` whose name
    starts with ``d_`` (``dem_copernicus_glo30`` for ``dem``) belongs to
    ``e``, so it does not match for ``d``.

    Args:
        input_columns: The column names of the input dataframe as the user
            gave them, before any rename.
        all_run_settings: Every parsed run config of the call.
        catalog_dataset_names: The names of every dataset in the merged catalog.
        latitude_column: The user's latitude column name.
        longitude_column: The user's longitude column name.
        coordinates_reprojected: True when envoi reprojects the coordinates
            from ``input_crs`` to WGS84 (``_validate_and_reproject_crs`` then
            adds ``lat_original``), so a tabular output gets the ``_wgs84``
            columns.

    Raises:
        ValueError: If one or more input columns have an output column name.
            The message names the columns and the outputs.
    """
    input_column_names = [column for column in input_columns if isinstance(column, str)]
    catalog_dataset_names = list(catalog_dataset_names)
    # Each matching input column, with the batch_id of the first output that adds it.
    colliding_columns: dict[str, str] = {}

    for run_settings in all_run_settings:
        if run_settings.output_type != "tabular":
            continue
        window_suffixes = tuple(f"_{window_size}m" for window_size in run_settings.window_sizes)

        for dataset, _ in run_settings.datasets:
            dataset_prefix = f"{dataset}_"
            # Prefixes of the longer catalog datasets whose names start with
            # this dataset's name, such as "dem_copernicus_glo30_" for "dem".
            longer_dataset_prefixes = tuple(
                f"{other_dataset}_"
                for other_dataset in catalog_dataset_names
                if len(other_dataset) > len(dataset) and other_dataset.startswith(dataset_prefix)
            )
            for column in input_column_names:
                if not column.startswith(dataset_prefix):
                    continue
                if longer_dataset_prefixes and column.startswith(longer_dataset_prefixes):
                    continue
                # The part after "<dataset>_", so that the "_" of the window
                # or point suffix cannot be the "_" after the dataset name.
                column_end = column[len(dataset_prefix) :]
                is_window_column = column_end.endswith(window_suffixes)
                is_point_column = column_end == "point" or column_end.endswith("_point")
                if is_window_column or is_point_column:
                    colliding_columns.setdefault(column, run_settings.batch_id)

        if coordinates_reprojected:
            for coordinate_column in (latitude_column, longitude_column):
                wgs84_column = f"{coordinate_column}_wgs84"
                if wgs84_column in input_column_names:
                    colliding_columns.setdefault(wgs84_column, run_settings.batch_id)

    if not colliding_columns:
        return
    # Keep the input order of the columns in the message, and list each once.
    column_list = [
        column for column in dict.fromkeys(input_column_names) if column in colliding_columns
    ]
    batch_id_list = list(dict.fromkeys(colliding_columns[column] for column in column_list))
    quoted_batch_ids = ", ".join(f"'{batch_id}'" for batch_id in batch_id_list)
    raise ValueError(
        f"Input column(s) {format_id_list(column_list, limit=_MAX_IDS_IN_MESSAGE)} have the "
        f"name of a column that envoi adds to the output tables of output(s) "
        f"{quoted_batch_ids}, so envoi would overwrite them. This happens, for example, when "
        f"the input table is the output table of an earlier envoi run. Remove or rename "
        f"these columns before you call extract()."
    )
