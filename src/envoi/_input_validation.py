# src/envoi/_input_validation.py
"""Input-DataFrame validation for :func:`envoi.extract`.

Three concerns, separately invokable:

* :func:`_validate_required_columns` — fail fast with an actionable message
  when id / latitude / longitude columns are missing under whichever names
  the user supplied.
* :func:`_parse_and_validate_dates` — accept GBIF / Darwin Core ISO 8601
  flexibility (intervals, time-of-day, timezone suffixes, year-only, year-month)
  and collapse each entry to a YYYY-MM-DD day. Drops rows whose date is null;
  records per-batch warnings instead of raising.
* :func:`_validate_and_reproject_crs` — reproject to WGS84 when the user
  declared a non-WGS84 input CRS; raise if any coordinate then sits outside
  ±90° / ±180°.

Each helper returns the warnings it generated so :func:`extract` can record
them in the metadata sidecar alongside its main run config.
"""

from __future__ import annotations

import warnings

import pandas as pd
from pyproj import Transformer


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
    """
    date_warnings: list[str] = []

    if "date" not in df.columns:
        message = (
            f"No '{date_column_name}' column found in input DataFrame; "
            f"proceeding without dates."
        )
        warnings.warn(message, stacklevel=2)
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

    # GBIF / Darwin Core ``eventDate`` follows ISO 8601 and is allowed to be
    # an interval like ``"2026-05-12T13:00/2026-05-12T15:45"`` (start/end), a
    # datetime with a time-of-day component, and may carry a timezone suffix
    # (``Z``, ``+02:00``). Nearest-image lookup downstream only uses day
    # precision, so collapse each entry to its date portion *before* handing
    # it to pandas: take the start of any interval, drop the ``T...`` time
    # component. This also avoids pandas's "mixed time zones" parsing path,
    # which returns an object-dtype Index rather than a DatetimeIndex.
    preprocessed_dates: list[str] = []
    interval_truncation_count = 0
    time_truncation_count = 0
    for raw_date in raw_dates:
        raw_date_str = str(raw_date).strip()
        # Split on the first "/" to keep the start half of an interval.
        if "/" in raw_date_str:
            interval_truncation_count += 1
            raw_date_str = raw_date_str.split("/", 1)[0]
        # Then split on "T" to drop any time-of-day and timezone suffix.
        if "T" in raw_date_str:
            time_truncation_count += 1
            raw_date_str = raw_date_str.split("T", 1)[0]
        preprocessed_dates.append(raw_date_str)

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

    # format="mixed" lets pandas infer the format per-element, which is needed
    # to accept a mix of full ("2021-06-15"), year-month ("2021-06"), and
    # year-only ("2021") dates in the same column. Without this, pandas locks
    # onto the first element's format and raises on any entry that doesn't match.
    try:
        parsed_dates = pd.to_datetime(preprocessed_dates, format="mixed")
    except Exception as e:
        raise ValueError(
            f"Error parsing '{date_column_name}' column: {e}. "
            f"Expected dates in YYYY-MM-DD format."
        )

    # Detect incomplete dates (year-only "2002" or year-month "2002-02") by
    # splitting on "-". A complete date has 3 parts (YYYY-MM-DD); year-only
    # has 1, year-month has 2. This correctly handles single-digit months/days
    # like "2002-1-1" (still 3 parts → complete).
    for preprocessed_date, parsed_date in zip(preprocessed_dates, parsed_dates):
        if len(preprocessed_date.split("-")) < 3:
            message = (
                f"Date '{preprocessed_date}' interpreted as {parsed_date.strftime('%Y-%m-%d')}. "
                f"Provide a full YYYY-MM-DD date if you want a specific day."
            )
            warnings.warn(message, stacklevel=2)
            date_warnings.append(message)

    return df, parsed_dates.strftime("%Y-%m-%d").tolist(), date_warnings


def _validate_and_reproject_crs(
    df: pd.DataFrame, input_crs: str | None
) -> tuple[pd.DataFrame, list[str]]:
    """Reproject coordinates to WGS84 if needed, and raise if any lat/lon are out of range.

    Returns a tuple of (df, crs_warnings) where crs_warnings is a list of messages
    about CRS handling (e.g. a reprojection notice) that the caller can record in
    the output metadata alongside the warnings_backlog.
    """
    crs_warnings: list[str] = []

    if input_crs is not None:
        input_crs_upper = input_crs.upper()
        if input_crs_upper != "EPSG:4326" and input_crs_upper != "WGS84":
            message = f"Reprojecting coordinates from {input_crs} to EPSG:4326 (WGS84)."
            warnings.warn(message, stacklevel=2)
            crs_warnings.append(message)
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
