# src/biodata/qc.py
from __future__ import annotations
from dataclasses import dataclass
import logging
import pandas as pd

_DATE_META_KEYS = (
    "date_clamped",
    "date_source",
    "image_time_start",
    "image_time_end",
)

# Substrings identifying QC columns in the final tabular output. QC columns
# are produced by `attach_quality_control` with a "{dataset}_<name>_<suffix>"
# shape, so each keyword is wrapped in underscores to avoid accidental matches
# against stat columns. Keep this list in sync with the columns emitted by
# `compute_qc_flags`, `extract_date_columns`, and `extract_crs_column`.
_QC_COLUMN_KEYWORDS = (
    "_in_extent_",
    "_n_pixels_",
    "_had_nodata_",
    "_coverage_pct_",
    "_image_time_start_",
    "_image_time_end_",
    "_date_clamped_",
    "_date_source_",
    "_region_crs_",
)


@dataclass
class QualityControlBuildResult:
    """Container for QC columns and naming metadata for one dataset.

    Attributes
    ----------
    quality_control_dataframe : pd.DataFrame
        QC columns already prefixed with dataset name and suffixed with the
        mode-specific suffix ("_point" or "_{window}m").
    column_suffix : str
        The suffix applied to all QC columns for this dataset.
    quality_key : str
        Key used when storing the per-dataset coverage summary in metadata:
        "point" for point-only reducers, otherwise the window size as a string.
    """

    quality_control_dataframe: pd.DataFrame
    column_suffix: str
    quality_key: str


def compute_qc_flags(meta_list, min_coverage_pct: int = 80) -> pd.DataFrame:
    """meta_list: list of dicts returned by adapters for each row"""
    df = pd.DataFrame(meta_list)
    low = df["coverage_pct"] < float(min_coverage_pct)
    if low.any():
        logging.warning("Low coverage for %d sample(s) (<%s%%).", int(low.sum()), min_coverage_pct)
    return df[["in_extent", "n_pixels", "had_nodata", "coverage_pct"]]


def build_quality_control_dataframe(
    meta_list: list[dict],
    dataset_name: str,
    reducer_names: list[str],
    window_size_m: int,
    min_coverage_pct: int = 80,
) -> QualityControlBuildResult:
    """Build dataset-prefixed QC columns from adapter metadata.

    This function centralizes QC DataFrame construction so extract orchestration
    code does not need to manage suffixing, date/crs joins, or merge mechanics.
    """
    # "point"-only requests keep the historical "_point" suffix. Any non-point
    # reducer means window-based statistics and therefore a "_{window}m" suffix.
    # quality_key matches the column suffix style ("100m" / "point") so the
    # metadata JSON is consistent across tabular and raster outputs.
    has_window_reducers = any(reducer_name != "point" for reducer_name in reducer_names)
    column_suffix = f"_{window_size_m}m" if has_window_reducers else "_point"
    quality_key = f"{window_size_m}m" if has_window_reducers else "point"

    # Core QC flags (in_extent, n_pixels, had_nodata, coverage_pct) come from
    # adapter meta dicts and are always present for tabular extraction.
    quality_control_dataframe = compute_qc_flags(meta_list, min_coverage_pct=min_coverage_pct)
    quality_control_dataframe = quality_control_dataframe.add_prefix(f"{dataset_name}_").add_suffix(
        column_suffix
    )

    # Date and CRS fields are optional, so we append only when they exist.
    additional_quality_dataframes: list[pd.DataFrame] = []

    date_columns_dataframe = extract_date_columns(meta_list)
    if not date_columns_dataframe.empty:
        additional_quality_dataframes.append(
            date_columns_dataframe.add_prefix(f"{dataset_name}_").add_suffix(column_suffix)
        )

    crs_column_dataframe = extract_crs_column(meta_list)
    if not crs_column_dataframe.empty:
        additional_quality_dataframes.append(
            crs_column_dataframe.add_prefix(f"{dataset_name}_").add_suffix(column_suffix)
        )

    band_coverage_dataframe = extract_band_coverage_columns(meta_list)
    if not band_coverage_dataframe.empty:
        additional_quality_dataframes.append(
            band_coverage_dataframe.add_prefix(f"{dataset_name}_").add_suffix(column_suffix)
        )

    # Reset indexes before concat so row alignment is explicit and robust.
    if additional_quality_dataframes:
        quality_control_dataframe = pd.concat(
            [quality_control_dataframe.reset_index(drop=True)]
            + [d.reset_index(drop=True) for d in additional_quality_dataframes],
            axis=1,
        )

    return QualityControlBuildResult(
        quality_control_dataframe=quality_control_dataframe,
        column_suffix=column_suffix,
        quality_key=quality_key,
    )


def summarize_coverage(
    qc_df: pd.DataFrame,
    coverage_column: str,
    ids: pd.Series | list | None = None,
) -> dict:
    """Count rows by coverage bucket (zero / partial / full) for metadata.

    Reads the per-row coverage_pct column and bins rows into three buckets
    plus a total count. NaN coverage values are treated as zero.

    When ``ids`` is provided, the returned dict also includes ``ids_no_data``
    — the list of input ids whose coverage fell into the zero bucket. This
    lets users see exactly which sample points had no valid pixels for a
    given dataset (e.g. points falling outside the dataset's extent).
    """
    # Missing coverage values (e.g. out-of-extent rows) are treated as 0%
    # so they fall into the n_zero bucket rather than being silently dropped.
    coverage_values = qc_df[coverage_column].fillna(0)

    summary = {
        "n_zero": int((coverage_values == 0).sum()),
        "n_partial": int(((coverage_values > 0) & (coverage_values < 100)).sum()),
        "n_full": int((coverage_values == 100).sum()),
        "total": int(coverage_values.shape[0]),
    }

    # When the caller supplies the id column, attach the list of ids whose
    # coverage was zero so the user can identify which exact points had no
    # data without having to cross-reference the QC csv.
    if ids is not None:
        # Reset indexes so positional alignment between coverage_values and
        # ids is reliable regardless of either input's original index.
        ids_series = pd.Series(list(ids)).reset_index(drop=True)
        zero_mask = coverage_values.reset_index(drop=True) == 0
        summary["ids_no_data"] = ids_series[zero_mask].tolist()

    return summary


def attach_quality_control(
    df: pd.DataFrame,
    *,
    meta_list: list[dict],
    dataset_name: str,
    reducer_names: list[str],
    window_size_m: int,
    min_coverage_pct: int = 80,
) -> tuple[pd.DataFrame, str, dict]:
    """Append QC columns to df and return (df, quality_key, coverage_summary).

    Combines QC DataFrame construction, row-wise concatenation onto the caller's
    DataFrame, and the per-dataset coverage summary used in metadata — so the
    extract orchestrator does not need to manage suffixing, column naming, or
    summary shape.
    """
    # Build the prefixed/suffixed QC DataFrame (core flags + optional date/crs).
    result = build_quality_control_dataframe(
        meta_list=meta_list,
        dataset_name=dataset_name,
        reducer_names=reducer_names,
        window_size_m=window_size_m,
        min_coverage_pct=min_coverage_pct,
    )

    # Reset indexes before concat so row alignment is explicit and robust
    # regardless of whatever index the caller's DataFrame is carrying.
    df_with_qc = pd.concat(
        [df.reset_index(drop=True), result.quality_control_dataframe.reset_index(drop=True)],
        axis=1,
    )

    # Reconstruct the fully-qualified coverage column name here so callers
    # never need to know the "{dataset}_coverage_pct{suffix}" convention.
    coverage_column = f"{dataset_name}_coverage_pct{result.column_suffix}"

    # Pass the input ids (when present) so the coverage summary can include
    # the list of point ids that had no data for this dataset.
    ids = df["id"] if "id" in df.columns else None
    coverage_summary = summarize_coverage(
        result.quality_control_dataframe, coverage_column, ids=ids
    )

    return df_with_qc, result.quality_key, coverage_summary


def split_stats_and_qc(
    df: pd.DataFrame, core_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a tabular output DataFrame into (stats_df, qc_df).

    Core columns (id, lat, lon, date) are preserved on both outputs so that
    each file is independently useful. QC columns are identified by the
    `_QC_COLUMN_KEYWORDS` list above — kept here, next to the code that
    produces them, so adding a new QC meta field only requires one edit.
    """
    # A column is classified as QC if any of the known QC keyword fragments
    # appears in its name (e.g. "dem_local_coverage_pct_200m").
    qc_columns = [c for c in df.columns if any(kw in c for kw in _QC_COLUMN_KEYWORDS)]
    stats_columns = [c for c in df.columns if c not in qc_columns]

    # Both outputs re-emit core columns first so they read naturally.
    # The `c not in core_columns` filter prevents duplicating core columns
    # if they somehow also match a QC keyword.
    stats_df = df[core_columns + [c for c in stats_columns if c not in core_columns]].copy()
    qc_df = df[core_columns + [c for c in qc_columns if c not in core_columns]].copy()
    return stats_df, qc_df


def extract_crs_column(meta_list: list[dict]) -> pd.DataFrame:
    """Extract the per-point region CRS from adapter meta dicts into a DataFrame.

    Returns a single-column DataFrame with column 'region_crs', or empty if
    no CRS info is present. No GEE calls — reads from already-collected meta dicts.
    """
    if not meta_list or "region_crs" not in meta_list[0]:
        return pd.DataFrame()
    return pd.DataFrame({"region_crs": [m.get("region_crs") for m in meta_list]})


def extract_band_coverage_columns(meta_list: list[dict]) -> pd.DataFrame:
    """Extract per-point, per-band coverage from adapter meta dicts into a DataFrame.

    Each row in the result corresponds to one sample point. Columns are named
    ``{band}_coverage_pct`` (e.g. ``bio01_coverage_pct``) so that after the
    caller applies add_prefix/add_suffix the final column names follow the
    ``{dataset}_{band}_coverage_pct_{window}m`` convention — which already
    matches the ``_coverage_pct_`` keyword in _QC_COLUMN_KEYWORDS and is
    therefore automatically routed to the QC output file.

    Returns an empty DataFrame when no meta dict contains band_coverage_pct
    (i.e. single-band datasets, point-only runs, or failure-path metas).
    """
    # Only emit columns when at least the first point has non-empty band data.
    # Failure-path metas carry band_coverage_pct: {} so this correctly skips
    # datasets where every point failed.
    first_band_data = next(
        (m.get("band_coverage_pct") for m in meta_list if m.get("band_coverage_pct")),
        None,
    )
    if not first_band_data:
        return pd.DataFrame()

    rows = [m.get("band_coverage_pct", {}) for m in meta_list]
    band_df = pd.DataFrame(rows)
    # Rename "bio01" → "bio01_coverage_pct" so the column name survives
    # add_prefix/add_suffix intact and still matches _QC_COLUMN_KEYWORDS.
    return band_df.rename(columns={col: f"{col}_coverage_pct" for col in band_df.columns})


def extract_date_columns(meta_list: list[dict]) -> pd.DataFrame:
    """Extract per-point date info from adapter meta dicts into a DataFrame.

    Returns columns: image_time_start, image_time_end, date_clamped,
    date_source. Returns an empty DataFrame if no date info is present in
    the meta dicts. No GEE calls — reads only from the already-collected
    meta dicts.
    """
    if not meta_list or _DATE_META_KEYS[0] not in meta_list[0]:
        return pd.DataFrame()
    rows = [{k: m.get(k) for k in _DATE_META_KEYS} for m in meta_list]
    return pd.DataFrame(rows)
