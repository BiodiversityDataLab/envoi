# src/biodata/qc.py
from __future__ import annotations
from dataclasses import dataclass
import logging
import pandas as pd

# Date fields written into adapter meta dicts (GEE ImageCollections only).
# Used exclusively by extract_date_columns below.
_DATE_META_KEYS = (
    "date_clamped",
    "date_source",
    "image_time_start",
    "image_time_end",
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
    qc_column_names : list[str]
        The exact column names present in quality_control_dataframe. Callers
        use this to split stats vs. QC columns without relying on substring
        matching against column names.
    """

    quality_control_dataframe: pd.DataFrame
    column_suffix: str
    quality_key: str
    qc_column_names: list[str]


# ---------------------------------------------------------------------------
# Atomic helpers — each extracts one category of fields from adapter meta dicts.
# These are called by build_quality_control_dataframe below.
# ---------------------------------------------------------------------------


def compute_qc_flags(meta_list: list[dict], min_coverage_pct: int | float) -> pd.DataFrame:
    """Build core QC flags from adapter metadata dicts.

    Extracts in_extent, n_pixels, had_nodata, and coverage_pct from each
    meta dict, warns when any point falls below min_coverage_pct, and
    returns a DataFrame with one row per input point.
    """
    if not meta_list:
        return pd.DataFrame(columns=["in_extent", "n_pixels", "had_nodata", "coverage_pct"])
    df = pd.DataFrame(meta_list)
    below_threshold_mask = df["coverage_pct"] < min_coverage_pct
    if below_threshold_mask.any():
        logging.warning(
            "Low coverage for %d sample(s) (<%s%%).",
            int(below_threshold_mask.sum()),
            min_coverage_pct,
        )
    return df[["in_extent", "n_pixels", "had_nodata", "coverage_pct"]]


def extract_date_columns(meta_list: list[dict]) -> pd.DataFrame:
    """Extract per-point date info from adapter meta dicts into a DataFrame.

    Returns columns: image_time_start, image_time_end, date_clamped,
    date_source. Returns an empty DataFrame if no date info is present in
    the meta dicts. No GEE calls — reads only from the already-collected
    meta dicts.
    """
    # Check all dicts, not just meta_list[0], because the first point may be a
    # failure-path meta dict that doesn't carry date keys.
    if not meta_list or not any(_DATE_META_KEYS[0] in m for m in meta_list):
        return pd.DataFrame()
    date_field_rows = [{k: m.get(k) for k in _DATE_META_KEYS} for m in meta_list]
    return pd.DataFrame(date_field_rows)


def extract_crs_column(meta_list: list[dict]) -> pd.DataFrame:
    """Extract the per-point region CRS from adapter meta dicts into a DataFrame.

    Returns a single-column DataFrame with column 'region_crs', or empty if
    no CRS info is present. No GEE calls — reads from already-collected meta dicts.
    """
    # Check all dicts, not just meta_list[0], because the first point may be a
    # failure-path meta dict that doesn't carry region_crs.
    if not meta_list or not any("region_crs" in m for m in meta_list):
        return pd.DataFrame()
    return pd.DataFrame({"region_crs": [m.get("region_crs") for m in meta_list]})


def extract_band_coverage_columns(meta_list: list[dict]) -> pd.DataFrame:
    """Extract per-point, per-band coverage from adapter meta dicts into a DataFrame.

    Each row in the result corresponds to one sample point. Columns are named
    ``{band}_coverage_pct`` (e.g. ``bio01_coverage_pct``) so that after the
    caller applies add_prefix/add_suffix the final column names follow the
    ``{dataset}_{band}_coverage_pct_{window}m`` convention and are included
    in the qc_column_names list returned by build_quality_control_dataframe.

    Returns an empty DataFrame when no meta dict contains band_coverage_pct
    (i.e. single-band datasets, point-only runs, or failure-path metas).
    """
    # Only emit columns when at least one point across the batch has non-empty
    # band data. Failure-path metas carry band_coverage_pct: {} so this correctly
    # skips datasets where every point failed.
    first_band_data = next(
        (m.get("band_coverage_pct") for m in meta_list if m.get("band_coverage_pct")),
        None,
    )
    if not first_band_data:
        return pd.DataFrame()

    band_coverage_rows = [m.get("band_coverage_pct", {}) for m in meta_list]
    band_coverage_dataframe = pd.DataFrame(band_coverage_rows)

    # Cheap uniformity check: when all bands share the same pixel grid
    # (e.g. a multi-band composite image like WorldClim) every band will
    # have identical coverage at every point. The existing coverage_pct
    # column already captures that value, so emitting 19 duplicate columns
    # adds no information — skip the per-band breakdown in that case.
    first_band_column = band_coverage_dataframe.iloc[:, 0]
    if all(
        band_coverage_dataframe[band_name].equals(first_band_column)
        for band_name in band_coverage_dataframe.columns[1:]
    ):
        logging.debug(
            "Skipping per-band coverage breakdown: all %d bands have identical coverage values.",
            len(band_coverage_dataframe.columns),
        )
        return pd.DataFrame()

    # Rename "bio01" → "bio01_coverage_pct" so the column name is self-describing
    # after add_prefix/add_suffix assembles the full "{dataset}_{band}_coverage_pct_{window}m" form.
    return band_coverage_dataframe.rename(
        columns={
            band_name: f"{band_name}_coverage_pct" for band_name in band_coverage_dataframe.columns
        }
    )


# ---------------------------------------------------------------------------
# Assemblers — combine the atomic helpers and attach QC data to the main DataFrame.
# ---------------------------------------------------------------------------


def build_quality_control_dataframe(
    meta_list: list[dict],
    dataset_name: str,
    reducer_names: list[str],
    window_size_m: int,
    min_coverage_pct: int | float,
) -> QualityControlBuildResult:
    """Build dataset-prefixed QC columns from adapter metadata.

    Produces a prefixed/suffixed DataFrame covering core flags (in_extent,
    n_pixels, had_nodata, coverage_pct) plus optional date, CRS, and per-band
    coverage columns when present in the meta dicts. Centralizes QC DataFrame
    construction so extract orchestration code does not need to manage
    suffixing, date/crs joins, or merge mechanics.
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

    # Date, CRS, and per-band coverage fields are optional, so we append only when they exist.
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
            + [
                additional_df.reset_index(drop=True)
                for additional_df in additional_quality_dataframes
            ],
            axis=1,
        )

    return QualityControlBuildResult(
        quality_control_dataframe=quality_control_dataframe,
        column_suffix=column_suffix,
        quality_key=quality_key,
        qc_column_names=quality_control_dataframe.columns.tolist(),
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

    coverage_summary = {
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
        zero_coverage_mask = coverage_values.reset_index(drop=True) == 0
        coverage_summary["ids_no_data"] = ids_series[zero_coverage_mask].tolist()

    return coverage_summary


def attach_quality_control(
    df: pd.DataFrame,
    *,
    meta_list: list[dict],
    dataset_name: str,
    reducer_names: list[str],
    window_size_m: int,
    min_coverage_pct: int | float,
) -> tuple[pd.DataFrame, str, dict, list[str]]:
    """Append QC columns to df and return (df, quality_key, coverage_summary, qc_column_names).

    Combines QC DataFrame construction, row-wise concatenation onto the caller's
    DataFrame, and the per-dataset coverage summary used in metadata — so the
    extract orchestrator does not need to manage suffixing, column naming, or
    summary shape. The returned qc_column_names list lets the caller accumulate
    exact QC column names across datasets for use with split_stats_and_qc.
    """
    # Build the prefixed/suffixed QC DataFrame (core flags + optional date/crs/band_coverage).
    qc_build_result = build_quality_control_dataframe(
        meta_list=meta_list,
        dataset_name=dataset_name,
        reducer_names=reducer_names,
        window_size_m=window_size_m,
        min_coverage_pct=min_coverage_pct,
    )

    # Reset indexes before concat so row alignment is explicit and robust
    # regardless of whatever index the caller's DataFrame is carrying.
    df_with_qc = pd.concat(
        [
            df.reset_index(drop=True),
            qc_build_result.quality_control_dataframe.reset_index(drop=True),
        ],
        axis=1,
    )

    # Reconstruct the fully-qualified coverage column name here so callers
    # never need to know the "{dataset}_coverage_pct{suffix}" convention.
    coverage_column = f"{dataset_name}_coverage_pct{qc_build_result.column_suffix}"

    # Pass the input ids (when present) so the coverage summary can include
    # the list of point ids that had no data for this dataset.
    ids = df["id"] if "id" in df.columns else None
    coverage_summary = summarize_coverage(
        qc_build_result.quality_control_dataframe, coverage_column, ids=ids
    )

    return (
        df_with_qc,
        qc_build_result.quality_key,
        coverage_summary,
        qc_build_result.qc_column_names,
    )


def split_stats_and_qc(
    df: pd.DataFrame,
    core_columns: list[str],
    qc_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a tabular output DataFrame into (stats_df, qc_df).

    Core columns (id, lat, lon, date) are preserved on both outputs so that
    each file is independently useful. QC columns are identified by the
    explicit ``qc_columns`` list, which is built up from the column names
    returned by ``attach_quality_control`` for each dataset — so there is no
    fragile substring matching against column names.
    """
    # Convert to a set so membership checks are fast regardless of list length.
    qc_column_set = set(qc_columns)
    stats_columns = [column_name for column_name in df.columns if column_name not in qc_column_set]

    # Both outputs re-emit core columns first so they read naturally.
    # The `column_name not in core_columns` filter prevents duplicating core columns.
    stats_df = df[
        core_columns
        + [column_name for column_name in stats_columns if column_name not in core_columns]
    ].copy()
    qc_df = df[
        core_columns
        + [column_name for column_name in qc_columns if column_name not in core_columns]
    ].copy()
    return stats_df, qc_df
