# src/envoi/_output_assembly.py
"""Tabular-output post-processing for :func:`envoi.extract`.

Lives between the per-dataset adapter results and the file written to disk.
Responsibilities:

* :func:`_append_stat_columns` — turn per-point ``(stats_dict, meta_dict)``
  results into named DataFrame columns, with special handling for the per-
  class keys produced by ``class_count`` / ``class_fraction``.
* :func:`_round_stat_columns` — round all non-core stat columns to a uniform
  decimal precision for output stability.
* :func:`_restore_user_column_names` — invert the canonical ``id``/``lat``/
  ``lon``/``date`` rename so user-chosen names round-trip through extract().
* :func:`_resolve_dataset_metadata` — record what bands were actually applied
  (per-call override vs catalog default) for the metadata sidecar.
* :func:`_write_tabular` — write a DataFrame to CSV or Parquet, creating the
  parent directory if needed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Matches the per-class stat keys produced by the categorical reducers, in
# either single-band form ("class_10_count") or multi-band form
# ("b2_class_10_fraction"). Used to zero-fill rows that didn't see a class
# observed by some other row in the same batch.
_CLASS_COLUMN_RE = re.compile(r"^(?:b\d+_)?class_-?\d+_(count|fraction)$")


def _round_stat_columns(
    stats_df: pd.DataFrame,
    core_columns: list[str],
    decimals: int,
) -> pd.DataFrame:
    """Round all non-core stat columns to the given number of decimal places.

    Core columns (id, lat, lon, date) are excluded so coordinate precision
    is preserved exactly as the user supplied it.
    """
    stat_columns = [c for c in stats_df.columns if c not in core_columns]
    stats_df[stat_columns] = stats_df[stat_columns].round(decimals)
    return stats_df


def _restore_user_column_names(
    stats_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    column_name_map: Dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rename canonical core columns back to the user's original names.

    `column_name_map` maps user names → canonical names (e.g. "sample_id" →
    "id"). We invert it to rename the output back so the user's chosen names
    round-trip through extract().
    """
    output_rename = {
        canonical: user for user, canonical in column_name_map.items() if user != canonical
    }
    if output_rename:
        stats_df = stats_df.rename(columns=output_rename)
        qc_df = qc_df.rename(columns=output_rename)
    return stats_df, qc_df


def _resolve_dataset_metadata(
    datasets: List[Tuple[str, Dict[str, Any]]],
    catalog_datasets: dict,
) -> List[Dict[str, Any]]:
    """Build the resolved per-dataset list stored in the metadata sidecar.

    For each (name, band_overrides) pair, records the bands and derived_bands
    that were actually used — the per-call override when supplied, otherwise
    the catalog's value, otherwise None. Storing this alongside the run config
    makes a run reproducible without having to consult the catalog separately.
    """
    resolved = []
    for dataset, band_overrides in datasets:
        dataset_config = catalog_datasets[dataset]
        resolved.append(
            {
                "name": dataset,
                "bands": band_overrides.get("bands", dataset_config.get("bands")),
                "derived_bands": band_overrides.get(
                    "derived_bands", dataset_config.get("derived_bands")
                ),
            }
        )
    return resolved


def _append_stat_columns(
    df: pd.DataFrame,
    dataset_name: str,
    window_size_m: int,
    stats_results: list[tuple[dict, dict]],
) -> pd.DataFrame:
    """Append adapter statistics columns to the output dataframe.

    Expected stat keys from adapters:
    - window, single-band: "{reducer}"      -> column "{dataset}_{reducer}_{buf}m"
    - window, multi-band:  "b{n}_{reducer}" -> column "{dataset}_b{n}_{reducer}_{buf}m"
    - point,  single-band: "point"          -> column "{dataset}_point"
    - point,  multi-band:  "{band}_point"   -> column "{dataset}_{band}_point"

    Point keys intentionally skip the buffer suffix to preserve the historical schema.

    Categorical reducers (``class_count`` / ``class_fraction``) produce
    per-class stat keys like ``class_10_count`` or ``b2_class_20_fraction``.
    Each class value found in *any* row's stat dict becomes an output column;
    rows whose dict didn't see that class are filled with 0 (the design
    decision is that "class not observed" = 0, regardless of whether the
    window itself was valid; the QC sidecar records extent/coverage
    separately).
    """
    # Collect keys in first-seen order so output columns stay stable across runs.
    stat_keys = dict.fromkeys(key for stat_dict, _ in stats_results for key in stat_dict)

    # Build a batch of new columns so we can add them in one concat.
    # Columns that already exist should be overwritten (not duplicated),
    # which can happen for point stats when multiple window sizes are run.
    new_columns: dict[str, list] = {}
    existing_columns = set(df.columns)
    for stat_key in stat_keys:
        if stat_key == "point" or stat_key.endswith("_point"):
            column_name = f"{dataset_name}_{stat_key}"
        else:
            column_name = f"{dataset_name}_{stat_key}_{window_size_m}m"

        # Decide the fill value for missing entries:
        # - class_*_count   -> 0   (absent class = zero pixels)
        # - class_*_fraction -> 0.0 (absent class = 0 % of window)
        # - everything else -> None (existing behaviour for missing stats)
        class_match = _CLASS_COLUMN_RE.match(stat_key)
        if class_match is not None:
            missing_fill = 0 if class_match.group(1) == "count" else 0.0
        else:
            missing_fill = None

        # Pull one value per sample row, using the chosen fill for absent keys.
        column_values = [
            stat_dict[stat_key] if stat_key in stat_dict else missing_fill
            for stat_dict, _ in stats_results
        ]

        # Overwrite existing columns (keeps schema stable across windows).
        if column_name in existing_columns:
            df[column_name] = column_values
        else:
            new_columns[column_name] = column_values

    # Add all new columns at once to avoid DataFrame fragmentation warnings.
    if new_columns:
        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    return df


def _write_tabular(df: pd.DataFrame, name: str, output_dir: Path, output_file_format: str) -> Path:
    """Write a DataFrame to a CSV or Parquet file and return the path written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_file_format == "parquet":
        path = output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
    else:
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
    return path
