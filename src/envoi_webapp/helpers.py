from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError

from envoi import extract as envoi_extract
from envoi import init_gee as envoi_init_gee
from envoi.progress import ProgressCallback

GBIF_REQUIRED_COLUMNS = ("occurrenceID", "decimalLatitude", "decimalLongitude")
GBIF_DATE_COLUMN = "eventDate"
TABULAR_OUTPUT = "tabular"
RASTER_OUTPUT = "raster"
RASTER_RESAMPLE_M = 10


@dataclass(frozen=True)
class CsvValidationResult:
    row_count: int
    columns: tuple[str, ...]
    has_date: bool


@dataclass(frozen=True)
class DatasetSelection:
    dataset: str
    window_sizes: tuple[int, ...]
    statistics: tuple[str, ...] = ()


def read_points_csv(source: Any) -> pd.DataFrame:
    """Read a points CSV while inferring common delimiters."""

    if hasattr(source, "seek"):
        source.seek(0)
    return pd.read_csv(source, sep=None, engine="python")


def validate_points_dataframe(df: pd.DataFrame) -> CsvValidationResult:
    """Validate the MVP GBIF-style input table."""

    if df.empty:
        raise ValueError("CSV must contain at least one data row.")

    missing = [column for column in GBIF_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        expected = ", ".join(GBIF_REQUIRED_COLUMNS)
        raise ValueError(
            f"CSV is missing required column(s): {missing}. Expected columns: {expected}; "
            f"optional date column: {GBIF_DATE_COLUMN}."
        )

    for column in ("decimalLatitude", "decimalLongitude"):
        try:
            values = pd.to_numeric(df[column], errors="raise")
        except Exception as exc:
            raise ValueError(f"Column '{column}' must contain numeric coordinate values.") from exc
        if values.isna().any():
            raise ValueError(f"Column '{column}' contains missing coordinate values.")

    return CsvValidationResult(
        row_count=len(df),
        columns=tuple(str(column) for column in df.columns),
        has_date=GBIF_DATE_COLUMN in df.columns,
    )


def normalize_crs(value: str | None) -> str:
    """Validate a CRS string and return a stable representation."""

    text = (value or "").strip()
    if not text:
        return "EPSG:4326"

    upper = text.upper()
    if upper == "WGS84":
        text = "EPSG:4326"
    elif text.isdigit():
        text = f"EPSG:{text}"

    try:
        crs = CRS.from_user_input(text)
    except (CRSError, ValueError) as exc:
        raise ValueError(f"Invalid CRS '{value}'. Use an EPSG code such as EPSG:4326.") from exc

    epsg = crs.to_epsg()
    return f"EPSG:{epsg}" if epsg is not None else crs.to_string()


def validate_wgs84_ranges(df: pd.DataFrame, input_crs: str) -> None:
    """Validate coordinate ranges only when the declared input CRS is WGS84."""

    crs = CRS.from_user_input(normalize_crs(input_crs))
    if not crs.equals(CRS.from_epsg(4326)):
        return

    latitudes = pd.to_numeric(df["decimalLatitude"], errors="raise")
    longitudes = pd.to_numeric(df["decimalLongitude"], errors="raise")
    bad_latitudes = latitudes.abs() > 90
    bad_longitudes = longitudes.abs() > 180
    if bad_latitudes.any() or bad_longitudes.any():
        raise ValueError(
            "Coordinates are outside valid WGS84 ranges. If the CSV uses another CRS, "
            "choose that CRS before running extraction."
        )


def validate_output_dir(path_value: str | Path) -> Path:
    """Create and validate the local output directory."""

    raw_path = str(path_value).strip()
    if not raw_path:
        raise ValueError("Output directory is required.")

    path = Path(raw_path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise ValueError(f"Output path is not a directory: {path}")

    fd, test_path = tempfile.mkstemp(prefix=".envoi-write-test-", dir=path)
    os.close(fd)
    Path(test_path).unlink(missing_ok=True)
    return path.resolve()


def parse_window_sizes(value: str) -> tuple[int, ...]:
    """Parse comma-separated positive integer window sizes."""

    parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if not parts:
        raise ValueError("At least one window size is required.")

    window_sizes: list[int] = []
    for part in parts:
        try:
            window_size = int(part)
        except ValueError as exc:
            raise ValueError(f"Window size must be a positive integer: {part!r}.") from exc
        if window_size <= 0:
            raise ValueError(f"Window size must be positive: {window_size}.")
        window_sizes.append(window_size)

    return tuple(window_sizes)


def default_statistics_for_dataset(dataset_entry: dict) -> tuple[str, ...]:
    """Choose practical default reducers from catalog data type."""

    if dataset_entry.get("data_type") == "categorical":
        return ("mode", "class_fraction")
    return ("mean", "std")


def _window_size_setting(window_sizes: Sequence[int]) -> int | list[int]:
    if len(window_sizes) == 1:
        return int(window_sizes[0])
    return [int(window_size) for window_size in window_sizes]


def _batch_id(index: int, dataset: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", dataset).strip("_").lower()
    slug = slug or "dataset"
    return f"extract_{index:02d}_{slug}"


def build_run_config(
    rows: Sequence[DatasetSelection],
    output_type: str,
) -> list[dict]:
    """Build the config list consumed by envoi.extract()."""

    if output_type not in {TABULAR_OUTPUT, RASTER_OUTPUT}:
        raise ValueError("Output type must be 'tabular' or 'raster'.")
    if not rows:
        raise ValueError("At least one dataset row is required.")

    configs: list[dict] = []
    for index, row in enumerate(rows, start=1):
        if not row.dataset:
            raise ValueError(f"Dataset row {index} is missing a dataset.")
        if not row.window_sizes:
            raise ValueError(f"Dataset row {index} is missing window sizes.")

        settings: dict = {
            "output_type": output_type,
            "window_size_m": _window_size_setting(row.window_sizes),
        }
        if output_type == TABULAR_OUTPUT:
            if not row.statistics:
                raise ValueError(f"Dataset row {index} needs at least one summary statistic.")
            settings["statistics"] = list(row.statistics)
            settings["output_file_format"] = "csv"
        else:
            settings["resample_m"] = RASTER_RESAMPLE_M

        configs.append(
            {
                "batch_id": _batch_id(index, row.dataset),
                "datasets": [row.dataset],
                "settings": settings,
            }
        )

    return configs


def validate_service_account_json(raw_bytes: bytes) -> dict:
    """Validate the minimum shape of a Google service-account key JSON."""

    if not raw_bytes:
        raise ValueError("Earth Engine service account key JSON is required.")

    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid service account JSON.") from exc

    if not isinstance(data, dict):
        raise ValueError("Service account key JSON must be an object.")

    required = {"type", "client_email", "private_key"}
    missing = sorted(required - set(data.keys()))
    if data.get("type") != "service_account" or missing:
        raise ValueError("The uploaded JSON is not a valid Google service account key.")

    return data


@contextmanager
def temporary_service_account_file(raw_bytes: bytes) -> Iterator[Path]:
    """Write credentials to a private temp file for one run, then delete it."""

    validate_service_account_json(raw_bytes)
    fd, path_string = tempfile.mkstemp(prefix="envoi-ee-", suffix=".json")
    path = Path(path_string)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw_bytes)
        os.chmod(path, 0o600)
        yield path
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def redact_credential_secrets(message: str, raw_credential_bytes: bytes | None) -> str:
    """Remove service-account secret values from a user-facing error message."""

    if raw_credential_bytes is None:
        return message

    redacted = message
    secret_values: list[str] = []
    try:
        credential_text = raw_credential_bytes.decode("utf-8")
        secret_values.append(credential_text)
        data = json.loads(credential_text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        data = None

    if isinstance(data, dict):
        for key in ("private_key", "private_key_id", "client_email"):
            value = data.get(key)
            if isinstance(value, str) and value:
                secret_values.append(value)
                if key == "private_key":
                    secret_values.extend(
                        line.strip()
                        for line in value.splitlines()
                        if line.strip() and not line.startswith("-----")
                    )

    for secret in sorted(set(secret_values), key=len, reverse=True):
        redacted = redacted.replace(secret, "[redacted]")
    return redacted


def run_extraction(
    points: pd.DataFrame,
    dataset_rows: Sequence[DatasetSelection],
    output_type: str,
    output_dir: str | Path,
    input_crs: str,
    credentials_json: bytes,
    *,
    progress_callback: ProgressCallback | None = None,
    init_gee_func: Callable = envoi_init_gee,
    extract_func: Callable = envoi_extract,
):
    """Validate web inputs, initialize Earth Engine, and run envoi.extract()."""

    validate_points_dataframe(points)
    normalized_crs = normalize_crs(input_crs)
    validate_wgs84_ranges(points, normalized_crs)
    resolved_output_dir = validate_output_dir(output_dir)
    config = build_run_config(dataset_rows, output_type)

    with temporary_service_account_file(credentials_json) as credentials_path:
        init_gee_func(credentials_path)
        return extract_func(
            points,
            config,
            output_dir=resolved_output_dir,
            input_crs=normalized_crs,
            id_column="occurrenceID",
            latitude_column="decimalLatitude",
            longitude_column="decimalLongitude",
            date_column="eventDate",
            quiet=True,
            progress_callback=progress_callback,
        )
