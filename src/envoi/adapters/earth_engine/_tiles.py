# src/envoi/adapters/earth_engine/_tiles.py
"""Everything that exports a GeoTIFF tile from GEE.

* The synchronous-download size limit constants and the pre-flight size guard
  (``_check_tile_size``) — pure Python so it can be unit-tested without GEE
  credentials.
* The tile download itself (``_download_tile_via_url``) — wraps
  ``ee.Image.getDownloadURL`` + ``requests`` with retry logic for transient
  failures.

The adapter still owns ``_export_single`` (which knows how to assemble the
per-point region) and the thin ``_guard_tile_size`` wrapper (which pulls the
band count off the live ee.Image before delegating here).
"""

from __future__ import annotations

import time
from pathlib import Path

import ee
import requests

# GEE's synchronous getDownloadURL endpoint refuses responses larger than
# about 32 MB (the documented limit; some empirical reports place it closer
# to 48–50 MB). We pre-check against the conservative number so users get an
# actionable error from us rather than an opaque 400 from GEE.
GEE_SYNC_LIMIT_BYTES = 32 * 1024 * 1024

# How long to wait for a single tile download before giving up. Tiles are at
# most a few tens of MB and GEE usually answers in seconds, but the request
# also covers GEE's server-side image assembly, which can take longer for
# heavy ImageCollection composites.
_TILE_DOWNLOAD_TIMEOUT_S = 120

# Number of times we retry transient HTTP failures (429 / 5xx) with
# exponential backoff. Three attempts means waits of roughly 1s, 2s, 4s
# between tries — enough to ride out brief GEE hiccups without making a
# failed batch crawl.
_TILE_DOWNLOAD_MAX_RETRIES = 3


def _check_tile_size(
    *,
    window_m: float,
    scale_m: float,
    band_count: int,
    dataset_name: str,
) -> None:
    """Raise ValueError if a sync download would exceed GEE's response limit.

    Pure-Python size estimate — no GEE calls — so it can be unit-tested
    without credentials and lifted out of :meth:`GeeRasterAdapter._guard_tile_size`.

    The error message names every knob the user can turn so they can fix
    the problem without reading our source.
    """
    # window_m is the tile side length in metres and scale_m is the pixel
    # size, so (window_m / scale_m) is the pixel count per side.
    # max(1, ...) mirrors the pixel floor used when building the export
    # region — a sub-pixel window still produces a 1×1 raster, not zero.
    pixels_per_side = max(1, round(window_m / scale_m))
    total_pixels = pixels_per_side * pixels_per_side

    # Assume float32 (4 bytes/pixel) as a conservative upper bound. GEE's
    # actual response is often smaller (int16 + compression), but
    # overshooting here is safer than letting an oversized request through.
    estimated_bytes = total_pixels * max(1, band_count) * 4
    if estimated_bytes <= GEE_SYNC_LIMIT_BYTES:
        return

    raise ValueError(
        f"Requested tile is too large for GEE's synchronous download "
        f"endpoint (estimated {estimated_bytes / 1e6:.1f} MB > "
        f"{GEE_SYNC_LIMIT_BYTES / 1e6:.0f} MB limit). "
        f"Reduce window_size_m (currently {window_m}), "
        f"increase resample_m (currently {scale_m}), or select fewer "
        f"bands for dataset '{dataset_name}'."
    )


def _download_tile_via_url(
    img: ee.Image,
    *,
    region: ee.Geometry,
    scale_m: float,
    crs: str,
    output_path: Path,
) -> None:
    """Download one tile via ``ee.Image.getDownloadURL`` + ``requests``.

    Asks GEE for a single multi-band GeoTIFF (``format='GEO_TIFF'``
    with ``filePerBand=False``) so we get a TIFF in the response body
    directly — no zip wrapping, no per-band extraction step. Streams
    the response straight to ``output_path`` with retries for
    transient HTTP errors.
    """
    # GEE's getDownloadURL is server-side: it builds the URL synchronously
    # but the URL itself points to a streamed export, so the actual image
    # generation happens during the GET below.
    url = img.getDownloadURL(
        {
            "scale": scale_m,
            "region": region,
            "crs": crs,
            "format": "GEO_TIFF",
            "filePerBand": False,
        }
    )

    # Retry loop for transient errors (429 = rate-limited, 5xx = GEE-side
    # blip). Permanent errors (4xx other than 429) raise immediately so
    # we don't waste time retrying a bad request.
    last_error: Exception | None = None
    for attempt in range(_TILE_DOWNLOAD_MAX_RETRIES):
        try:
            response = requests.get(url, stream=True, timeout=_TILE_DOWNLOAD_TIMEOUT_S)
            status_code = response.status_code
            if status_code == 429 or 500 <= status_code < 600:
                # Drain the connection and back off before retrying.
                response.close()
                last_error = RuntimeError(
                    f"GEE tile download returned HTTP {status_code} on attempt {attempt + 1}"
                )
                time.sleep(2**attempt)
                continue
            response.raise_for_status()

            # Stream to disk in 1 MiB chunks so peak memory stays bounded
            # even for the largest allowed tiles (~32 MB).
            with output_path.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        output_file.write(chunk)
            return
        except requests.RequestException as request_error:
            # Connection errors / timeouts also get the backoff treatment.
            last_error = request_error
            time.sleep(2**attempt)

    # Exhausted retries — surface the last error so export_tiles can
    # log it per-point and mark the result slot as None.
    raise RuntimeError(
        f"Failed to download GEE tile after {_TILE_DOWNLOAD_MAX_RETRIES} attempts: {last_error}"
    )
