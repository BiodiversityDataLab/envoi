# src/envoi/adapters/earth_engine/_tiles.py
"""Everything that exports a GeoTIFF tile from GEE.

* The synchronous-download size limit constants and the pre-flight size guard
  (``_check_tile_size``) — pure Python so it can be unit-tested without GEE
  credentials.
* The tile download itself (``_download_tile_via_url``) — wraps
  ``ee.Image.getDownloadURL`` + ``requests`` with retry logic for transient
  failures.

The adapter still owns ``_export_single`` (which knows how to compute the
per-point pixel grid) and the thin ``_guard_tile_size`` wrapper (which pulls the
band count off the live ee.Image before delegating here).
"""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from pathlib import Path

import ee
import requests

from ...geo import tile_pixel_count

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
# exponential backoff. Three attempts means waits of 1 s and 2 s between
# tries — enough to ride out brief GEE hiccups without making a failed batch
# crawl. There is no wait after the last attempt.
_TILE_DOWNLOAD_MAX_RETRIES = 3

# How much of the body of an HTTP error response envoi reads to find Earth
# Engine's error message, and how long the message in the error can be.
# Reading only the start of the body keeps a large body out of memory.
_MAX_ERROR_BODY_BYTES = 4096
_MAX_HTTP_ERROR_TEXT_LENGTH = 300


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
    # size. tile_pixel_count is the pixel count per side that _export_single
    # requests, so the estimate matches the real tile.
    pixels_per_side = tile_pixel_count(window_m, scale_m)
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


def _read_http_error_text(response: requests.Response) -> str:
    """Return the error message in the body of an HTTP error response.

    Reads at most the first 4 KiB of the body. Earth Engine answers with JSON
    ``{"error": {"message": ...}}``; for any other body, the text itself is the
    message.

    Returns:
        The message on one line, at most 300 characters. An empty string when
        the body cannot be read.
    """
    try:
        body = next(response.iter_content(chunk_size=_MAX_ERROR_BODY_BYTES), b"")
    except requests.RequestException:
        return ""
    body_text = body.decode("utf-8", errors="replace")
    try:
        message = json.loads(body_text)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        message = body_text
    return " ".join(str(message).split())[:_MAX_HTTP_ERROR_TEXT_LENGTH]


def _download_tile_via_url(
    img: ee.Image,
    *,
    crs: str,
    crs_transform: list[float],
    dimensions: str,
    output_path: Path,
) -> None:
    """Download one tile via ``ee.Image.getDownloadURL`` + ``requests``.

    Asks GEE for a single multi-band GeoTIFF (``format='GEO_TIFF'``
    with ``filePerBand=False``) so we get a TIFF in the response body
    directly — no zip wrapping, no per-band extraction step. Streams
    the response to a temporary file next to ``output_path``, with
    retries for transient errors, and moves it to ``output_path`` when it
    is complete. A failed download leaves no file: it also removes a file
    at ``output_path`` from an earlier run, so that the folder matches the
    tile manifest, which marks the tile as not exported.

    Args:
        img: The image to download.
        crs: CRS of the tile, for example ``"EPSG:32633"``.
        crs_transform: Affine transform of the tile in ``crs``, in the order
            ``[x pixel size, 0, west edge, 0, -y pixel size, north edge]``.
        dimensions: Tile size in pixels, ``"<width>x<height>"``.
        output_path: Path of the GeoTIFF to write. Its folder must exist.

    Raises:
        requests.HTTPError: Earth Engine answered with an HTTP status other
            than 200, 429, and 5xx (for example 400). The message has the
            status and Earth Engine's error text. It is not retried.
        RuntimeError: Every attempt failed with HTTP 429, a 5xx status, or a
            connection error or timeout (also during the stream). The error
            is chained to the error of the last attempt.
    """
    # Stream the body to a temporary file in the same folder, and give it the
    # tile's name only after the last chunk. A download that fails part way,
    # with any error, then leaves no partial GeoTIFF that looks like a valid
    # tile. The temporary name is short and unique: the tile's name plus a
    # suffix could exceed the 255-byte limit of a file name.
    partial_path = output_path.with_name(f"download-{uuid.uuid4().hex}.part")
    tile_written = False
    try:
        # GEE's getDownloadURL is server-side: it builds the URL synchronously
        # but the URL itself points to a streamed export, so the actual image
        # generation happens during the GET below.
        # With crs, crs_transform and dimensions, and without region or scale,
        # the Earth Engine client turns the dimensions into a rectangle in that
        # projection, so Earth Engine returns exactly this rectangle of pixels
        # (ee.Image._apply_crs_and_affine). Do not add a region or a scale.
        # With a region, Earth Engine raises an error. A scale has no use:
        # Earth Engine ignores it when crs and crs_transform are set
        # (getDownloadURL documentation), and with it the client no longer
        # turns the dimensions into this rectangle.
        url = img.getDownloadURL(
            {
                "crs": crs,
                "crs_transform": crs_transform,
                "dimensions": dimensions,
                "format": "GEO_TIFF",
                "filePerBand": False,
            }
        )

        # Retry loop for transient errors (429 = rate-limited, 5xx = GEE-side
        # blip, and connection errors or timeouts, also during the stream).
        # Any status other than 200 is permanent: a retry gets the same
        # answer, so it raises at once.
        last_error: Exception | None = None
        for attempt in range(_TILE_DOWNLOAD_MAX_RETRIES):
            # Back off before each retry (1 s, then 2 s), not after the last attempt.
            if attempt > 0:
                time.sleep(2 ** (attempt - 1))
            try:
                response = requests.get(url, stream=True, timeout=_TILE_DOWNLOAD_TIMEOUT_S)
            except requests.RequestException as request_error:
                last_error = request_error
                continue

            status_code = response.status_code
            if status_code == 429 or 500 <= status_code < 600:
                # Drain the connection before retrying.
                response.close()
                last_error = RuntimeError(
                    f"GEE tile download returned HTTP {status_code} on attempt {attempt + 1}"
                )
                continue
            if status_code != 200:
                # Keep Earth Engine's own message (for example why the request
                # is invalid): raise_for_status() would give only the status.
                error_message = f"GEE tile download returned HTTP {status_code}"
                error_text = _read_http_error_text(response)
                if error_text:
                    error_message += f": {error_text}"
                response.close()
                raise requests.HTTPError(error_message, response=response)

            # Stream to disk in 1 MiB chunks so peak memory stays bounded
            # even for the largest allowed tiles (~32 MB).
            try:
                with partial_path.open("wb") as partial_file:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            partial_file.write(chunk)
            except requests.RequestException as stream_error:
                last_error = stream_error
                continue

            # The tile is complete. replace() also overwrites a tile of an
            # earlier run with the same name, on Windows too.
            partial_path.replace(output_path)
            tile_written = True
            return

        # Exhausted retries — surface the last error so export_tiles can
        # log it per-point and mark the result slot as None.
        raise RuntimeError(
            f"Failed to download GEE tile after {_TILE_DOWNLOAD_MAX_RETRIES} attempts: "
            f"{last_error}"
        ) from last_error
    finally:
        # Remove the partial download. After a successful download, the
        # temporary file no longer exists. A failed removal must not hide the
        # download error or turn a successful download into a failed tile.
        with contextlib.suppress(OSError):
            partial_path.unlink(missing_ok=True)
        # When the tile failed, the tile manifest marks it as not exported.
        # Remove a tile with the same name from an earlier run, so that the
        # folder matches the manifest.
        if not tile_written:
            with contextlib.suppress(OSError):
                output_path.unlink(missing_ok=True)
