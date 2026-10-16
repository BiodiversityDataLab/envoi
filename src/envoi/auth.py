# src/envoi/auth.py
from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import ee

# Environment variable users can set to point at their service account JSON.
# Takes priority over every other lookup so CI / Docker / headless runs can
# inject credentials without touching the filesystem layout.
ENV_VAR = "ENVOI_EE_CREDENTIALS"

# User-level config location, resolved per-platform so each OS gets its
# conventional path:
#   * Windows: %APPDATA%\envoi\ee_credentials.json
#     (typically C:\Users\<you>\AppData\Roaming\envoi\...)
#   * macOS / Linux: ~/.config/envoi/ee_credentials.json (XDG-style)
# Falls back to the standard Roaming subpath on Windows when %APPDATA% is
# unset (rare — usually only in stripped-down containers).
if sys.platform == "win32":
    _windows_appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    USER_CONFIG_PATH = Path(_windows_appdata) / "envoi" / "ee_credentials.json"
else:
    USER_CONFIG_PATH = Path.home() / ".config" / "envoi" / "ee_credentials.json"

# Project-local fallback. Lets the dev workflow keep working without any
# env-var setup: drop the key at <project>/credentials/ee_credentials.json
# and run from the project root.
CWD_RELATIVE_PATH = Path("credentials") / "ee_credentials.json"


def _default_credentials_path() -> Path | None:
    """Find the credentials file in the first location that exists.

    Lookup order:
      1. ``$ENVOI_EE_CREDENTIALS`` environment variable
      2. user config dir (``~/.config/envoi/ee_credentials.json`` on
         macOS/Linux, ``%APPDATA%\\envoi\\ee_credentials.json`` on Windows)
      3. ``./credentials/ee_credentials.json`` (relative to current working dir)

    Returns the matching :class:`Path`, or ``None`` when nothing is found.
    The env-var path is returned even if the file does not exist so the
    error message can point at the user's explicit choice instead of
    silently falling through to the next tier.
    """
    env_value = os.environ.get(ENV_VAR)
    if env_value:
        return Path(env_value)
    if USER_CONFIG_PATH.exists():
        return USER_CONFIG_PATH
    cwd_path = Path.cwd() / CWD_RELATIVE_PATH
    if cwd_path.exists():
        return cwd_path
    return None


def _build_credentials_from_json(credentials_json: str | bytes | Mapping[str, Any]) -> Any:
    """Build Earth Engine credentials from the content of a service account key.

    Args:
        credentials_json: The key as JSON text (``str``, or ``bytes`` in
            UTF-8) or as a mapping of the parsed JSON.

    Returns:
        The credentials object from ``ee.ServiceAccountCredentials``.

    Raises:
        ValueError: when ``credentials_json`` has a wrong type or is not a
            valid service account key. The error message, its cause, and its
            context contain no part of the key.
    """
    if not isinstance(credentials_json, (str, bytes, Mapping)):
        raise ValueError(
            "credentials_json must be the content of a service account key: JSON text "
            f"(str or bytes) or a dict. Got {type(credentials_json).__name__}."
        )

    # Convert the key to JSON text and build the credentials. For a key without
    # "token_uri", or with a malformed "private_key", ee.ServiceAccountCredentials
    # retries the parsed key as a PEM key, and google-auth then raises an error
    # whose message contains the whole key, private key included.
    key_is_invalid = False
    try:
        if isinstance(credentials_json, bytes):
            key_text = credentials_json.decode("utf-8")
        elif isinstance(credentials_json, str):
            key_text = credentials_json
        else:
            key_text = json.dumps(dict(credentials_json))
        credentials = ee.ServiceAccountCredentials(email=None, key_data=key_text)
    # Broad catch, re-raised below as a clear ValueError. On purpose, the
    # ValueError does not chain the original error, unlike "Chain converted
    # third-party exceptions" in docs/coding_guidelines.md: the original error
    # can contain the key. A "raise ... from None" inside this block still keeps
    # the original error as __context__ (PEP 415), so this block only sets a flag.
    except Exception:
        key_is_invalid = True
    if key_is_invalid:
        raise ValueError(
            "The supplied service account key is not a valid Google service account key. "
            "Supply the complete, unchanged content of the JSON key file from the Google "
            "Cloud Console. It must contain 'type', 'client_email', 'private_key', "
            "and 'token_uri'."
        )
    return credentials


def init_gee(
    credentials_path: str | Path | None = None,
    *,
    credentials_json: str | bytes | Mapping[str, Any] | None = None,
) -> None:
    """Initialize Earth Engine from a Google service account key JSON.

    The key file is the JSON downloaded from the Google Cloud Console for
    a service account that has Earth Engine access. It is the standard
    Google-issued file — no extra wrapper is needed.

    Args:
        credentials_path: Path to the service account JSON. When omitted,
            looks for the file in (1) ``$ENVOI_EE_CREDENTIALS``,
            (2) the user config dir — ``~/.config/envoi/ee_credentials.json``
            on macOS/Linux or ``%APPDATA%\\envoi\\ee_credentials.json`` on
            Windows, then (3) ``./credentials/ee_credentials.json``. Pass
            an explicit path to bypass the lookup.
        credentials_json: The content of the service account key instead of
            a path: JSON text (``str``, or ``bytes`` in UTF-8) or a dict of
            the parsed JSON. Use it when the key comes from a secret store or
            an upload. Earth Engine then uses only this key, and the file
            lookup above does not run. Do not combine it with
            ``credentials_path``.

    Raises:
        ValueError: when both ``credentials_path`` and ``credentials_json``
            are given, or when ``credentials_json`` is not a valid service
            account key. The error contains no part of the key.
        FileNotFoundError: when no credentials file is found in any of the
            checked locations. The error message lists every location it
            looked at so the user can fix it without guessing.
        RuntimeError: when Earth Engine refuses the credentials (typically
            because the service account lacks GEE access).
    """
    if credentials_path is not None and credentials_json is not None:
        raise ValueError(
            "init_gee() takes credentials_path or credentials_json, not both. "
            "Pass the path of the key file, or the content of the key."
        )

    # Build the credentials. A key given as content skips the file lookup, so a
    # key file on the computer cannot replace it.
    if credentials_json is not None:
        credentials = _build_credentials_from_json(credentials_json)
        key_source = "the supplied key"
        key_container = "key"
    else:
        # Resolve the path: explicit argument wins, otherwise walk the lookup chain.
        if credentials_path is not None:
            path = Path(credentials_path)
        else:
            path = _default_credentials_path()

        if path is None or not path.exists():
            # Build the location list so the error message is actionable. The
            # env-var line shows the current value when set so the user can
            # spot typos in their config.
            env_value = os.environ.get(ENV_VAR)
            env_line = (
                f"  - ${ENV_VAR} (currently: {env_value!r})"
                if env_value
                else f"  - ${ENV_VAR} (not set)"
            )
            checked = "\n".join(
                [
                    env_line,
                    f"  - {USER_CONFIG_PATH}",
                    f"  - {Path.cwd() / CWD_RELATIVE_PATH}",
                ]
            )
            raise FileNotFoundError(
                "Google Earth Engine credentials not found. Checked:\n"
                f"{checked}\n"
                "Download a service account JSON from the GCP Console and either "
                f"set ${ENV_VAR} or place it at one of the paths above."
            )

        # ee.ServiceAccountCredentials reads the email out of the JSON itself,
        # so we don't need to crack the file open here — just pass the path.
        credentials = ee.ServiceAccountCredentials(email=None, key_file=str(path))
        key_source = f"'{path}'"
        key_container = "file"

    try:
        ee.Initialize(credentials)
    except Exception as e:
        raise RuntimeError(
            f"Google Earth Engine authentication failed using {key_source}.\n"
            f"Check that the service account in the {key_container} is valid and has "
            f"access to GEE.\n"
            f"Original error: {e}"
        ) from e
