# src/envoi/auth.py
import os
from pathlib import Path

import ee

# Environment variable users can set to point at their service account JSON.
# Takes priority over every other lookup so CI / Docker / headless runs can
# inject credentials without touching the filesystem layout.
ENV_VAR = "ENVOI_EE_CREDENTIALS"

# User-level config location, following the XDG-style convention used by
# many cross-platform Python tools. Works the same on Linux, macOS, and
# Windows (Path.home() resolves correctly on all three).
USER_CONFIG_PATH = Path.home() / ".config" / "envoi" / "ee_credentials.json"

# Project-local fallback. Lets the dev workflow keep working without any
# env-var setup: drop the key at <project>/credentials/ee_credentials.json
# and run from the project root.
CWD_RELATIVE_PATH = Path("credentials") / "ee_credentials.json"


def _default_credentials_path() -> Path | None:
    """Find the credentials file in the first location that exists.

    Lookup order:
      1. ``$ENVOI_EE_CREDENTIALS`` environment variable
      2. ``~/.config/envoi/ee_credentials.json``
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


def init_gee(credentials_path: str | Path | None = None) -> None:
    """Initialize Earth Engine from a Google service account key JSON.

    The key file is the JSON downloaded from the Google Cloud Console for
    a service account that has Earth Engine access. It is the standard
    Google-issued file — no extra wrapper is needed.

    Args:
        credentials_path: Path to the service account JSON. When omitted,
            looks for the file in (1) ``$ENVOI_EE_CREDENTIALS``,
            (2) ``~/.config/envoi/ee_credentials.json``, then
            (3) ``./credentials/ee_credentials.json``. Pass an explicit
            path to bypass the lookup.

    Raises:
        FileNotFoundError: when no credentials file is found in any of the
            checked locations. The error message lists every location it
            looked at so the user can fix it without guessing.
        RuntimeError: when Earth Engine refuses the credentials (typically
            because the service account lacks GEE access).
    """
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
    try:
        ee.Initialize(credentials)
    except Exception as e:
        raise RuntimeError(
            f"Google Earth Engine authentication failed using '{path}'.\n"
            f"Check that the service account in the file is valid and has "
            f"access to GEE.\n"
            f"Original error: {e}"
        ) from e
