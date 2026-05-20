# src/envoi/auth.py
import json
from pathlib import Path

import ee

CREDENTIALS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "credentials" / "ee_credentials.json"
)


def init_gee(credentials_path: str | Path | None = None):
    """Initialize Earth Engine from a JSON credentials file.

    The JSON file must contain 'service_account' and 'private_key' keys.
    """
    path = Path(credentials_path) if credentials_path else CREDENTIALS_PATH
    if not path.exists():
        raise FileNotFoundError(f"GEE credentials not found: {path}")

    with open(path) as f:
        creds_data = json.load(f)

    service_account = creds_data["service_account"]
    key_file = creds_data["private_key"]

    credentials = ee.ServiceAccountCredentials(service_account, key_file)
    try:
        ee.Initialize(credentials)
    except Exception as e:
        raise RuntimeError(
            f"Google Earth Engine authentication failed.\n"
            f"Check that the service account in '{path}' is valid and has access to GEE.\n"
            f"Original error: {e}"
        ) from e
