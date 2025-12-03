# auth.py
import ee
from pathlib import Path

KEY_JSON = Path("secrets/REDACTED_KEY.json")
SERVICE_ACCOUNT = "REDACTED_SERVICE_ACCOUNT"  # fill in
PROJECT_ID = "REDACTED_PROJECT"  # optional but recommended


def init_gee():
    """
    Initialize Earth Engine using the service account JSON key
    already in secrets/.
    """
    if not KEY_JSON.exists():
        raise FileNotFoundError(f"Service account key not found: {KEY_JSON}")

    creds = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, str(KEY_JSON))
    ee.Initialize(creds, project=PROJECT_ID)
