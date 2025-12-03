# auth.py
import ee
from pathlib import Path

KEY_JSON = Path("secrets/gge1-473507-9efd9dfa4563.json")
SERVICE_ACCOUNT = "thantzin@gge1-473507.iam.gserviceaccount.com"  # fill in
PROJECT_ID = "gge1-473507"  # optional but recommended


def init_gee():
    """
    Initialize Earth Engine using the service account JSON key
    already in secrets/.
    """
    if not KEY_JSON.exists():
        raise FileNotFoundError(f"Service account key not found: {KEY_JSON}")

    creds = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, str(KEY_JSON))
    ee.Initialize(creds, project=PROJECT_ID)
