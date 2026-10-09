"""Smoke check of the hosted envoi web app. It exits with 1 when a check fails.

In the container image (the working folder is /app):

    docker run --rm --entrypoint python <image> smoke_check.py

Locally, from deploy/serve/:

    ENVOI_WEBAPP_MODE=hosted python smoke_check.py

Checks (they need no network access):

1. The page script ``app.py`` in this folder runs once in hosted mode, with
   Streamlit's ``AppTest``. The page must show no exception and no error, must
   show the hosted section 3, and must not call ``ee.Initialize()`` in the
   server process.
2. The real ``JobManager`` starts a real worker process for a job with a
   generated service-account key without ``token_uri``. The job must end as
   FAILED with the "not a valid key" error. The error message and its run-log
   lines must contain no key material, and the job workspace must be deleted.

The check puts the job workspaces in a new temporary folder, in the folder that
``ENVOI_WEBAPP_WORKSPACE`` names, or in the system temporary folder when the
variable is not set. The start-up clean-up of the job manager then cannot
delete the workspaces of a running server. The check removes the temporary
folder at the end.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import sys
import tempfile
import time
import urllib.parse
from pathlib import Path

import ee
import pandas as pd
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from streamlit.testing.v1 import AppTest

from envoi_webapp.helpers import TABULAR_OUTPUT, DatasetSelection, build_run_config
from envoi_webapp.job_protocol import JobState
from envoi_webapp.jobs import JobManager
from envoi_webapp.settings import (
    WORKSPACE_SUBFOLDER_NAME,
    WORKSPACE_VARIABLE,
    WebappSettings,
    load_settings,
)

APP_SCRIPT_PATH = Path(__file__).resolve().parent / "app.py"
# The first run of the page imports pandas, geopandas, and ee, which can take
# long in a new container.
APP_TIMEOUT_S = 60
# The worker process imports the same libraries before it checks the key.
JOB_TIMEOUT_S = 120
JOB_POLL_INTERVAL_S = 0.5

# Section 3 has this title only in hosted mode.
HOSTED_SECTION_TITLE = "3. Results and your data"
LOCAL_OUTPUT_FIELD_LABEL = "Output directory"
# Part of the error of init_gee() for a key that is not valid. It tells the
# key error apart from a worker that crashed before it read the key.
INVALID_KEY_TEXT = "not a valid Google service account key"


def check_app_script() -> list[str]:
    """Run ``app.py`` once with ``AppTest`` in hosted mode. Return the problems found."""

    # The Streamlit server process must never initialize Earth Engine (spec R1).
    # A call shows up as an exception on the page.
    def refuse_initialize(*args, **kwargs):
        raise AssertionError("The Streamlit server process called ee.Initialize().")

    ee.Initialize = refuse_initialize
    app_test = AppTest.from_file(str(APP_SCRIPT_PATH), default_timeout=APP_TIMEOUT_S).run()

    problems = [
        f"The page shows an exception: {exception.value}" for exception in app_test.exception
    ]
    problems.extend(f"The page shows an error: {error.value}" for error in app_test.error)
    if HOSTED_SECTION_TITLE not in [subheader.value for subheader in app_test.subheader]:
        problems.append(f"The page has no section {HOSTED_SECTION_TITLE!r}, so it is not hosted.")
    if any(text_input.label == LOCAL_OUTPUT_FIELD_LABEL for text_input in app_test.text_input):
        problems.append("The page shows the output-folder field, which hosted mode must hide.")
    return problems


def check_job_with_invalid_key(settings: WebappSettings) -> list[str]:
    """Run one job with a key that is not valid through the real job manager.

    Return the problems found. The message of a problem never contains the
    error message when that message contains key material.
    """
    key_text = generated_key_without_token_uri()
    points = pd.DataFrame(
        {
            "occurrenceID": ["smoke-check-1"],
            "decimalLatitude": [59.858],
            "decimalLongitude": [17.639],
        }
    )
    run_configs = build_run_config(
        [
            DatasetSelection(
                dataset="dem_copernicus_glo30",
                output_type=TABULAR_OUTPUT,
                window_sizes=(0,),
                statistics=("mean",),
            )
        ]
    )

    # Submit the job and wait for its final state. shutdown() stops the worker
    # if it is still running, and deletes the workspaces that are due.
    manager = JobManager(settings)
    try:
        job_id = manager.submit(
            session_token="smoke-check",
            points=points,
            run_configs=run_configs,
            input_crs="EPSG:4326",
            credentials_json=key_text,
        )
        deadline = time.monotonic() + JOB_TIMEOUT_S
        snapshot = manager.snapshot(job_id)
        while snapshot is not None and not snapshot.state.is_final:
            if time.monotonic() > deadline:
                break
            time.sleep(JOB_POLL_INTERVAL_S)
            snapshot = manager.snapshot(job_id)
    finally:
        manager.shutdown()

    # The final state and the error message.
    if snapshot is None:
        return ["The job manager does not know the job."]
    if snapshot.state is not JobState.FAILED:
        if snapshot.state is JobState.RUNNING:
            return [f"The job did not end within {JOB_TIMEOUT_S} s."]
        return [f"The job ended as {snapshot.state.value}, not as failed."]

    problems: list[str] = []
    error_texts = [snapshot.error["message"], *snapshot.error["run_log_tail"]]
    secret_parts = key_material(key_text)
    leaked_parts = [part for part in secret_parts if any(part in text for text in error_texts)]
    if leaked_parts:
        problems.append(
            "The error message or its run-log lines contain key material "
            f"({len(leaked_parts)} of {len(secret_parts)} key parts)."
        )
    elif INVALID_KEY_TEXT not in snapshot.error["message"]:
        problems.append(f"The job failed for another reason: {snapshot.error['message']}")

    # A failed hosted job's workspace is deleted when the job ends (spec R9).
    workspace_root = settings.workspace_parent / WORKSPACE_SUBFOLDER_NAME
    leftover_names = sorted(path.name for path in workspace_root.iterdir())
    if leftover_names:
        problems.append(f"The workspace root still holds: {', '.join(leftover_names)}.")
    return problems


def generated_key_without_token_uri() -> str:
    """Return a new service-account key as JSON text, without ``token_uri``.

    The key has a real 2048-bit RSA private key in PKCS#8 PEM, as Google's keys
    do. Without ``token_uri``, the Earth Engine library gives an error whose
    text contains the whole key, before any network call.
    """
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    client_email = "envoi-smoke-check@envoi-smoke-check.iam.gserviceaccount.com"
    key = {
        "type": "service_account",
        "project_id": "envoi-smoke-check",
        "private_key_id": secrets.token_hex(20),
        "private_key": private_key_pem,
        "client_email": client_email,
        "client_id": "100000000000000000000",
        "client_x509_cert_url": (
            "https://www.googleapis.com/robot/v1/metadata/x509/"
            + urllib.parse.quote(client_email, safe="")
        ),
    }
    return json.dumps(key, indent=2)


def key_material(key_text: str) -> list[str]:
    """Return each part of a key that no message may contain.

    The parts are the full key text, the private key, each body line of the
    private key, the private key ID, the client email (also URL-encoded), and
    the certificate URL.
    """
    key = json.loads(key_text)
    private_key = key["private_key"]
    material = [
        key_text,
        private_key,
        key["private_key_id"],
        key["client_email"],
        urllib.parse.quote(key["client_email"], safe=""),
        key["client_x509_cert_url"],
    ]
    material.extend(
        line.strip()
        for line in private_key.splitlines()
        if line.strip() and not line.startswith("-----")
    )
    return material


def main() -> int:
    """Run the checks, print the result of each, and return the exit code."""
    # A new workspace parent for this check only, in the configured folder or
    # in the system temporary folder.
    configured_parent = os.environ.get(WORKSPACE_VARIABLE, "").strip() or None
    temporary_parent = tempfile.mkdtemp(prefix="envoi-smoke-check-", dir=configured_parent)
    os.environ[WORKSPACE_VARIABLE] = temporary_parent

    try:
        # The image sets ENVOI_WEBAPP_MODE=hosted. The check fails without it.
        settings = load_settings()
        if not settings.is_hosted:
            print("FAIL: ENVOI_WEBAPP_MODE is not 'hosted'.")
            return 1

        failed = False
        checks = [
            ("app script in hosted mode", check_app_script),
            ("job with a key that is not valid", lambda: check_job_with_invalid_key(settings)),
        ]
        for check_name, check in checks:
            problems = check()
            if problems:
                failed = True
                print(f"FAIL: {check_name}")
                for problem in problems:
                    print(f"  - {problem}")
            else:
                print(f"PASS: {check_name}")
        return 1 if failed else 0
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
