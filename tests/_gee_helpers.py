"""Shared helpers for GEE-backed tests.

Kept as a private module (leading underscore) so it's clear at a glance
that it is test scaffolding, not part of the public envoi API.

The two responsibilities here:
  * ``gee_credentials_available()`` — cheap, network-free check used by
    the ``pytestmark = skipif(...)`` line at the top of each GEE test
    file. Running this at collection time avoids triggering a network
    call (and the auth side effect) just to enumerate the test list.
  * ``SWEDEN_SAMPLE_DF`` — the two-point fixture used by most GEE tests.
    Centralising it removes the identical copy-pasted DataFrame that
    used to live at the top of every GEE test module.
"""

from __future__ import annotations

import pandas as pd

from envoi.auth import _default_credentials_path


def gee_credentials_available() -> bool:
    """Return True when a GEE service account JSON is findable locally.

    Mirrors the lookup order in :func:`envoi.auth.init_gee` but stops after
    locating the file — it never calls ``ee.Initialize`` or talks to GEE.
    That way ``pytest --collect-only`` does no network I/O even when the
    GEE test files are part of the collection.
    """
    credentials_path = _default_credentials_path()
    return credentials_path is not None and credentials_path.exists()


# Two-point sample DataFrame used by most GEE tests. Northern Sweden was
# picked because every global dataset has coverage there and the points sit
# close enough together that GEE caches the per-image fetch across them.
SWEDEN_SAMPLE_DF = pd.DataFrame(
    {
        "occurrenceID": ["A", "B"],
        "decimalLatitude": [62.9768783, 62.9812956],
        "decimalLongitude": [18.026823, 18.0309905],
        "eventDate": ["2020-06-01", "2020-06-01"],
    }
)
