from __future__ import annotations

import json
import traceback

import ee
import pytest

from envoi.auth import ENV_VAR, init_gee

# Static parts of a fake key. The body lines take the place of the base64 lines
# of a real private key. The text between BEGIN and END must stay shorter than
# 64 characters. Otherwise the gitleaks pre-commit hook reports it as a key.
_FAKE_PRIVATE_KEY_LINES = ("FAKEKEYLINE1", "FAKEKEYLINE2")
_FAKE_PRIVATE_KEY = (
    "-----BEGIN PRIVATE KEY-----\nFAKEKEYLINE1\nFAKEKEYLINE2\n-----END PRIVATE KEY-----\n"
)
_FAKE_PRIVATE_KEY_ID = "fake-private-key-id-for-leak-test"
_FAKE_CLIENT_EMAIL = "leak-test@fake-project.iam.gserviceaccount.com"

# Returned by the fake ee.ServiceAccountCredentials, so a test can check that
# init_gee() gives the same object to ee.Initialize().
_FAKE_CREDENTIALS = object()


def _fake_key(**changes) -> dict:
    key = {
        "type": "service_account",
        "project_id": "fake-project",
        "private_key_id": _FAKE_PRIVATE_KEY_ID,
        "private_key": _FAKE_PRIVATE_KEY,
        "client_email": _FAKE_CLIENT_EMAIL,
        "client_id": "123456789",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    key.update(changes)
    return key


def _key_without_token_uri(private_key: str) -> dict:
    key = _fake_key(private_key=private_key)
    del key["token_uri"]
    return key


def _key_material(key_text: str) -> list[str]:
    """Return each part of a key that an error must not contain.

    The parts are the full key text, the private key, each body line of the
    private key, the private key ID, and the client email.
    """
    material = [key_text]
    try:
        key = json.loads(key_text)
    except ValueError:
        return material
    private_key = key["private_key"]
    material.extend([private_key, key["private_key_id"], key["client_email"]])
    material.extend(
        line.strip()
        for line in private_key.splitlines()
        if line.strip() and not line.startswith("-----")
    )
    return material


@pytest.fixture(scope="module")
def generated_private_key() -> str:
    """A new 2048-bit RSA private key in PKCS#8 PEM, the format of Google's keys."""
    rsa = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.rsa")
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")


@pytest.fixture
def fake_ee(monkeypatch) -> dict[str, list]:
    """Replace ee.ServiceAccountCredentials and ee.Initialize with recorders."""
    calls: dict[str, list] = {"credentials": [], "initialize": []}

    def fake_service_account_credentials(email=None, key_file=None, key_data=None):
        calls["credentials"].append({"email": email, "key_file": key_file, "key_data": key_data})
        return _FAKE_CREDENTIALS

    monkeypatch.setattr(ee, "ServiceAccountCredentials", fake_service_account_credentials)
    monkeypatch.setattr(ee, "Initialize", calls["initialize"].append)
    return calls


def _refuse_initialize(credentials):
    raise ee.EEException("Not signed up for Earth Engine.")


class TestCredentialsJson:
    @pytest.mark.parametrize("key_form", ["str", "bytes", "dict"])
    def test_each_form_reaches_key_data(self, fake_ee, key_form):
        """Text, bytes, and a dict all reach ee.ServiceAccountCredentials as JSON text."""
        key = _fake_key()
        key_text = json.dumps(key)
        credentials_json = {"str": key_text, "bytes": key_text.encode("utf-8"), "dict": key}[
            key_form
        ]

        init_gee(credentials_json=credentials_json)

        [credentials_call] = fake_ee["credentials"]
        assert credentials_call["email"] is None
        assert credentials_call["key_file"] is None
        assert isinstance(credentials_call["key_data"], str)
        assert json.loads(credentials_call["key_data"]) == key
        assert fake_ee["initialize"] == [_FAKE_CREDENTIALS]

    def test_key_content_skips_file_lookup(self, fake_ee, monkeypatch, tmp_path):
        """With credentials_json, init_gee() does not look for a key file."""
        monkeypatch.setenv(ENV_VAR, str(tmp_path / "missing.json"))

        init_gee(credentials_json=json.dumps(_fake_key()))

        assert [call["key_file"] for call in fake_ee["credentials"]] == [None]

    def test_generated_key_builds_real_credentials(self, monkeypatch, generated_private_key):
        """A complete key with a real private key gives credentials for its service account."""
        initialized = []
        monkeypatch.setattr(ee, "Initialize", initialized.append)

        init_gee(credentials_json=json.dumps(_fake_key(private_key=generated_private_key)))

        [credentials] = initialized
        assert credentials.service_account_email == _FAKE_CLIENT_EMAIL

    def test_path_and_content_together_raise(self, fake_ee, tmp_path):
        """Giving both a path and key content raises before any credentials are built."""
        key_path = tmp_path / "key.json"
        key_path.write_text(json.dumps(_fake_key()), encoding="utf-8")

        with pytest.raises(ValueError, match="not both"):
            init_gee(key_path, credentials_json=json.dumps(_fake_key()))

        assert fake_ee["credentials"] == []
        assert fake_ee["initialize"] == []

    @pytest.mark.parametrize("credentials_json", [123, ["not", "a", "key"]])
    def test_wrong_type_raises(self, fake_ee, credentials_json):
        """A value that is not text, bytes, or a mapping raises ValueError."""
        with pytest.raises(ValueError, match="credentials_json must be the content"):
            init_gee(credentials_json=credentials_json)

        assert fake_ee["credentials"] == []

    def test_initialize_failure_names_the_supplied_key(self, fake_ee, monkeypatch):
        """An Earth Engine refusal names the supplied key and keeps the original error text."""
        monkeypatch.setattr(ee, "Initialize", _refuse_initialize)
        key_text = json.dumps(_fake_key())

        with pytest.raises(RuntimeError, match="failed using the supplied key") as exc_info:
            init_gee(credentials_json=key_text)

        message = str(exc_info.value)
        assert "Check that the service account in the key is valid" in message
        assert "Original error: Not signed up for Earth Engine." in message
        for secret in _key_material(key_text):
            assert secret not in message


class TestInvalidKeyErrorsContainNoKeyMaterial:
    """These tests use the real ee.ServiceAccountCredentials. They need no network."""

    @pytest.fixture(autouse=True)
    def _initialize_must_not_run(self, monkeypatch):
        def fail(credentials):
            raise AssertionError("ee.Initialize must not run for an invalid key.")

        monkeypatch.setattr(ee, "Initialize", fail)

    @pytest.mark.parametrize(
        "key_case",
        [
            "generated_key_without_token_uri",
            "generated_key_truncated",
            "garbage_private_key",
            "not_json",
        ],
    )
    def test_error_contains_no_key_material(self, generated_private_key, key_case):
        """The ValueError, its cause, its context, and its traceback contain no key material."""
        # Build the invalid key, in each of the three accepted forms.
        if key_case == "generated_key_without_token_uri":
            key_text = json.dumps(_key_without_token_uri(generated_private_key))
            credentials_json = key_text
        elif key_case == "generated_key_truncated":
            truncated_private_key = generated_private_key[: len(generated_private_key) // 2]
            key_text = json.dumps(_fake_key(private_key=truncated_private_key))
            credentials_json = key_text.encode("utf-8")
        elif key_case == "garbage_private_key":
            key = _fake_key()
            key_text = json.dumps(key)
            credentials_json = key
        else:
            key_text = f"private key {_FAKE_PRIVATE_KEY_LINES[0]} for {_FAKE_CLIENT_EMAIL}"
            credentials_json = key_text

        with pytest.raises(ValueError, match="not a valid Google service account key") as exc_info:
            init_gee(credentials_json=credentials_json)

        error = exc_info.value
        assert error.__cause__ is None
        assert error.__context__ is None
        error_texts = [
            str(error),
            repr(error),
            repr(error.__cause__),
            repr(error.__context__),
            "".join(traceback.format_exception(error)),
        ]
        secrets = {
            *_key_material(key_text),
            *_FAKE_PRIVATE_KEY_LINES,
            _FAKE_PRIVATE_KEY_ID,
            _FAKE_CLIENT_EMAIL,
        }
        for secret in secrets:
            for error_text in error_texts:
                assert secret not in error_text


class TestCredentialsPath:
    def test_path_reaches_key_file(self, fake_ee, tmp_path):
        """An explicit path goes to ee.ServiceAccountCredentials as key_file."""
        key_path = tmp_path / "key.json"
        key_path.write_text(json.dumps(_fake_key()), encoding="utf-8")

        init_gee(key_path)

        assert fake_ee["credentials"] == [
            {"email": None, "key_file": str(key_path), "key_data": None}
        ]
        assert fake_ee["initialize"] == [_FAKE_CREDENTIALS]

    def test_no_argument_uses_environment_variable(self, fake_ee, monkeypatch, tmp_path):
        """Without arguments, init_gee() uses the file that the environment variable names."""
        key_path = tmp_path / "key.json"
        key_path.write_text(json.dumps(_fake_key()), encoding="utf-8")
        monkeypatch.setenv(ENV_VAR, str(key_path))

        init_gee()

        assert [call["key_file"] for call in fake_ee["credentials"]] == [str(key_path)]

    def test_missing_file_raises_file_not_found(self, fake_ee, tmp_path):
        """A path to a file that does not exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="credentials not found"):
            init_gee(tmp_path / "missing.json")

        assert fake_ee["credentials"] == []

    def test_initialize_failure_names_the_path(self, fake_ee, monkeypatch, tmp_path):
        """An Earth Engine refusal gives the same message as before, with the file path."""
        key_path = tmp_path / "key.json"
        key_path.write_text(json.dumps(_fake_key()), encoding="utf-8")
        monkeypatch.setattr(ee, "Initialize", _refuse_initialize)

        with pytest.raises(RuntimeError) as exc_info:
            init_gee(key_path)

        assert str(exc_info.value) == (
            f"Google Earth Engine authentication failed using '{key_path}'.\n"
            "Check that the service account in the file is valid and has access to GEE.\n"
            "Original error: Not signed up for Earth Engine."
        )
        assert isinstance(exc_info.value.__cause__, ee.EEException)
