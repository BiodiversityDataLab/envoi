from __future__ import annotations

from envoi_webapp.app import _apply_pending_dataset_remove, _escape_applescript_string


class _FakeStreamlit:
    def __init__(self, session_state):
        self.session_state = session_state


class _SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _has_dataset_widget_keys(session_state) -> bool:
    return any(
        key.startswith("dataset_select_")
        or key.startswith("windows_input_")
        or key.startswith("stats_select_")
        or key.startswith("remove_dataset_button_")
        or key.startswith("windows_")
        or key.startswith("stats_")
        or (key.startswith("dataset_") and key[8:].isdigit())
        for key in session_state
    )


def test_escape_applescript_string_escapes_quotes_and_backslashes():
    assert _escape_applescript_string('/tmp/a "quoted" folder\\name') == (
        '/tmp/a \\"quoted\\" folder\\\\name'
    )


def test_apply_pending_dataset_remove_removes_only_requested_row_and_clears_widget_cache():
    session_state = _SessionState(
        {
            "dataset_rows": [
                {"dataset": "dem", "window_sizes": "100", "statistics": ["mean"]},
                {"dataset": "agb", "window_sizes": "200", "statistics": ["mean"]},
                {"dataset": "lulc", "window_sizes": "300", "statistics": ["mode"]},
            ],
            "_pending_dataset_remove": 1,
            "_dataset_widget_version": 4,
            "dataset_select_4_0": "dem",
            "dataset_select_4_1": "agb",
            "dataset_select_4_2": "lulc",
            "windows_input_4_1": "200",
            "stats_select_4_1_agb": ["mean"],
            "remove_dataset_button_4_1": True,
            "dataset_0": "dem",
            "dataset_1": "agb",
            "dataset_2": "lulc",
            "windows_1": "200",
            "stats_1_agb": ["mean"],
        }
    )

    _apply_pending_dataset_remove(_FakeStreamlit(session_state))

    assert session_state["dataset_rows"] == [
        {"dataset": "dem", "window_sizes": "100", "statistics": ["mean"]},
        {"dataset": "lulc", "window_sizes": "300", "statistics": ["mode"]},
    ]
    assert "_pending_dataset_remove" not in session_state
    assert session_state["_dataset_widget_version"] == 5
    assert not _has_dataset_widget_keys(session_state)


def test_apply_pending_dataset_remove_removes_first_row_from_two_rows():
    session_state = _SessionState(
        {
            "dataset_rows": [
                {"dataset": "dem", "window_sizes": "100", "statistics": ["mean"]},
                {"dataset": "lulc", "window_sizes": "300", "statistics": ["mode"]},
            ],
            "_pending_dataset_remove": 0,
            "_dataset_widget_version": 2,
            "dataset_select_2_0": "dem",
            "dataset_select_2_1": "lulc",
            "windows_input_2_0": "100",
            "windows_input_2_1": "300",
            "stats_select_2_0_dem": ["mean"],
            "stats_select_2_1_lulc": ["mode"],
        }
    )

    _apply_pending_dataset_remove(_FakeStreamlit(session_state))

    assert session_state["dataset_rows"] == [
        {"dataset": "lulc", "window_sizes": "300", "statistics": ["mode"]},
    ]
    assert "_pending_dataset_remove" not in session_state
    assert session_state["_dataset_widget_version"] == 3
    assert not _has_dataset_widget_keys(session_state)


def test_apply_pending_dataset_remove_clears_final_row_and_widget_cache():
    session_state = _SessionState(
        {
            "dataset_rows": [
                {"dataset": "agb", "window_sizes": "200", "statistics": ["mean"]},
            ],
            "_pending_dataset_remove": 0,
            "_dataset_widget_version": 7,
            "dataset_select_7_0": "agb",
            "windows_input_7_0": "200",
            "stats_select_7_0_agb": ["mean"],
            "remove_dataset_button_7_0": True,
            "dataset_0": "agb",
            "windows_0": "200",
            "stats_0_agb": ["mean"],
        }
    )

    _apply_pending_dataset_remove(_FakeStreamlit(session_state))

    assert session_state["dataset_rows"] == [{"dataset": "", "window_sizes": "", "statistics": []}]
    assert "_pending_dataset_remove" not in session_state
    assert session_state["_dataset_widget_version"] == 8
    assert not _has_dataset_widget_keys(session_state)
