from __future__ import annotations

import pandas as pd

from envoi import ProgressEvent, extract, update_catalog


def test_extract_progress_callback_emits_monotonic_events_when_quiet(
    dem_tif,
    tmp_path,
    capsys,
):
    update_catalog(
        {
            "datasets": {
                "dem_local": {
                    "data_source": "local",
                    "path": str(dem_tif),
                    "bands": 1,
                }
            }
        }
    )
    points = pd.DataFrame(
        {
            "occurrenceID": ["a", "b", "c"],
            "decimalLatitude": [62.976878, 62.981296, 62.976671],
            "decimalLongitude": [18.026823, 18.030991, 18.021154],
            "eventDate": ["2025-06-18", "2020-12-12", "1960-04-02"],
        }
    )
    events: list[ProgressEvent] = []

    extract(
        points,
        {
            "batch_id": "progress",
            "datasets": ["dem_local"],
            "settings": {
                "output_type": "tabular",
                "statistics": ["mean"],
                "window_size_m": 100,
            },
        },
        output_dir=tmp_path,
        quiet=True,
        progress_callback=events.append,
    )

    assert [event.completed for event in events] == [0, 1, 2, 3]
    assert all(event.total == 3 for event in events)
    assert all(event.batch_id == "progress" for event in events)
    assert all(event.dataset == "dem_local" for event in events)
    assert all(event.window_size_m == 100 for event in events)
    assert all(event.mode == "tabular" for event in events)
    assert all(event.unit == "pt" for event in events)

    captured = capsys.readouterr()
    assert "Local stats" not in captured.err
