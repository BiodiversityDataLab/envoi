# src/envoi/catalog_docs.py
"""Render the dataset catalog as human-readable Markdown.

The catalog itself lives in YAML (``configs/ee_catalog.yml``) and is exposed
programmatically through :func:`envoi.list_datasets`, which returns plain
Python lists/dicts. That is convenient for code but hard to read — a raw dump
of 26 nested dicts tells you very little at a glance.

This module turns the same data into a formatted Markdown document: a summary
table per theme, followed by a detail block for every dataset with its
resolution, temporal coverage, bands, licence, and citation.

Two ways to use it:

* ``envoi.catalog_markdown()`` — get the Markdown as a string, e.g. to render
  it inside a Jupyter notebook::

      from IPython.display import Markdown, display
      from envoi import catalog_markdown
      display(Markdown(catalog_markdown()))

* ``python scripts/generate_dataset_docs.py`` — regenerate ``docs/datasets.md``
  in the repository after editing the catalog.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# Datasets whose catalog entry has no `category` key are collected under this
# heading so nothing silently disappears from the generated document.
UNCATEGORISED_LABEL = "Uncategorised"

# Order the theme sections appear in. Categories not listed here (for example
# ones a user added through update_catalog()) are appended afterwards in
# alphabetical order, with "Uncategorised" always last.
CATEGORY_ORDER = [
    "Terrain",
    "Climate",
    "Land cover / land use",
    "Satellite imagery",
    "Vegetation & productivity",
    "Human impact",
    "Other",
]

# Human-readable explanations for the `dataset_spec` options. These are
# internal switches that change how the Earth Engine adapter queries an asset,
# so the generated docs spell out what each one means rather than dumping the
# raw key/value pair on the reader.
DATASET_SPEC_EXPLANATIONS = {
    "native_scale_m": (
        "Native resolution is set manually to {value} m "
        "(the asset does not report a usable scale to Earth Engine)."
    ),
    "use_utm_zone": (
        "Tiled collection: the tile matching each point's UTM zone is selected, "
        "so points near tile edges get the right image."
    ),
    # collection_date_policy is rendered by _explain_date_policy() instead of a
    # single template, so each policy gets a sentence describing only itself
    # rather than listing every alternative on every dataset page.
}

# One sentence per collection_date_policy value. Falls back to a generic line
# for a value the docs don't know about yet.
DATE_POLICY_EXPLANATIONS = {
    "nearest": (
        "Given a sample date, the image with the closest timestamp is used "
        "(dates outside the collection's range are clamped to the nearest end)."
    ),
    "contains": ("Given a sample date, the image whose time interval covers that date is used."),
    "mosaic": (
        "A static product tiled by area rather than a time series: sample dates are "
        "ignored, and the tiles covering each point are mosaicked together."
    ),
}


def _explain_date_policy(value: Any) -> str:
    """Describe one collection_date_policy value for the generated docs."""
    return DATE_POLICY_EXPLANATIONS.get(
        str(value).lower(), f"Image selection from a sample date uses the `{value}` policy."
    )


def _escape_table_cell(value: Any) -> str:
    """Make an arbitrary value safe to drop into a Markdown table cell.

    Pipes would end the cell early and newlines would end the row, so both are
    neutralised. Missing values become an em dash so the column stays aligned.
    """
    if value is None or value == "":
        return "—"
    text = str(value).strip()
    text = text.replace("|", "\\|")
    text = re.sub(r"\s*\n\s*", " ", text)
    return text


def _heading_anchor(heading_text: str) -> str:
    """Build the GitHub-style anchor a Markdown heading gets, for TOC links.

    GitHub lowercases the text, drops punctuation other than hyphens and
    underscores, and replaces spaces with hyphens — e.g. "Land cover / land
    use" becomes "land-cover--land-use".
    """
    anchor = heading_text.strip().lower()
    anchor = re.sub(r"[^\w\s-]", "", anchor)  # keep letters, digits, _, spaces, -
    anchor = anchor.replace(" ", "-")
    return anchor


def _first_sentence(text: str | None) -> str:
    """Shorten a description to its first sentence for the summary tables.

    Descriptions in the catalog are often one long sentence followed by a
    parenthetical band list; the tables only need the opening statement.
    """
    if not text:
        return ""
    # Split on the first sentence-ending period followed by whitespace.
    match = re.search(r"\.\s", text)
    if match:
        return text[: match.start()].strip()
    return text.strip().rstrip(".")


def _format_bands(bands: Any) -> str:
    """Render a `bands` value (string or list) as inline code, comma separated."""
    if bands is None:
        return ""
    if isinstance(bands, str):
        band_names = [bands]
    else:
        band_names = list(bands)
    return ", ".join(f"`{band}`" for band in band_names)


def _group_by_category(
    dataset_records: Sequence[Mapping[str, Any]],
) -> list[tuple[str, list[Mapping[str, Any]]]]:
    """Group dataset records by their `category`, in the documented order.

    Returns a list of (category_name, records) pairs. Records inside each
    category are sorted by display name, which is the label the reader
    actually sees in the tables and headings.
    """
    records_by_category: dict[str, list[Mapping[str, Any]]] = {}
    for record in dataset_records:
        category_name = record.get("category") or UNCATEGORISED_LABEL
        records_by_category.setdefault(str(category_name), []).append(record)

    # Known categories first, in CATEGORY_ORDER; then anything else
    # alphabetically; then the catch-all bucket last.
    ordered_category_names = [name for name in CATEGORY_ORDER if name in records_by_category]
    extra_category_names = sorted(
        name
        for name in records_by_category
        if name not in CATEGORY_ORDER and name != UNCATEGORISED_LABEL
    )
    ordered_category_names += extra_category_names
    if UNCATEGORISED_LABEL in records_by_category:
        ordered_category_names.append(UNCATEGORISED_LABEL)

    return [
        (name, sorted(records_by_category[name], key=lambda record: _display_name(record).lower()))
        for name in ordered_category_names
    ]


def _display_name(record: Mapping[str, Any]) -> str:
    """Return the friendly label for a dataset, falling back to its catalog key.

    Built-in entries carry a `display_name` (usually the dataset's title in the
    Earth Engine catalog, e.g. "Copernicus DEM GLO-30"). Datasets a user
    registers may not, so the key itself is used instead.
    """
    return str(record.get("display_name") or record["name"])


def _unique_anchor(heading_text: str, used_anchor_counts: dict[str, int]) -> str:
    """Allocate the anchor for one heading, disambiguating repeats.

    GitHub appends "-1", "-2", ... when the same heading text occurs more than
    once in a document. Mirroring that here keeps the generated links working
    even if two datasets happen to share a display name.
    """
    base_anchor = _heading_anchor(heading_text)
    occurrence_index = used_anchor_counts.get(base_anchor, 0)
    used_anchor_counts[base_anchor] = occurrence_index + 1
    return base_anchor if occurrence_index == 0 else f"{base_anchor}-{occurrence_index}"


def _allocate_anchors(
    grouped_records: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Work out the anchor for every heading before any Markdown is written.

    Anchors have to be allocated in the order the headings appear in the
    finished document (category heading, then its datasets), because the
    duplicate counter depends on that order. Links are then rendered from the
    resulting maps rather than recomputed, so a link can never disagree with
    the heading it points at.

    Returns:
        (anchors by category name, anchors by dataset name).
    """
    used_anchor_counts: dict[str, int] = {}
    category_anchors: dict[str, str] = {}
    dataset_anchors: dict[str, str] = {}

    for category_name, records in grouped_records:
        category_anchors[category_name] = _unique_anchor(category_name, used_anchor_counts)
        for record in records:
            dataset_anchors[record["name"]] = _unique_anchor(
                _display_name(record), used_anchor_counts
            )

    return category_anchors, dataset_anchors


def _source_label(record: Mapping[str, Any]) -> str:
    """Describe where a dataset comes from, with its asset ID or file path."""
    data_source = record.get("data_source")
    path = record.get("path", "")
    if data_source == "earth_engine":
        return f"Earth Engine — `{path}`"
    if data_source == "local":
        return f"Local raster — `{path}`"
    return f"{_escape_table_cell(data_source)} — `{path}`"


def _render_summary_table(
    dataset_records: Iterable[Mapping[str, Any]],
    dataset_anchors: Mapping[str, str],
) -> list[str]:
    """Build the compact overview table shown at the top of each category."""
    lines = [
        "| Dataset | What it is | Resolution | Temporal | Values |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in dataset_records:
        dataset_information = record.get("dataset_information") or {}
        name = record["name"]
        # First cell: the friendly label linked to the dataset's own detail
        # section, with the catalog key underneath — that key is what users
        # actually pass to extract(), so both belong in the table.
        name_cell = (
            f"[**{_escape_table_cell(_display_name(record))}**](#{dataset_anchors[name]})"
            f"<br>`{name}`"
        )
        description_cell = _escape_table_cell(
            _first_sentence(dataset_information.get("description"))
        )
        resolution_cell = _escape_table_cell(dataset_information.get("spatial_resolution"))
        # Temporal column combines cadence and coverage, e.g. "annual, 2001-2020".
        temporal_parts = [
            str(dataset_information.get(key))
            for key in ("temporal_resolution", "temporal_range")
            if dataset_information.get(key) not in (None, "", "n/a")
        ]
        temporal_cell = _escape_table_cell(
            ", ".join(temporal_parts) if temporal_parts else "static"
        )
        values_cell = _escape_table_cell(record.get("data_type"))
        lines.append(
            f"| {name_cell} | {description_cell} | {resolution_cell} "
            f"| {temporal_cell} | {values_cell} |"
        )
    lines.append("")
    return lines


def _render_dataset_detail(record: Mapping[str, Any]) -> list[str]:
    """Build the detail block for a single dataset."""
    dataset_information = record.get("dataset_information") or {}
    dataset_spec = record.get("dataset_spec") or {}
    lines: list[str] = []

    lines.append(f"#### {_display_name(record)}")
    lines.append("")

    description = dataset_information.get("description")
    if description:
        lines.append(str(description).strip())
        lines.append("")

    # Key facts as a two-column table — easier to scan than a bullet list when
    # every dataset repeats the same fields. The catalog key comes first: it is
    # the string the user has to type into extract().
    fact_rows: list[tuple[str, str]] = [
        ("Use in `extract()`", f"`{record['name']}`"),
        ("Source", _source_label(record)),
    ]

    if record.get("data_type"):
        fact_rows.append(("Values", _escape_table_cell(record["data_type"])))
    if dataset_information.get("spatial_resolution"):
        fact_rows.append(
            ("Spatial resolution", _escape_table_cell(dataset_information["spatial_resolution"]))
        )
    if dataset_information.get("temporal_resolution"):
        fact_rows.append(
            ("Temporal resolution", _escape_table_cell(dataset_information["temporal_resolution"]))
        )
    if dataset_information.get("temporal_range") not in (None, "", "n/a"):
        fact_rows.append(
            ("Temporal coverage", _escape_table_cell(dataset_information["temporal_range"]))
        )

    # Band defaults: which bands are read unless the call overrides them, and
    # which derived bands (slope/aspect) the dataset is allowed to produce.
    if record.get("bands"):
        fact_rows.append(("Default bands", _format_bands(record["bands"])))
    else:
        fact_rows.append(("Default bands", "all bands"))
    if record.get("derived_bands"):
        fact_rows.append(("Derived bands (default)", _format_bands(record["derived_bands"])))
    if record.get("supported_derived_bands"):
        fact_rows.append(
            ("Derived bands available", _format_bands(record["supported_derived_bands"]))
        )

    if dataset_information.get("license"):
        fact_rows.append(("Licence", _escape_table_cell(dataset_information["license"])))

    # Documentation links, collapsed into a single row so the table stays short.
    link_parts = []
    if dataset_information.get("ee_source_url"):
        link_parts.append(f"[Earth Engine catalog]({dataset_information['ee_source_url']})")
    if dataset_information.get("source_url"):
        link_parts.append(f"[Provider documentation]({dataset_information['source_url']})")
    if link_parts:
        fact_rows.append(("Links", " · ".join(link_parts)))

    lines.append("| | |")
    lines.append("| --- | --- |")
    for label, value in fact_rows:
        lines.append(f"| **{label}** | {value} |")
    lines.append("")

    # Translate the adapter switches in `dataset_spec` into plain sentences.
    spec_notes = []
    for spec_key, spec_value in dataset_spec.items():
        # The date policy has one sentence per value, so it gets its own helper
        # rather than a single template shared by every policy.
        if spec_key == "collection_date_policy":
            spec_notes.append(_explain_date_policy(spec_value))
            continue
        explanation = DATASET_SPEC_EXPLANATIONS.get(spec_key)
        if explanation:
            spec_notes.append(explanation.format(value=spec_value))
        else:
            spec_notes.append(f"`{spec_key}`: {spec_value}")
    if spec_notes:
        lines.append("Extraction notes:")
        lines.append("")
        for note in spec_notes:
            lines.append(f"- {note}")
        lines.append("")

    citation = dataset_information.get("citation")
    if citation:
        lines.append(f"> **Cite as:** {str(citation).strip()}")
        lines.append("")

    return lines


def render_catalog_markdown(
    dataset_records: Sequence[Mapping[str, Any]],
    *,
    title: str = "Built-in datasets",
    generated_note: str | None = None,
) -> str:
    """Render dataset catalog records as a Markdown document.

    Args:
        dataset_records: Catalog entries in the shape returned by
            ``list_datasets("full")`` — one dict per dataset, each with a
            ``name`` key plus the catalog fields.
        title: Level-1 heading for the document.
        generated_note: Optional italic note placed under the title, used by
            the docs generator to say the file is auto-generated.

    Returns:
        The complete Markdown document as a single string.
    """
    grouped_records = _group_by_category(dataset_records)
    # Anchors are allocated up front, in heading order, so the contents list
    # and the summary tables link to exactly the headings that get written.
    category_anchors, dataset_anchors = _allocate_anchors(grouped_records)

    lines: list[str] = [f"# {title}", ""]

    if generated_note:
        lines += [f"*{generated_note}*", ""]

    # Intro: what the list is and how to use a name from it.
    lines += [
        (
            f"envoi ships with **{len(dataset_records)} datasets** ready to use — no downloads, "
            "no configuration. Each entry below shows the dataset's name followed by the "
            "catalog key in `code font` — that key is what you pass to `extract()`:"
        ),
        "",
        "```python",
        "from envoi import extract",
        "",
        "extract(points_dataframe, {",
        '    "batch_id": "terrain",',
        '    "datasets": ["dem_copernicus_glo30"],',
        '    "settings": {"statistics": ["mean"], "window_size_m": 200},',
        "})",
        "```",
        "",
        (
            "To add your own local rasters or Earth Engine assets, register them with "
            "`update_catalog()` — see [advanced_usage.md](advanced_usage.md)."
        ),
        "",
    ]

    # Table of contents — one line per category with its dataset count.
    lines += ["## Contents", ""]
    for category_name, records in grouped_records:
        lines.append(
            f"- [{category_name}](#{category_anchors[category_name]}) "
            f"({len(records)} dataset{'s' if len(records) != 1 else ''})"
        )
    lines.append("")

    # One section per category: summary table first, then the detail blocks.
    for category_name, records in grouped_records:
        lines += [f"## {category_name}", ""]
        lines += _render_summary_table(records, dataset_anchors)
        for record in records:
            lines += _render_dataset_detail(record)

    # Trailing pointer to the machine-readable source of truth.
    lines += [
        "---",
        "",
        (
            "The catalog source, with every field for every entry, is "
            "[`src/envoi/configs/ee_catalog.yml`](../src/envoi/configs/ee_catalog.yml). "
            'The same information is available programmatically via `list_datasets("full")`.'
        ),
        "",
    ]

    return "\n".join(lines)


def catalog_markdown(
    dataset_records: Sequence[Mapping[str, Any]] | None = None,
    *,
    title: str = "Built-in datasets",
) -> str:
    """Return the current dataset catalog rendered as Markdown.

    Args:
        dataset_records: Catalog entries to render. Defaults to everything
            ``list_datasets("full")`` sees — the built-in catalog plus any
            datasets registered with ``update_catalog()``.
        title: Level-1 heading for the document.

    Returns:
        A Markdown string. Print it, write it to a ``.md`` file, or render it
        in a notebook with ``IPython.display.Markdown``.

    Example:
        >>> from envoi import catalog_markdown
        >>> print(catalog_markdown())
    """
    if dataset_records is None:
        # Imported here rather than at module scope to avoid a circular import
        # (catalog.py is imported by envoi/__init__.py before this module).
        from .catalog import list_datasets

        dataset_records = list_datasets("full")

    return render_catalog_markdown(dataset_records, title=title)
