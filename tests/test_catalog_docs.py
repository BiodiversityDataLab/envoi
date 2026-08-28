"""Unit tests for envoi.catalog_docs — the Markdown rendering of the catalog.

Two things are checked here: that the renderer produces well-formed Markdown
from arbitrary catalog records, and that the committed ``docs/datasets.md``
still matches what the current built-in catalog would generate (so the docs
cannot silently drift when a dataset is added or edited).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from envoi import catalog_markdown, list_datasets
from envoi.catalog import BUILTIN_EE_CATALOG, load_catalogs
from envoi.catalog_docs import UNCATEGORISED_LABEL, render_catalog_markdown

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DOCS_PATH = REPOSITORY_ROOT / "docs" / "datasets.md"
GENERATOR_SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "generate_dataset_docs.py"


def _load_generator_script():
    """Import scripts/generate_dataset_docs.py as a module.

    The script lives outside the package (it is a repo tool, not shipped
    code), so it is loaded by path rather than by import name.
    """
    spec = importlib.util.spec_from_file_location("generate_dataset_docs", GENERATOR_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# A minimal, fully-controlled catalog used by the rendering tests, so they do
# not break every time a real dataset is added to the bundled catalog.
EXAMPLE_RECORDS = [
    {
        "name": "example_dem",
        "display_name": "Example DEM 30 m",
        "data_source": "earth_engine",
        "path": "EXAMPLE/DEM",
        "data_type": "continuous",
        "category": "Terrain",
        "bands": ["DEM"],
        "supported_derived_bands": ["slope", "aspect"],
        "dataset_spec": {"native_scale_m": 30, "use_utm_zone": True},
        "dataset_information": {
            "description": "An example DEM. Second sentence with a | pipe character.",
            "spatial_resolution": "30 meters",
            "temporal_resolution": "static",
            "temporal_range": "n/a",
            "ee_source_url": "https://example.org/ee",
            "source_url": "https://example.org/docs",
            "citation": "Example et al. (2024).",
            "license": "CC-BY-4.0",
        },
    },
    {
        "name": "example_local_raster",
        "data_source": "local",
        "path": "/data/example.tif",
    },
]


class TestRenderCatalogMarkdown:
    """The renderer turns catalog records into a readable Markdown document."""

    def test_includes_every_dataset_name(self):
        # Nothing may be dropped: each record must appear as its own heading.
        # Headings use the friendly label, so the reader sees "Example DEM
        # 30 m" rather than the pythonic catalog key.
        markdown_text = render_catalog_markdown(EXAMPLE_RECORDS)
        assert "#### Example DEM 30 m" in markdown_text
        # A record without a display_name falls back to its catalog key.
        assert "#### example_local_raster" in markdown_text

    def test_shows_the_catalog_key_alongside_the_display_name(self):
        # The display name is for reading; the catalog key is what the user
        # has to type into extract(). Both must be present — in the summary
        # table row and in the dataset's fact table.
        markdown_text = render_catalog_markdown(EXAMPLE_RECORDS)
        assert "<br>`example_dem`" in markdown_text
        assert "| **Use in `extract()`** | `example_dem` |" in markdown_text

    def test_duplicate_display_names_get_distinct_anchors(self):
        # Two datasets sharing a label must still be individually linkable —
        # GitHub disambiguates repeated headings by appending "-1", "-2", ...
        records = [
            {"name": "first_copy", "display_name": "Same Label", "data_source": "local"},
            {"name": "second_copy", "display_name": "Same Label", "data_source": "local"},
        ]
        markdown_text = render_catalog_markdown(records)
        assert "(#same-label)" in markdown_text
        assert "(#same-label-1)" in markdown_text

    def test_groups_by_category_and_buckets_missing_ones(self):
        # Records with a category get that section heading; records without
        # one fall into the catch-all bucket instead of vanishing.
        markdown_text = render_catalog_markdown(EXAMPLE_RECORDS)
        assert "## Terrain" in markdown_text
        assert f"## {UNCATEGORISED_LABEL}" in markdown_text

    def test_table_of_contents_links_resolve_to_headings(self):
        # The contents list uses GitHub-style anchors derived from the
        # heading text — "Land cover / land use" -> "#land-cover--land-use".
        records = [{**EXAMPLE_RECORDS[0], "category": "Land cover / land use"}]
        markdown_text = render_catalog_markdown(records)
        assert "[Land cover / land use](#land-cover--land-use)" in markdown_text
        assert "## Land cover / land use" in markdown_text

    def test_escapes_pipes_in_table_cells(self):
        # A pipe inside a description would break the table layout, so the
        # summary table must escape it. (The description used here puts the
        # pipe in its second sentence, which the summary trims — check the
        # escaping helper via a description that is a single sentence.)
        records = [
            {
                **EXAMPLE_RECORDS[0],
                "dataset_information": {
                    **EXAMPLE_RECORDS[0]["dataset_information"],
                    "description": "Bands a | b are included",
                },
            }
        ]
        markdown_text = render_catalog_markdown(records)
        summary_row = next(
            line
            for line in markdown_text.splitlines()
            if "`example_dem`" in line and "<br>" in line
        )
        assert "a \\| b" in summary_row
        # Escaped pipes aside, the row must still have the expected column count.
        assert summary_row.count("|") - summary_row.count("\\|") == 6

    def test_explains_dataset_spec_options_in_plain_language(self):
        # dataset_spec keys are internal adapter switches; the docs should
        # describe what they do rather than print the raw key/value pair.
        markdown_text = render_catalog_markdown(EXAMPLE_RECORDS)
        assert "Native resolution is set manually to 30 m" in markdown_text
        assert "UTM zone" in markdown_text

    def test_renders_citation_and_links(self):
        markdown_text = render_catalog_markdown(EXAMPLE_RECORDS)
        assert "> **Cite as:** Example et al. (2024)." in markdown_text
        assert "[Earth Engine catalog](https://example.org/ee)" in markdown_text
        assert "[Provider documentation](https://example.org/docs)" in markdown_text

    def test_local_datasets_show_their_file_path(self):
        markdown_text = render_catalog_markdown(EXAMPLE_RECORDS)
        assert "Local raster — `/data/example.tif`" in markdown_text

    def test_generated_note_is_optional(self):
        without_note = render_catalog_markdown(EXAMPLE_RECORDS)
        with_note = render_catalog_markdown(EXAMPLE_RECORDS, generated_note="Auto-generated.")
        assert "*Auto-generated.*" in with_note
        assert "*Auto-generated.*" not in without_note


class TestCatalogMarkdownPublicFunction:
    """envoi.catalog_markdown() renders whatever list_datasets() sees."""

    def test_defaults_to_the_live_catalog(self):
        markdown_text = catalog_markdown()
        assert markdown_text.startswith("# Built-in datasets")
        # Every dataset currently registered should be documented, listed by
        # the catalog key the user passes to extract().
        for name in list_datasets("names"):
            assert f"| **Use in `extract()`** | `{name}` |" in markdown_text


class TestBuiltinCatalogLabels:
    """Every bundled dataset should carry its documentation/menu labels."""

    def test_all_builtin_datasets_have_a_category(self):
        datasets = load_catalogs(BUILTIN_EE_CATALOG)["datasets"]
        missing_category = sorted(
            name for name, entry in datasets.items() if not entry.get("category")
        )
        assert not missing_category, (
            "Built-in datasets without a `category` key (add one in "
            f"configs/ee_catalog.yml): {missing_category}"
        )

    def test_all_builtin_datasets_have_a_display_name(self):
        # The web app's dataset menu and the generated docs both show
        # display_name, so a built-in entry without one would surface its raw
        # pythonic key to users.
        datasets = load_catalogs(BUILTIN_EE_CATALOG)["datasets"]
        missing_display_name = sorted(
            name for name, entry in datasets.items() if not entry.get("display_name")
        )
        assert not missing_display_name, (
            "Built-in datasets without a `display_name` key (add one in "
            f"configs/ee_catalog.yml): {missing_display_name}"
        )

    def test_display_names_are_unique(self):
        # Duplicated labels would be ambiguous in a dropdown menu, where the
        # catalog key is not shown.
        datasets = load_catalogs(BUILTIN_EE_CATALOG)["datasets"]
        display_names = [entry.get("display_name") for entry in datasets.values()]
        duplicated = sorted(
            {label for label in display_names if display_names.count(label) > 1 and label}
        )
        assert not duplicated, f"Duplicate display_name values in the catalog: {duplicated}"


class TestGeneratedDocsAreUpToDate:
    """The committed docs/datasets.md must match the current catalog."""

    def test_docs_file_matches_generator_output(self):
        if not GENERATED_DOCS_PATH.exists():
            pytest.fail(
                "docs/datasets.md is missing — run: python scripts/generate_dataset_docs.py"
            )

        generator = _load_generator_script()
        expected_markdown = generator.build_dataset_docs()
        actual_markdown = GENERATED_DOCS_PATH.read_text(encoding="utf-8")

        assert actual_markdown == expected_markdown, (
            "docs/datasets.md is out of date with the catalog — "
            "run: python scripts/generate_dataset_docs.py"
        )
