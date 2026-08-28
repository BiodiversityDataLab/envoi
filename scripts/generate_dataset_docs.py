#!/usr/bin/env python
"""Regenerate docs/datasets.md from the built-in dataset catalog.

Run this after editing ``src/envoi/configs/ee_catalog.yml``:

    python scripts/generate_dataset_docs.py

The generated file is committed to the repository so the dataset list is
browsable on GitHub without installing or running anything. A test
(``tests/test_catalog_docs.py``) fails if the committed file has drifted out of
sync with the catalog.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "docs" / "datasets.md"

# Allow running straight from a source checkout without installing the package.
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from envoi.catalog import BUILTIN_EE_CATALOG, load_catalogs
from envoi.catalog_docs import render_catalog_markdown

GENERATED_NOTE = (
    "Generated from `src/envoi/configs/ee_catalog.yml` by "
    "`scripts/generate_dataset_docs.py` — edit the catalog, not this file."
)


def build_dataset_docs() -> str:
    """Render the built-in catalog (only) as a Markdown document.

    Deliberately reads the bundled catalog directly instead of calling
    ``list_datasets()``, so datasets a user registered at runtime with
    ``update_catalog()`` can never leak into the committed documentation.
    """
    catalog = load_catalogs(BUILTIN_EE_CATALOG)
    datasets = catalog.get("datasets", {})
    dataset_records = [{"name": name, **datasets[name]} for name in sorted(datasets)]

    return render_catalog_markdown(
        dataset_records,
        title="Built-in datasets",
        generated_note=GENERATED_NOTE,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to write the Markdown file (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit non-zero if the file on disk is out of date.",
    )
    arguments = parser.parse_args()

    markdown_text = build_dataset_docs()

    if arguments.check:
        # Used by CI / the test suite: verify the committed file matches what
        # the current catalog would produce.
        if not arguments.output.exists():
            print(f"{arguments.output} does not exist — run this script without --check.")
            return 1
        if arguments.output.read_text(encoding="utf-8") != markdown_text:
            print(
                f"{arguments.output} is out of date. "
                f"Run: python scripts/generate_dataset_docs.py"
            )
            return 1
        print(f"{arguments.output} is up to date.")
        return 0

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(markdown_text, encoding="utf-8")
    print(f"Wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
