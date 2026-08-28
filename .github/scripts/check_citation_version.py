"""Check that CITATION.cff and src/envoi/_version.py agree on the version number.

The version string lives in two hand-edited places: `__version__` in
`src/envoi/_version.py` (what gets published to PyPI) and the `version:` field in
`CITATION.cff` (what ends up on the Zenodo DOI record and in the citation GitHub
generates). Nothing keeps them in sync automatically, so it is easy to bump one and
forget the other — which would archive a release under the wrong version label.

This script is run by the `package` job in .github/workflows/ci.yml. It exits with
status 1 and an explanatory message when the two disagree, so the mistake surfaces
as a failed build rather than as a permanently wrong citation record.
"""

import ast
import sys
from pathlib import Path

import yaml

# The repository root, resolved from this script's own location
# (.github/scripts/check_citation_version.py -> up three levels).
repository_root = Path(__file__).resolve().parents[2]

citation_file_path = repository_root / "CITATION.cff"
version_module_path = repository_root / "src" / "envoi" / "_version.py"


def read_version_from_citation_file() -> str:
    """Return the `version:` field of CITATION.cff as a string."""
    citation_metadata = yaml.safe_load(citation_file_path.read_text(encoding="utf-8"))
    # YAML would happily parse an unquoted 0.2.1 as a string but 1.0 as a float,
    # so normalise to str before comparing.
    return str(citation_metadata["version"])


def read_version_from_package() -> str:
    """Return `__version__` from src/envoi/_version.py without importing the package.

    The file is parsed into a syntax tree and the `__version__ = "..."` assignment is
    read straight out of it. Nothing is executed and nothing is imported, so this works
    without installing envoi and all of its heavy geospatial dependencies just to read
    one string.
    """
    version_module_syntax_tree = ast.parse(version_module_path.read_text(encoding="utf-8"))

    # Walk the module's top-level statements looking for `__version__ = <literal>`.
    for statement in version_module_syntax_tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        for assignment_target in statement.targets:
            if isinstance(assignment_target, ast.Name) and assignment_target.id == "__version__":
                # literal_eval only accepts plain literals, so a string stays a string.
                return str(ast.literal_eval(statement.value))

    raise RuntimeError(f"No `__version__` assignment found in {version_module_path}")


def main() -> int:
    citation_version = read_version_from_citation_file()
    package_version = read_version_from_package()

    if citation_version != package_version:
        print(
            "Version mismatch between the citation metadata and the package:\n"
            f"  CITATION.cff          version: {citation_version}\n"
            f"  src/envoi/_version.py __version__: {package_version}\n\n"
            "Both must be updated when cutting a release. Remember to update the\n"
            "`date-released` field in CITATION.cff at the same time.",
            file=sys.stderr,
        )
        return 1

    print(f"Citation metadata and package agree on version {package_version}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
