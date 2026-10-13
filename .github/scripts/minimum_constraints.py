"""Print pip constraints that pin every dependency to its minimum supported version.

The CI job ``minimum-versions`` installs envoi with these constraints, so the
tests run against the oldest versions that ``pyproject.toml`` allows. The
minimums come from ``pyproject.toml`` itself (each ``>=`` becomes ``==``), so
the job cannot drift from the declared requirements.

Pinned: the runtime dependencies and the ``webapp`` extra. Not pinned: the
``dev`` extra (pytest, ruff, black, ...), which are tools, not dependencies of
envoi.

Usage: ``python .github/scripts/minimum_constraints.py > minimum-constraints.txt``
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

# A requirement such as "pandas>=2.1" or "requests>=2.31,<3": the name and its minimum.
_MINIMUM_PATTERN = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*>=\s*([^,;\s]+)"
)


def main() -> int:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    requirements = list(project["dependencies"]) + list(
        project.get("optional-dependencies", {}).get("webapp", [])
    )

    pins = []
    for requirement in requirements:
        match = _MINIMUM_PATTERN.match(requirement)
        if match is None:
            print(f"No '>=' minimum in requirement {requirement!r}.", file=sys.stderr)
            return 1
        name, minimum = match.groups()
        pins.append(f"{name}=={minimum}")

    print("\n".join(pins))
    return 0


if __name__ == "__main__":
    sys.exit(main())
