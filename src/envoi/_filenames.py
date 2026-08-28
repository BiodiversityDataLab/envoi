# src/envoi/_filenames.py
"""Portable filename construction for exported raster tiles.

Tile filenames are built from the user's sample ID column, which comes straight
out of an uploaded CSV and can contain anything at all. Real GBIF occurrenceIDs
routinely look like ``http://arctos.database.museum/guid/MSB:Mamm:1`` or
``urn:catalog:MO:Tropicos:100123`` — the ``/`` is read as a path separator, and
``:`` is outright illegal on Windows. Since envoi (and the webapp built on it)
must run on Windows, macOS and Linux alike, every filename component is reduced
to a single conservative character set that is valid on all three, and inside
zip archives and URLs as well.

The three public helpers:

* :func:`sanitize_filename_component` — turn one arbitrary string into a safe
  filename component.
* :func:`build_tile_filenames` — map a whole batch of sample IDs to tile
  filenames in one pass, so both adapters name their tiles identically.
* :func:`is_safe_path_component` — a boolean check used to *reject* unsafe
  config-supplied path parts (batch IDs, dataset keys) rather than silently
  rewriting them.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence

# The only characters we allow through. This is a whitelist rather than a
# blacklist of "bad" characters on purpose: it is simultaneously valid on NTFS,
# APFS and ext4, inside zip archives, and in a URL path segment, so we never
# have to track which exotic character breaks which consumer.
_ALLOWED_CHARACTERS_PATTERN = re.compile(r"[^A-Za-z0-9._-]")

# Runs of the replacement character collapse to one, so a messy ID like
# "a // b" becomes "a_b" rather than "a___b".
_REPEATED_UNDERSCORES_PATTERN = re.compile(r"_+")

# Windows refuses to create files with these names (case-insensitively, and
# even with an extension appended — "CON.tif" is rejected just like "CON").
# Our tile stems are "<id>-<dataset>" so a bare reserved name cannot normally
# occur, but this module is shared and must hold for any caller.
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{digit}" for digit in range(1, 10)),
        *(f"LPT{digit}" for digit in range(1, 10)),
    }
)

# Used when a value sanitizes down to nothing at all (e.g. an ID of "///").
_EMPTY_COMPONENT_FALLBACK = "id"

# Most filesystems cap a single path component at 255 bytes. We budget against
# this so that the *whole* assembled filename — id, dataset name, window suffix
# and ".tif" — fits, not just the ID part.
MAX_FILENAME_BYTES = 255

# Length of the hash fragment appended to an ID that had to be rewritten. Eight
# hex characters is 32 bits: ample to keep thousands of points distinct, while
# staying short enough to leave the filename readable.
_HASH_SUFFIX_LENGTH = 8


def _truncate_to_bytes(text: str, max_bytes: int) -> str:
    """Trim ``text`` so its UTF-8 encoding fits within ``max_bytes``.

    Sanitized components are pure ASCII (one byte per character), so this is
    normally a plain slice. The encode/decode round-trip is kept anyway so the
    helper stays correct if it is ever handed non-ASCII text.
    """
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    # errors="ignore" drops a partial multi-byte character at the cut point
    # rather than raising.
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def sanitize_filename_component(value: object, *, max_length: int = MAX_FILENAME_BYTES) -> str:
    """Reduce an arbitrary value to a filename component that is safe everywhere.

    The rules, in order:

    1. Coerce to ``str`` and strip surrounding whitespace.
    2. Replace every character outside ``[A-Za-z0-9._-]`` with ``_``. This also
       folds accented and non-Latin letters (``é`` becomes ``_``) — ASCII-only
       is the only character set we can rely on across every filesystem and
       archive tool our users might open the output with.
    3. Collapse repeated underscores, then strip leading/trailing ``.``, ``_``
       and ``-``. Windows silently rejects trailing dots and spaces.
    4. Rename Windows reserved device names out of the way.
    5. Fall back to ``"id"`` if nothing survived.
    6. Truncate to ``max_length`` bytes.
    7. If — and only if — the result differs from the original string, append a
       short hash of that original.

    Step 7 is what makes the result collision-free: two different IDs that
    sanitize to the same text (``"a/b"`` and ``"a_b"`` both become ``"a_b"``)
    get different hashes and therefore different filenames, so neither tile
    silently overwrites the other. It also means an ID that was already clean
    passes through completely untouched, leaving existing users' filenames
    byte-for-byte unchanged.

    Args:
        value: The raw value to convert — typically one cell of the user's ID
            column, but anything with a string representation is accepted.
        max_length: Maximum length of the returned component in bytes,
            including the hash suffix when one is added.

    Returns:
        A non-empty string containing only ``A-Z``, ``a-z``, ``0-9``, ``.``,
        ``_`` and ``-``.
    """
    original_text = str(value).strip()

    # Rule 2: everything outside the whitelist becomes an underscore.
    sanitized = _ALLOWED_CHARACTERS_PATTERN.sub("_", original_text)

    # Rule 3: tidy up the underscores this may have produced, then trim
    # separator characters off both ends.
    sanitized = _REPEATED_UNDERSCORES_PATTERN.sub("_", sanitized)
    sanitized = sanitized.strip("._-")

    # Rule 4: a reserved Windows device name gets a prefix so it is no longer
    # reserved, while staying recognisable to the user.
    if sanitized.upper() in _WINDOWS_RESERVED_NAMES:
        sanitized = f"{sanitized}_"

    # Rule 5: nothing left (e.g. the ID was "///" or an empty cell).
    if not sanitized:
        sanitized = _EMPTY_COMPONENT_FALLBACK

    # A value needs the hash suffix if the whitelist changed it, *or* if it is
    # too long to fit whole. Truncation alone would let two clean IDs sharing a
    # long common prefix collapse onto one filename, so it counts as a rewrite.
    was_rewritten = sanitized != original_text or len(sanitized.encode("utf-8")) > max_length

    if not was_rewritten:
        # Clean and short enough: return it exactly as the user wrote it.
        return sanitized

    # Rule 7: append a stable short hash of the *original* value so distinct
    # inputs never collapse onto one filename. sha1 is used purely as a
    # content fingerprint here, not for any security purpose.
    hash_suffix = hashlib.sha1(original_text.encode("utf-8"), usedforsecurity=False).hexdigest()[
        :_HASH_SUFFIX_LENGTH
    ]

    # Reserve room for the hash and its separating hyphen, then truncate the
    # readable part to whatever budget remains.
    reserved_for_hash = len(hash_suffix) + 1
    readable_part = _truncate_to_bytes(sanitized, max(0, max_length - reserved_for_hash))
    # Truncation can re-expose a trailing separator; strip it again so the
    # name never reads as "abc_-1a2b3c4d".
    readable_part = readable_part.rstrip("._-") or _EMPTY_COMPONENT_FALLBACK

    return f"{readable_part}-{hash_suffix}"


def build_tile_filenames(
    sample_ids: Iterable[object],
    dataset_name: str,
    filename_suffix: str | None = None,
) -> list[str]:
    """Build one tile filename per sample ID, safe on every operating system.

    Both the Earth Engine and local adapters call this once before their
    per-point export loop and then index into the result, so the two code paths
    are guaranteed to name their tiles identically.

    The naming scheme is unchanged from before sanitization existed:
    ``"<id>-<dataset>.tif"``, or ``"<id>-<dataset>-<suffix>.tif"`` when the
    caller passes a suffix (used to keep multiple window sizes distinct inside
    one dataset folder).

    Args:
        sample_ids: The per-point IDs, in input order.
        dataset_name: Catalog key of the dataset being exported.
        filename_suffix: Optional extra component, e.g. ``"200m"``.

    Returns:
        A list of filenames aligned with ``sample_ids``.

    Raises:
        ValueError: if two IDs somehow produce the same filename. The hash
            suffix added by :func:`sanitize_filename_component` makes this
            unreachable for distinct IDs, and duplicate IDs are rejected
            upstream by ``_validate_sample_ids``; this is a last-resort guard
            against silently overwriting a user's tile.
    """
    # The dataset name and suffix come from the catalog and from our own code
    # respectively, but they land in the same filename, so they get the same
    # treatment rather than being trusted.
    safe_dataset_name = sanitize_filename_component(dataset_name)
    suffix_part = f"-{sanitize_filename_component(filename_suffix)}" if filename_suffix else ""

    # Everything after the sample ID is fixed-length, so compute the budget for
    # the ID once and hand it to the sanitizer, keeping the assembled filename
    # within the filesystem's per-component limit.
    fixed_part_length = len(f"-{safe_dataset_name}{suffix_part}.tif".encode())
    id_budget = max(1, MAX_FILENAME_BYTES - fixed_part_length)

    filenames: list[str] = []
    # Maps each generated filename back to the ID that produced it, purely so
    # the collision guard below can name both culprits.
    filename_to_source_id: dict[str, object] = {}

    for sample_id in sample_ids:
        safe_id = sanitize_filename_component(sample_id, max_length=id_budget)
        filename = f"{safe_id}-{safe_dataset_name}{suffix_part}.tif"

        if filename in filename_to_source_id:
            raise ValueError(
                f"Sample IDs {filename_to_source_id[filename]!r} and {sample_id!r} both "
                f"produce the tile filename {filename!r}. Give these points distinct IDs."
            )

        filename_to_source_id[filename] = sample_id
        filenames.append(filename)

    return filenames


def is_safe_path_component(value: object) -> bool:
    """Return True if ``value`` can be used as a path component unchanged.

    Used for values that come from a run config or a user catalog rather than
    from a data file — batch IDs and dataset keys, both of which become
    directory names. Those are hand-authored, so the right response to an
    unsafe one is a clear error telling the author to fix it, not a silent
    rewrite. This also rejects ``".."``, which would otherwise let a batch ID
    write outside the chosen output directory.
    """
    text = str(value)
    if not text or text != text.strip():
        return False
    # "." and ".." are valid filename characters but are directory references,
    # not names.
    if text in {".", ".."}:
        return False
    if text.upper() in _WINDOWS_RESERVED_NAMES:
        return False
    if len(text.encode("utf-8")) > MAX_FILENAME_BYTES:
        return False
    # Trailing dots are stripped by Windows, silently changing the name.
    if text.endswith("."):
        return False
    return _ALLOWED_CHARACTERS_PATTERN.search(text) is None


def describe_unsafe_path_component(value: object) -> str:
    """Explain, for an error message, why ``value`` is not a usable path component.

    Kept next to :func:`is_safe_path_component` so the two stay in step. The
    returned text is a sentence fragment intended to be embedded in a larger
    ValueError message by the caller.
    """
    text = str(value)
    if not text:
        return "it is empty"
    if text != text.strip():
        return "it has leading or trailing whitespace"
    if text in {".", ".."}:
        return "it is a directory reference"
    if text.upper() in _WINDOWS_RESERVED_NAMES:
        return f"'{text}' is a reserved device name on Windows"
    if len(text.encode("utf-8")) > MAX_FILENAME_BYTES:
        return f"it is longer than {MAX_FILENAME_BYTES} bytes"
    if text.endswith("."):
        return "it ends with a '.', which Windows silently strips"

    illegal_characters = sorted(set(_ALLOWED_CHARACTERS_PATTERN.findall(text)))
    if illegal_characters:
        rendered = ", ".join(repr(character) for character in illegal_characters)
        return f"it contains the character(s) {rendered}, which are not valid on all systems"

    return "it is not a valid path component"


def format_id_list(values: Sequence[object], limit: int = 5) -> str:
    """Render a few example IDs for an error message, eliding the rest.

    Error messages about bad IDs are much more useful when they name the
    offending values, but an input with thousands of bad rows should not print
    thousands of them.
    """
    shown = [repr(value) for value in values[:limit]]
    if len(values) > limit:
        shown.append(f"... and {len(values) - limit} more")
    return ", ".join(shown)
