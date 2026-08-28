"""Tests for the portable-filename helpers in ``envoi._filenames``.

These are pure-Python unit tests — no rasters, no GEE, no filesystem. The
integration side (tiles actually landing on disk with these names) lives in
``test_extract.py::TestTileNaming``.
"""

import re

import pytest

from envoi._filenames import (
    MAX_FILENAME_BYTES,
    build_tile_filenames,
    describe_unsafe_path_component,
    format_id_list,
    is_safe_path_component,
    sanitize_filename_component,
)

# Every sanitized component must match this: the character set that is valid
# on NTFS, APFS and ext4 alike, and inside zip archives and URLs.
PORTABLE_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class TestSanitizeFilenameComponent:
    @pytest.mark.parametrize(
        "clean_id",
        [
            "T2T1DJ",
            "obs.42",
            "site-1_plot-2",
            "1234567890",
            "a",
        ],
    )
    def test_already_safe_ids_pass_through_untouched(self, clean_id):
        """IDs that are already portable must not be rewritten.

        This is the backward-compatibility guard: existing users' tile
        filenames have to stay byte-for-byte identical, so no hash suffix may
        be appended to an ID that never needed one.
        """
        assert sanitize_filename_component(clean_id) == clean_id

    @pytest.mark.parametrize(
        "hostile_id",
        [
            "http://arctos.database.museum/guid/MSB:Mamm:1",  # URL-style occurrenceID
            "urn:catalog:MO:Tropicos:100123",  # Darwin Core urn form
            "record 12 ",  # spaces, trailing whitespace
            "émile",  # non-ASCII letters
            "a*b?c|d",  # Windows-illegal punctuation
            "../escape",  # path traversal attempt
            "CON",  # Windows reserved device name
            "///",  # sanitizes down to nothing
            "x" * 400,  # longer than NAME_MAX
        ],
    )
    def test_hostile_ids_become_portable(self, hostile_id):
        """Anything the user's CSV can contain reduces to a portable name."""
        result = sanitize_filename_component(hostile_id)
        assert PORTABLE_COMPONENT_PATTERN.match(result), f"{result!r} is not portable"
        assert len(result.encode("utf-8")) <= MAX_FILENAME_BYTES
        # Windows silently strips these, which would change the name behind
        # our back.
        assert not result.endswith((".", " "))
        assert not result.startswith((".", " "))

    def test_path_separators_are_removed(self):
        """The `/` that caused the original bug must not survive."""
        result = sanitize_filename_component("http://x/guid/A:B")
        assert "/" not in result
        assert ":" not in result

    def test_ids_that_sanitize_alike_stay_distinct(self):
        """`a/b` and `a_b` both reduce to `a_b`, so they need the hash suffix.

        Without it the two points would write to the same filename and one
        tile would silently overwrite the other.
        """
        assert sanitize_filename_component("a/b") != sanitize_filename_component("a_b")

    def test_long_ids_sharing_a_prefix_stay_distinct(self):
        """Truncation alone would collapse these two onto one name."""
        shared_prefix = "x" * 300
        first = sanitize_filename_component(shared_prefix + "-first")
        second = sanitize_filename_component(shared_prefix + "-second")
        assert first != second
        assert len(first.encode("utf-8")) <= MAX_FILENAME_BYTES

    def test_hash_suffix_is_stable_across_calls(self):
        """The same input must always produce the same filename.

        Re-running an extraction has to overwrite the previous tiles rather
        than accumulating a second copy under a different name.
        """
        assert sanitize_filename_component("a/b") == sanitize_filename_component("a/b")

    def test_reserved_windows_name_is_renamed(self):
        """`CON` is unusable as a filename on Windows even with an extension."""
        result = sanitize_filename_component("CON")
        assert result.upper() != "CON"

    def test_empty_result_falls_back_to_a_usable_name(self):
        """An ID made entirely of illegal characters still needs some name."""
        result = sanitize_filename_component("///")
        assert PORTABLE_COMPONENT_PATTERN.match(result)

    def test_max_length_is_respected(self):
        """The caller's budget caps the result, hash suffix included."""
        result = sanitize_filename_component("y" * 500, max_length=40)
        assert len(result.encode("utf-8")) <= 40

    def test_non_string_values_are_accepted(self):
        """Numeric ID columns are common; they must not raise."""
        assert sanitize_filename_component(12345) == "12345"


class TestBuildTileFilenames:
    def test_naming_scheme_is_unchanged_for_clean_ids(self):
        """The historical `<id>-<dataset>.tif` layout is preserved."""
        assert build_tile_filenames(["T2T1DJ", "T6WPKD"], "dem_local") == [
            "T2T1DJ-dem_local.tif",
            "T6WPKD-dem_local.tif",
        ]

    def test_suffix_is_inserted_before_the_extension(self):
        """Multi-window runs disambiguate tiles by window size."""
        assert build_tile_filenames(["abc"], "dem_local", "200m") == ["abc-dem_local-200m.tif"]

    def test_colliding_ids_produce_distinct_filenames(self):
        assert len(set(build_tile_filenames(["a/b", "a_b", "a:b"], "dem_local"))) == 3

    def test_whole_filename_fits_within_the_filesystem_limit(self):
        """The ID budget accounts for the dataset name and suffix too."""
        filenames = build_tile_filenames(["z" * 500], "dem_copernicus_glo30", "1000m")
        assert len(filenames[0].encode("utf-8")) <= MAX_FILENAME_BYTES

    def test_duplicate_ids_raise_rather_than_overwrite(self):
        """Last-resort guard: identical IDs would name the same file."""
        with pytest.raises(ValueError, match="produce the tile filename"):
            build_tile_filenames(["same", "same"], "dem_local")

    def test_output_is_aligned_with_input_order(self):
        ids = ["p1", "p2", "p3"]
        filenames = build_tile_filenames(ids, "dem_local")
        assert len(filenames) == len(ids)
        assert filenames[1].startswith("p2-")


class TestIsSafePathComponent:
    @pytest.mark.parametrize("safe_value", ["terrain", "output1", "a.b-c_d", "DEM30"])
    def test_accepts_portable_names(self, safe_value):
        assert is_safe_path_component(safe_value)

    @pytest.mark.parametrize(
        "unsafe_value",
        [
            "",  # empty
            "my/batch",  # path separator
            "..",  # directory reference
            ".",
            "with space",
            " leading",
            "trailing ",
            "trailing.",  # Windows strips the dot
            "CON",  # reserved device name
            "urn:catalog:1",  # illegal on Windows
            "x" * 300,  # too long
        ],
    )
    def test_rejects_unportable_names(self, unsafe_value):
        assert not is_safe_path_component(unsafe_value)

    def test_every_rejection_has_an_explanation(self):
        """describe_unsafe_path_component must stay in step with the check."""
        for unsafe_value in ["", "my/batch", "..", "CON", "trailing.", "x" * 300]:
            explanation = describe_unsafe_path_component(unsafe_value)
            assert explanation and explanation != "it is not a valid path component"


class TestFormatIdList:
    def test_short_lists_are_shown_in_full(self):
        assert format_id_list(["a", "b"]) == "'a', 'b'"

    def test_long_lists_are_elided(self):
        rendered = format_id_list([str(index) for index in range(20)], limit=3)
        assert "and 17 more" in rendered
        # Only the first `limit` values are named.
        assert "'3'" not in rendered
