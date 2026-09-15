"""Values: the control plane and the values plane say the same thing about a signal.

The two planes are written by different code and land in different files. The control plane holds
the length of a signal and the locators that find its chunks. The values plane holds the chunks.
Nothing ties the two together at write time, so these checks tie them together before the commit.

A locator is backend-neutral: an artifact and two integers. Its column types cannot say that a
Parquet chunk needs both integers and a Zarr chunk needs only one. The backend of the artifact says
it instead, once the rows are in.
"""

from typing import Final

from timenet.control_plane.checks._check import Check


VALUES: Final = (
    Check(
        name="time_series_chunks_artifact_declared",
        invariant="Every chunk is in an artifact that the values plane declared.",
        sql=(
            "SELECT c.artifact_id FROM time_series_chunks c "
            "ANTI JOIN values_artifacts a ON a.artifact_id = c.artifact_id"
        ),
        remedy="Report this build. The values plane wrote a chunk into a file that it did not declare.",
    ),
    Check(
        name="parquet_chunks_have_a_row_offset",
        invariant="A Parquet locator holds a row group index and a row offset in that group.",
        sql=(
            "SELECT a.chunk_file FROM time_series_chunks c JOIN values_artifacts a ON a.artifact_id = c.artifact_id "
            "WHERE a.backend = 'parquet' AND c.chunk_minor_idx IS NULL"
        ),
        remedy="Report this build. A reader cannot find the chunk without both indexes.",
    ),
    Check(
        name="zarr_chunks_have_no_row_offset",
        invariant="A Zarr locator holds an element offset and no row offset.",
        sql=(
            "SELECT a.chunk_file FROM time_series_chunks c JOIN values_artifacts a ON a.artifact_id = c.artifact_id "
            "WHERE a.backend = 'zarr' AND c.chunk_minor_idx IS NOT NULL"
        ),
        remedy="Report this build. A Zarr chunk has one index, so the second one has no meaning.",
    ),
    Check(
        name="time_series_length_matches_chunks",
        invariant="The length of a series is the sum of the lengths of its chunks.",
        sql=(
            "SELECT s.time_series_id FROM time_series s JOIN ("
            "SELECT time_series_id, sum(n_values) AS total FROM time_series_chunks GROUP BY time_series_id"
            ") c ON c.time_series_id = s.time_series_id WHERE c.total <> s.n_values"
        ),
        remedy="Report this build. The two planes disagree about how many values the series has.",
    ),
    Check(
        name="time_series_has_chunks",
        invariant="Every series has at least one chunk in the values plane.",
        sql=(
            "SELECT s.time_series_id FROM time_series s "
            "ANTI JOIN time_series_chunks c ON c.time_series_id = s.time_series_id"
        ),
        remedy="Give the series its values, or remove the series from the dataset.",
    ),
)
"""What the control plane says about a signal is what the values plane wrote."""
