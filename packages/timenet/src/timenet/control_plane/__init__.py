"""The control plane: one embedded DuckDB database per dataset version.

A version's structure (its records, their series, the annotations, the tasks, and the chunk
locators that point into the values plane) lives in ``control.duckdb``. The values plane keeps its
own shards and is not touched here. :mod:`timenet.control_plane.schema` holds the table definitions,
:mod:`timenet.control_plane.checks` the queries the writer runs before it publishes,
:mod:`timenet.control_plane.payload` which table holds each field of a typed task,
:mod:`timenet.control_plane.writer` loads a dataset into a fresh database, and
:mod:`timenet.control_plane.reader` answers the reader's questions with one query per table per
batch.
"""

from timenet.control_plane.reader import ControlPlaneReader
from timenet.control_plane.writer import write_control_plane


__all__ = ["ControlPlaneReader", "write_control_plane"]
