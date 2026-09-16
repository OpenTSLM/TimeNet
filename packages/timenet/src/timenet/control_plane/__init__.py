"""The control plane: one embedded DuckDB database per dataset version.

A version's structure (its records, their series, the annotations, the tasks, and the chunk
locators that point into the values plane) lives in ``control.duckdb``. The values plane keeps its
own shards, which this package does not touch.

- :mod:`timenet.control_plane.schema` holds the table definitions.
- :mod:`timenet.control_plane.checks` holds the queries the writer runs before it publishes.
- :mod:`timenet.control_plane.payload` says which table holds each field of a typed task.
- :mod:`timenet.control_plane.writer` loads a dataset into a fresh database.
- :mod:`timenet.control_plane.reader` answers with one query per table per batch.
"""

from timenet.control_plane.reader import ControlPlaneReader
from timenet.control_plane.writer import write_control_plane


__all__ = ["ControlPlaneReader", "write_control_plane"]
