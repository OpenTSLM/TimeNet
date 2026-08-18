"""Async, scheme-dispatching artifact downloads for connectors.

The high-level entry points are :func:`fetch_files` (a list of :class:`Artifact`, any mix of ``s3://``
and ``http(s)://``) and :func:`ensure_archive` (download a zip and extract it once). The per-transport
backends live in :mod:`.http` and :mod:`.s3`.
"""

from timenet_connectors.download.fetch import Artifact, ensure_archive, fetch_files


__all__ = [
    "Artifact",
    "ensure_archive",
    "fetch_files",
]
