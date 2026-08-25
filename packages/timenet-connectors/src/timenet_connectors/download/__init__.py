"""Async, scheme-dispatching artifact downloads for connectors.

The high-level entry points are :func:`download_files` (a list of :class:`Artifact`, any mix of ``s3://``
and ``http(s)://``) and :func:`ensure_archive` (download a zip and extract it once). Downloads report
progress through the ambient :func:`progress_sink`. The per-transport backends live in :mod:`.http` and
:mod:`.s3`.
"""

from timenet_connectors.download.fetch import Artifact, download_files, ensure_archive
from timenet_connectors.download.progress import DownloadProgress, ProgressCallback, progress_sink


__all__ = [
    "Artifact",
    "DownloadProgress",
    "ProgressCallback",
    "download_files",
    "ensure_archive",
    "progress_sink",
]
