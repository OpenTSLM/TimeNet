"""Helpers for downloading files from pinned Hugging Face dataset revisions.

:func:`hub_snapshot` calls ``huggingface_hub.snapshot_download`` for the files that match the given
patterns, such as a filename or ``data/*.parquet``. The Hub client runs the downloads in its own thread
pool. Its progress bars are hidden; the file bytes go to the
:mod:`~timenet_connectors.download.progress` sink, so the build CLI shows them like any other download.
``huggingface_hub`` is listed in the connector's ``requirements.txt`` and imported lazily, so
``--no-isolation`` still works. The Hub client reads ``HF_TOKEN`` from the environment for gated datasets.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path
import threading
from typing import Any

from timenet.errors import TimeNetDownloadError
from timenet_connectors.download.progress import DownloadProgress, current_sink


# The Hub gives this name to the snapshot bar that counts file bytes, from huggingface_hub 1.1 on. Its other
# bars count finished files or, from 1.28, compressed network bytes.
_FILE_BYTES_BAR = "huggingface_hub.snapshot_download"

_MISSING_HUB_CLIENT = (
    "downloading from the Hugging Face Hub needs huggingface_hub, declared in the connector's "
    "requirements.txt. Run the build without --no-isolation, or install it yourself"
)


async def hub_snapshot(repo: str, revision: str, cache_dir: Path, patterns: Sequence[str]) -> Path:
    """Download the files of a dataset revision that match ``patterns``.

    Args:
        repo: The Hub dataset id, such as ``"BASH-Lab/SensorCaps"``.
        revision: The revision to read, normally a pinned commit.
        cache_dir: Directory where the connector caches Hub files.
        patterns: Repo-relative filenames or globs, such as ``"data/*.parquet"``.

    Returns:
        The local snapshot directory, with the files at their repo paths. The caller checks
        that the files it needs exist.

    Raises:
        ImportError: If ``huggingface_hub``, declared in the connector's requirements, is missing.
        TimeNetDownloadError: If the download fails.
    """
    try:
        # Optional connector dependency.
        from huggingface_hub import snapshot_download  # ruff: ignore[import-outside-top-level]
        from huggingface_hub.utils import tqdm  # ruff: ignore[import-outside-top-level]
    except ImportError as exc:
        raise ImportError(_MISSING_HUB_CLIENT) from exc

    sink = current_sink()
    key = f"{repo}@{revision[:7]}"

    # A silent tqdm_class that sends the file-bytes bar to the sink. Subclassing the Hub's tqdm makes
    # the Hub pass bar names. The Hub updates from its own threads, where the sink's ContextVar is
    # not set, so the class uses the sink captured above.
    class _SinkProgress(tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.forward = kwargs.get("name") == _FILE_BYTES_BAR
            self.downloaded = 0
            self.lock = threading.Lock()
            super().__init__(*args, **{**kwargs, "disable": True})

        def update(self, n: float | None = 1) -> bool | None:
            if self.forward and sink is not None:
                # A disabled tqdm does not count, so keep the byte count here. Worker threads update at once.
                with self.lock:
                    self.downloaded += n or 0
                    sink(DownloadProgress(key, int(self.downloaded), self.total or None))
            return super().update(n)

    try:
        # A lambda keeps the call statically resolvable to snapshot_download's str-returning overload.
        snapshot = Path(
            await asyncio.to_thread(
                lambda: snapshot_download(
                    repo,
                    repo_type="dataset",
                    revision=revision,
                    cache_dir=cache_dir,
                    allow_patterns=list(patterns),
                    tqdm_class=_SinkProgress,
                )
            )
        )
    except Exception as exc:
        raise TimeNetDownloadError(f"Could not download {repo}@{revision}: {exc}") from exc
    return snapshot
