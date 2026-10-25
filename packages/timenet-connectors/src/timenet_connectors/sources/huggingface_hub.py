"""Helpers for downloading files from pinned Hugging Face dataset revisions.

:func:`download_hub_files` downloads with ``huggingface_hub.hf_hub_download``, one worker thread per
file up to a concurrency limit. Each file's bytes reach the ambient
:mod:`~timenet_connectors.download.progress` sink instead of the Hub client's own progress bars, so the
build CLI shows them like any other connector download. ``huggingface_hub`` is declared in the
connector's ``requirements.txt`` and imported lazily, which keeps ``--no-isolation`` usable. The Hub
client reads ``HF_TOKEN`` from the environment, so gated datasets work with no extra wiring.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from timenet.errors import TimeNetDownloadError
from timenet_connectors.download.progress import DownloadProgress, ProgressCallback, current_sink


_DEFAULT_MAX_CONCURRENCY = 8


def _hub() -> ModuleType:
    """Import ``huggingface_hub`` lazily. Raise an error that states the fix if it is missing.

    Returns:
        The imported ``huggingface_hub`` module.

    Raises:
        ImportError: If ``huggingface_hub``, declared in the connector's requirements, is missing.
    """
    try:
        import huggingface_hub  # noqa: PLC0415 - optional connector dependency
    except ImportError as exc:
        raise ImportError(
            "downloading from the Hugging Face Hub needs huggingface_hub, declared in the connector's "
            "requirements.txt. Run the build without --no-isolation, or install it yourself"
        ) from exc
    return huggingface_hub


def _progress_class(sink: ProgressCallback, url: str) -> type:
    """Build a ``tqdm_class`` for ``hf_hub_download`` that reports one file's bytes to ``sink``.

    The Hub client constructs and updates it from its own threads, where the ambient sink's
    :class:`~contextvars.ContextVar` is not set, so the class calls the captured ``sink`` directly.

    Args:
        sink: The download-progress sink installed by the caller.
        url: The file's URL, which keys the file's row in the build CLI.

    Returns:
        A silent ``tqdm`` subclass.
    """
    from tqdm import tqdm  # noqa: PLC0415 - installed with huggingface_hub

    class _SinkProgress(tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.pop("name", None)  # a Hub-only keyword that plain tqdm rejects
            super().__init__(*args, **{**kwargs, "disable": True})
            self.downloaded = kwargs.get("initial", 0)

        def update(self, n: float | None = 1) -> bool | None:
            # A disabled tqdm does not count, so keep the byte count here.
            self.downloaded += n or 0
            sink(DownloadProgress(url, int(self.downloaded), self.total))
            return super().update(n)

        # With these defined, the Hub client's Xet transfers report file bytes through this one
        # bar instead of opening a second bar for network bytes.
        def update_transfer(self, n: float = 1) -> None:
            pass

        def set_transfer_postfix_str(self, *args: Any, **kwargs: Any) -> None:
            pass

    return _SinkProgress


def _download_one(repo: str, filename: str, revision: str, cache_dir: Path, sink: ProgressCallback | None) -> Path:
    """Download one file from a dataset revision into the Hub cache.

    Returns:
        The local path of the downloaded file.

    Raises:
        TimeNetDownloadError: If the file cannot be downloaded, or is missing after the download.
    """
    hub = _hub()
    url = hub.hf_hub_url(repo, filename, repo_type="dataset", revision=revision)
    try:
        path = Path(
            hub.hf_hub_download(
                repo,
                filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=str(cache_dir),
                tqdm_class=None if sink is None else _progress_class(sink, url),
            )
        )
    except Exception as exc:
        raise TimeNetDownloadError(f"could not download {filename} from {repo}@{revision}: {exc}") from exc
    if not path.is_file():
        raise TimeNetDownloadError(f"{filename} from {repo}@{revision} is missing from {path} after download")
    return path


async def download_hub_files(
    repo: str,
    filenames: Sequence[str],
    *,
    revision: str,
    cache_dir: Path,
    max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
) -> list[Path]:
    """Download files from a dataset revision concurrently, failing fast.

    Args:
        repo: The Hub dataset id, such as ``"BASH-Lab/SensorCaps"``.
        filenames: Repository-relative paths of the files to download.
        revision: The revision to read, normally a pinned commit.
        cache_dir: Directory where the connector caches Hub files.
        max_concurrency: Maximum number of downloads running at once.

    Returns:
        The local paths, in the order of ``filenames``.

    Raises:
        ImportError: If ``huggingface_hub``, declared in the connector's requirements, is missing.
        TimeNetDownloadError: If a file is absent from the revision, access is denied, or the transfer
            fails. The message names the file and revision.
    """  # noqa: DOC502 - both are raised by _hub and _download_one
    _hub()  # fail on a missing library before any download starts
    sink = current_sink()
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded(filename: str) -> Path:
        async with semaphore:
            return await asyncio.to_thread(_download_one, repo, filename, revision, cache_dir, sink)

    return list(await asyncio.gather(*(bounded(filename) for filename in filenames)))
