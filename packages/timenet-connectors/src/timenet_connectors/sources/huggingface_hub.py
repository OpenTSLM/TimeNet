"""Helpers for downloading files from pinned Hugging Face dataset revisions.

:func:`download_hub_files` downloads with ``huggingface_hub.snapshot_download``, limited to the
requested files. The Hub client runs the downloads in its own thread pool. Its progress bars stay
hidden: the bar that counts file bytes reaches the ambient :mod:`~timenet_connectors.download.progress`
sink instead, so the build CLI shows the download like any other connector download.
``huggingface_hub`` is declared in the connector's ``requirements.txt`` and imported lazily, which
keeps ``--no-isolation`` usable. The Hub client reads ``HF_TOKEN`` from the environment, so gated
datasets work with no extra wiring.
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


async def download_hub_files(repo: str, filenames: Sequence[str], *, revision: str, cache_dir: Path) -> list[Path]:
    """Download files from a dataset revision with one ``snapshot_download`` call.

    Args:
        repo: The Hub dataset id, such as ``"BASH-Lab/SensorCaps"``.
        filenames: Repository-relative paths of the files to download.
        revision: The revision to read, normally a pinned commit.
        cache_dir: Directory where the connector caches Hub files.

    Returns:
        The local paths, in the order of ``filenames``.

    Raises:
        ImportError: If ``huggingface_hub``, declared in the connector's requirements, is missing.
        TimeNetDownloadError: If the download fails, or a file is absent from the revision.
    """
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415 - optional connector dependency
        from huggingface_hub.utils import tqdm  # noqa: PLC0415 - optional connector dependency
    except ImportError as exc:
        raise ImportError(
            "downloading from the Hugging Face Hub needs huggingface_hub, declared in the connector's "
            "requirements.txt. Run the build without --no-isolation, or install it yourself"
        ) from exc

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
                    allow_patterns=list(filenames),
                    tqdm_class=_SinkProgress,
                )
            )
        )
    except Exception as exc:
        raise TimeNetDownloadError(f"Could not download {repo}@{revision}: {exc}") from exc
    paths = [snapshot / name for name in filenames]
    missing = [name for name, path in zip(filenames, paths, strict=True) if not path.is_file()]
    if missing:
        raise TimeNetDownloadError(f"{repo}@{revision} has no {', '.join(missing)}")
    return paths


def hub_snapshot(repo: str, revision: str, cache_dir: Path, patterns: tuple[str, ...]) -> Path:
    """Download files that match the patterns from a pinned dataset revision.

    Returns:
        The local snapshot directory.

    Raises:
        TimeNetDownloadError: If the revision cannot be listed or no files match.
    """
    from fnmatch import fnmatchcase  # noqa: PLC0415

    from huggingface_hub import list_repo_files  # noqa: PLC0415

    try:
        names = list_repo_files(repo, repo_type="dataset", revision=revision)
    except Exception as exc:
        raise TimeNetDownloadError(f"Could not list {repo}@{revision}: {exc}") from exc
    selected = [name for name in names if any(fnmatchcase(name, pattern) for pattern in patterns)]
    if not selected:
        raise TimeNetDownloadError(f"{repo}@{revision} has no files matching {patterns}")
    paths = asyncio.run(download_hub_files(repo, selected, revision=revision, cache_dir=cache_dir))
    return paths[0].parents[len(Path(selected[0]).parts) - 1]
