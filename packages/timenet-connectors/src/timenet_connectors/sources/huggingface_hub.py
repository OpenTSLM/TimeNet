"""Helpers for downloading files from pinned Hugging Face dataset revisions.

``huggingface_hub`` is an optional dependency that each connector declares in its
``requirements.txt``. The helpers import it lazily and turn a missing install into an error that
states the fix, so connectors call them without wrapping the import themselves.
"""

from pathlib import Path
from types import ModuleType

from timenet.errors import TimeNetDownloadError


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


def hub_files(repo: str, revision: str) -> tuple[str, ...]:
    """List the files at a pinned dataset revision.

    Returns:
        Paths relative to the repository root.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        TimeNetDownloadError: If the pinned revision cannot be listed.
    """  # noqa: DOC502 - ImportError is raised by _hub
    hub = _hub()
    try:
        return tuple(hub.list_repo_files(repo, repo_type="dataset", revision=revision))
    except Exception as exc:
        raise TimeNetDownloadError(f"could not list {repo}@{revision}: {exc}") from exc


def hub_file(repo: str, revision: str, cache_dir: Path, filename: str) -> Path:
    """Download one file from a dataset revision.

    Returns:
        The local path of the downloaded file.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        TimeNetDownloadError: If the file cannot be downloaded.
    """  # noqa: DOC502 - ImportError is raised by _hub
    hub = _hub()
    try:
        return Path(
            hub.hf_hub_download(repo, filename, repo_type="dataset", revision=revision, cache_dir=str(cache_dir))
        )
    except Exception as exc:
        raise TimeNetDownloadError(f"could not download {filename} from {repo}@{revision}: {exc}") from exc


def hub_snapshot(repo: str, revision: str, cache_dir: Path, patterns: tuple[str, ...]) -> Path:
    """Download selected files from a pinned dataset revision.

    Returns:
        The local snapshot directory.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        TimeNetDownloadError: If the snapshot cannot be downloaded.
    """  # noqa: DOC502 - ImportError is raised by _hub
    hub = _hub()
    try:
        return Path(
            hub.snapshot_download(
                repo,
                repo_type="dataset",
                revision=revision,
                cache_dir=str(cache_dir),
                allow_patterns=list(patterns),
            )
        )
    except Exception as exc:
        raise TimeNetDownloadError(f"could not download {repo}@{revision}: {exc}") from exc
