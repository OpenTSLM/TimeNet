"""Helpers for downloading files from pinned Hugging Face dataset revisions."""

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
        TimeNetDownloadError: If the pinned revision cannot be listed.
    """
    hub = _hub()
    try:
        return tuple(hub.list_repo_files(repo, repo_type="dataset", revision=revision))
    except Exception as exc:
        raise TimeNetDownloadError(f"could not list {repo}@{revision}: {exc}") from exc


def hub_file(repo: str, filename: str, *, revision: str, cache_dir: Path) -> Path:
    """Download one file from a dataset revision.

    Returns:
        The local path of the downloaded file.

    Raises:
        TimeNetDownloadError: If the file cannot be downloaded.
    """
    hub = _hub()
    try:
        return Path(
            hub.hf_hub_download(repo, filename, repo_type="dataset", revision=revision, cache_dir=str(cache_dir))
        )
    except Exception as exc:
        raise TimeNetDownloadError(f"could not download {filename} from {repo}@{revision}: {exc}") from exc
