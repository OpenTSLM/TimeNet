"""Small helpers for downloading pinned Hugging Face dataset files."""

from pathlib import Path

from timenet.errors import TimeNetDownloadError


def hub_files(repo: str, revision: str) -> tuple[str, ...]:
    """List the files at a pinned dataset revision.

    Returns:
        Paths relative to the repository root.

    Raises:
        TimeNetDownloadError: If the pinned revision cannot be listed.
    """
    from huggingface_hub import list_repo_files  # noqa: PLC0415 - optional connector dependency

    try:
        return tuple(list_repo_files(repo, repo_type="dataset", revision=revision))
    except Exception as exc:
        raise TimeNetDownloadError(f"could not list {repo}@{revision}: {exc}") from exc


def hub_snapshot(repo: str, revision: str, cache_dir: Path, patterns: tuple[str, ...]) -> Path:
    """Download selected files from a pinned dataset revision.

    Returns:
        The local snapshot directory.

    Raises:
        TimeNetDownloadError: If the snapshot cannot be downloaded.
    """
    from huggingface_hub import snapshot_download  # noqa: PLC0415 - optional connector dependency

    try:
        return Path(
            snapshot_download(
                repo,
                repo_type="dataset",
                revision=revision,
                cache_dir=str(cache_dir),
                allow_patterns=list(patterns),
            )
        )
    except Exception as exc:
        raise TimeNetDownloadError(f"could not download {repo}@{revision}: {exc}") from exc
