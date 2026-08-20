"""Download an ``s3://bucket/key`` object for connectors.

This helper downloads an ``s3://bucket/key`` object to a local path with boto3. boto3 runs the transfer
in parallel with multipart downloads. If the environment provides credentials, the client uses them. If
not, the client falls back to anonymous (unsigned) requests, so public buckets such as PhysioNet's
``physionet-open`` work with no credentials. The code imports ``boto3`` lazily, so users who curate only
offline datasets do not need it.
"""

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from timenet.errors import TimeFValidationError
from timenet_connectors.download.progress import DownloadProgress, ProgressCallback, current_sink


def _s3_client() -> Any:
    """Build a signed or anonymous S3 client.

    If both ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY`` are set, the client uses them. boto3 also
    picks up ``AWS_SESSION_TOKEN``. If not, this function returns an unsigned client, so public buckets
    stay accessible with no credentials. This function reads only the environment. It ignores ``~/.aws``
    profiles and SSO, so a bad local profile cannot break anonymous access.

    Returns:
        A boto3 S3 client.

    Raises:
        ImportError: If the caller has not installed ``boto3`` (the ``physionet`` extra).
    """
    try:
        import boto3  # noqa: PLC0415
        from botocore import UNSIGNED  # noqa: PLC0415
        from botocore.config import Config  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "downloading from S3 needs the physionet extra: pip install 'timenet-connectors[physionet]'"
        ) from exc
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return boto3.client("s3")
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def download_s3_object(s3_url: str, dest: Path) -> None:
    """Download an ``s3://bucket/key`` object to ``dest``, creating parent directories.

    Writes atomically. The bytes land in a ``.part`` temp file, and a rename moves it into place
    only on success. So an interrupted download never leaves a truncated file that a later
    ``skip_existing`` check would trust.

    Args:
        s3_url: The object URL, ``s3://<bucket>/<key>``.
        dest: The local destination path.

    Raises:
        TimeFValidationError: If ``s3_url`` is not an ``s3://`` URL carrying both a bucket and a key.
    """
    parsed = urlparse(s3_url)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme != "s3" or not bucket or not key:
        raise TimeFValidationError(f"not an s3://bucket/key URL: {s3_url!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.parent / f"{dest.name}.part"
    client = _s3_client()
    # boto3 invokes the Callback from its own transfer worker threads. These threads do not inherit
    # the ambient ContextVar sink, so capture it here on the calling thread and call it directly.
    sink = current_sink()
    try:
        if sink is not None:
            # boto3 reports bytes incrementally. A HEAD gives the total for a full progress figure.
            total = client.head_object(Bucket=bucket, Key=key)["ContentLength"]
            transferred = 0

            def _on_bytes(count: int, report: ProgressCallback = sink) -> None:
                nonlocal transferred
                transferred += count
                report(DownloadProgress(s3_url, transferred, total))

            client.download_file(bucket, key, str(part), Callback=_on_bytes)
        else:
            client.download_file(bucket, key, str(part))
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)  # a partial or interrupted download must not masquerade as complete
        raise
