"""Download an ``s3://bucket/key`` object for connectors.

This helper downloads an ``s3://bucket/key`` object to a local path with boto3, which runs the transfer
in parallel with multipart downloads. When AWS credentials are configured (environment, ``AWS_PROFILE``
/ the shared ``~/.aws`` config / SSO, container and instance roles), the client uses them, so a private
source stays reachable. When none are configured, it reads anonymously, which is what a public bucket
such as physionet-open needs. The code imports ``boto3`` lazily, so users who curate only offline
datasets do not need it.
"""

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from timenet.errors import TimeFValidationError, TimeNetBuildError
from timenet_connectors.download.progress import DownloadProgress, ProgressCallback, current_sink


_LOG = logging.getLogger(__name__)


def _s3_client() -> Any:
    """Build a boto3 S3 client, using credentials when available and anonymous access otherwise.

    boto3 resolves credentials through its full chain: environment variables, ``AWS_PROFILE`` and the
    shared ``~/.aws`` config (including SSO), and container or instance roles. When that yields
    credentials, the client uses them, so a private source stays reachable. When no credentials are
    configured, or the provider chain fails to load (for example an SSO profile missing an optional
    dependency), the client falls back to anonymous access, which is what a public bucket such as
    physionet-open needs.

    Returns:
        A boto3 S3 client.

    Raises:
        TimeNetBuildError: If ``boto3``, declared in the connector's requirements, is not installed.
    """
    try:
        import boto3  # noqa: PLC0415
        from botocore import UNSIGNED  # noqa: PLC0415
        from botocore.config import Config  # noqa: PLC0415
        from botocore.exceptions import BotoCoreError  # noqa: PLC0415
    except ImportError as exc:
        raise TimeNetBuildError(
            "downloading from S3 needs boto3, declared in the connector's requirements.txt. "
            "Run the build without --no-isolation, or install it yourself"
        ) from exc
    session = boto3.session.Session()
    try:
        credentials = session.get_credentials()
    except BotoCoreError:
        # A broken or incomplete credential provider (for example an SSO profile missing an optional
        # dependency) must not stop an anonymous read of a public bucket.
        credentials = None
    if credentials is not None:
        return session.client("s3")
    _LOG.info("no usable AWS credentials; reading S3 with unsigned (anonymous) requests")
    return session.client("s3", config=Config(signature_version=UNSIGNED))


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
