"""A small, reusable S3 download helper for connectors.

Downloads an ``s3://bucket/key`` object to a local path via boto3 (multipart/parallel). Credentials come
from the environment / boto3's default chain when present, and fall back to anonymous (unsigned) requests
otherwise, so public buckets such as PhysioNet's ``physionet-open`` work with no credentials. ``boto3`` is
imported lazily so base users who only curate offline datasets don't need it.
"""

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _s3_client() -> Any:
    """Build an S3 client: signed from environment credentials, else anonymous.

    Uses the ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` environment variables when both are set
    (boto3 also picks up ``AWS_SESSION_TOKEN``); otherwise returns an unsigned client so public buckets
    stay accessible with no credentials. Only the environment is consulted — ``~/.aws`` profiles and SSO
    are deliberately not, so a misconfigured local profile never breaks anonymous access.

    Returns:
        A boto3 S3 client.

    Raises:
        ImportError: If ``boto3`` (the ``physionet`` extra) is not installed.
    """
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError as exc:
        raise ImportError(
            "downloading from S3 needs the physionet extra: pip install 'timenet-connectors[physionet]'"
        ) from exc
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return boto3.client("s3")
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def download_s3_object(s3_url: str, dest: Path) -> None:
    """Download an ``s3://bucket/key`` object to ``dest``, creating parent directories.

    Args:
        s3_url: The object URL, ``s3://<bucket>/<key>``.
        dest: The local destination path.

    Raises:
        ValueError: If ``s3_url`` is not an ``s3://`` URL carrying both a bucket and a key.
    """
    parsed = urlparse(s3_url)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme != "s3" or not bucket or not key:
        raise ValueError(f"not an s3://bucket/key URL: {s3_url!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    _s3_client().download_file(bucket, key, str(dest))
