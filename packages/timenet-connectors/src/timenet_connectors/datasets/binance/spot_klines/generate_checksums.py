"""Write checksums.json, the pinned SHA-256 digest of every monthly kline archive.

Binance publishes a ``.CHECKSUM`` file next to each archive on data.binance.vision. This script reads
those files once for the pairs and months that the connector builds, and stores the digests in
checksums.json. The connector then verifies every download against the stored digest, so a later
change to an archive on Binance's side fails the build instead of changing the dataset silently.

Run it from the repository root after you change the pairs or months in ``layout.py``::

    uv run python -m timenet_connectors.datasets.binance.spot_klines.generate_checksums
"""

import json
from pathlib import Path
import urllib.request

from timenet_connectors.datasets.binance.spot_klines.layout import PAIRS, archive_name, archive_url, months


CHECKSUMS_PATH = Path(__file__).with_name("checksums.json")
_SHA256_HEX_LENGTH = 64


def _fetch_digest(url: str, expected_name: str) -> str:
    """Read the SHA-256 digest from a ``.CHECKSUM`` file.

    Args:
        url: The URL of the ``.CHECKSUM`` file.
        expected_name: The archive name that the file must name next to its digest.

    Returns:
        The lowercase hexadecimal digest.

    Raises:
        RuntimeError: If the file does not hold one digest for ``expected_name``.
    """
    with urllib.request.urlopen(url, timeout=60) as response:  # ruff: ignore[suspicious-url-open-usage] (a fixed https URL)
        text = response.read().decode().strip()
    digest, _, name = text.partition("  ")
    if name.strip() != expected_name or len(digest) != _SHA256_HEX_LENGTH:
        raise RuntimeError(f"unexpected checksum file at {url}: {text!r}")
    return digest.lower()


def main() -> None:
    """Fetch every digest and write checksums.json."""
    checksums: dict[str, str] = {}
    for pair in PAIRS:
        for year, month in months():
            name = archive_name(pair, year, month)
            checksums[name] = _fetch_digest(archive_url(pair, year, month) + ".CHECKSUM", name)
            print(name, checksums[name])
    CHECKSUMS_PATH.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
