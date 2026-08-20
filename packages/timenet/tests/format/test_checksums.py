import hashlib

import pyarrow.fs as pafs

from timenet.format.checksums import CHECKSUM_PREFIX, file_checksum, stream_checksum


def test_file_and_stream_checksums_agree(tmp_path):
    # the writer hashes a local Path; the reader hashes a stream opened through a pyarrow filesystem.
    # verify() only means anything if both routes produce the same digest.
    path = tmp_path / "blob.bin"
    payload = b"timenet" * 500_000  # 3.5 MB, spanning several 1 MiB hash blocks
    path.write_bytes(payload)

    expected = CHECKSUM_PREFIX + hashlib.sha256(payload).hexdigest()
    assert file_checksum(path) == expected
    with path.open("rb") as handle:
        assert stream_checksum(handle) == expected
    with pafs.LocalFileSystem().open_input_file(str(path)) as handle:
        assert stream_checksum(handle) == expected


def test_checksum_of_empty_file(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert file_checksum(path) == CHECKSUM_PREFIX + hashlib.sha256(b"").hexdigest()
