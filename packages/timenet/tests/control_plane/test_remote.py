"""What a DuckDB control plane means for a registry served over HTTP or S3.

Two questions, and they have different answers.

The first is whether the existing fetch path still works. It does, and without a code change: the
registry stages every file the manifest names, and the control database is just another named file
with a checksum and a size.

The second is whether a consumer can read the control plane *in place*, over range requests, without
fetching it at all. That would let a client answer "which records carry this annotation" against a
remote dataset before deciding to download any of it. These tests pin down which of those actually
works today, so the answer is measured rather than assumed.
"""

from __future__ import annotations

import functools
import http.server
import shutil
import threading

import duckdb
import pytest

from timenet.control_plane import TimeFReader, TimeFWriter
from timenet.format.checksums import file_checksum
from timenet.format.constants import CONTROL_DB_FILE
from timenet.registry.local import LocalRegistry
from timenet.testing import make_dataset


def _values(reader, record_external_id, signal_external_id):
    """Read one signal's values, reaching its surrogate id the way a caller does."""
    signals = reader.record(record_external_id).signals()
    (signal,) = [found for found in signals if found.external_id == signal_external_id]
    return reader.values(signal.signal_id)


@pytest.fixture
def published(tmp_path):
    """A local registry holding one committed version."""
    dataset = make_dataset(n_records=3, n_values=128)
    root = tmp_path / "registry"
    with TimeFWriter(root, dataset.metadata) as writer:
        manifest = writer.write(dataset)
    return root, dataset.metadata, manifest


@pytest.fixture
def server(published):
    """Serve the registry over plain HTTP, with range requests, and yield its base URL."""
    root, _, _ = published
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_manifest_names_the_control_database(published):
    _, _, manifest = published
    assert manifest.files.control_db is not None
    assert manifest.files.control_db.path == "control.duckdb"
    assert manifest.files.control_db.checksum.startswith("sha256:")
    assert manifest.files.control_db.size > 0


def test_the_control_database_is_in_the_download_set(published):
    """The generic fetch path picks the database up with no special case for it."""
    _, _, manifest = published
    assert "control.duckdb" in manifest.files.all_parts()
    assert any(part.path.startswith("time_series/") for part in manifest.files.all_files())


def test_download_then_read(published, tmp_path):
    """The consumer path end to end: stage every file, swap it in, open it locally."""
    root, metadata, _ = published
    destination = tmp_path / "downloaded"
    LocalRegistry(root).download_version(metadata.dataset_id, "1.0.0", destination)

    assert (destination / "control.duckdb").exists()
    assert (destination / "manifest.json").exists()
    with TimeFReader(destination) as reader:
        assert reader.record_ids() == ["record-000", "record-001", "record-002"]
        assert _values(reader, "record-000", "record-000-lead-i").shape == (128,)


def test_a_read_leaves_the_downloaded_copy_byte_identical(published, tmp_path):
    """A read-only open must not write a WAL beside the file, or the checksum stops matching."""
    root, metadata, manifest = published
    destination = tmp_path / "downloaded"
    LocalRegistry(root).download_version(metadata.dataset_id, "1.0.0", destination)

    with TimeFReader(destination) as reader:
        reader.record("record-000")
    assert not (destination / "control.duckdb.wal").exists()
    assert file_checksum(destination / "control.duckdb") == manifest.files.control_db.checksum


def test_http_range_requests_reach_the_values_plane(server):
    """Parquet over HTTP already works, which is what the values plane needs."""
    connection = duckdb.connect()
    connection.execute("INSTALL httpfs")
    connection.execute("LOAD httpfs")
    url = f"{server}/test/bedside/1.0.0/time_series/part-00000000.parquet"
    row = connection.execute("SELECT count(*) FROM read_parquet(?)", [url]).fetchone()
    assert row is not None
    rows = row[0]
    connection.close()
    assert rows > 0


def test_reading_the_control_database_over_http_without_downloading_it(server):
    """Attach the control plane in place and query it over range requests.

    This is the capability that would let a client filter datasets before fetching them. If DuckDB
    ever stops supporting a remote read-only attach, this test says so rather than the behaviour
    silently disappearing.
    """
    connection = duckdb.connect()
    connection.execute("INSTALL httpfs")
    connection.execute("LOAD httpfs")
    connection.execute(f"ATTACH '{server}/test/bedside/1.0.0/control.duckdb' AS remote (READ_ONLY)")
    found = connection.execute(
        "SELECT DISTINCT r.external_id FROM remote.record_annotations a "
        "JOIN remote.annotations n USING (annotation_id) "
        "JOIN remote.records r USING (record_id) "
        "WHERE n.name = 'patient_sex' ORDER BY 1"
    ).fetchall()
    connection.close()
    assert [row[0] for row in found] == ["record-000", "record-001", "record-002"]


def test_remote_reader_reads_a_version_it_never_downloaded(server, tmp_path):
    """The reader itself, pointed at a remote database, with only the values plane left local."""
    with TimeFReader(tmp_path, database=f"{server}/test/bedside/1.0.0/control.duckdb") as reader:
        assert reader.record_ids() == ["record-000", "record-001", "record-002"]
        record = reader.record("record-000")
        assert [source.name for source in record.sources] == ["Bedside monitor"]
        assert reader.records_with("patient_sex", "male") == ["record-000", "record-001", "record-002"]


def test_a_remote_walk_finds_the_attached_catalog(server, tmp_path):
    """A walk streams its ids from a second cursor, which starts on the instance's own catalog.

    Over a URL the control plane is attached beside an empty in-memory database, so a cursor that
    keeps that default answers an unqualified ``FROM records`` with a catalog error. A local file is
    the whole database, which is why only a remote open ever showed this.
    """
    with TimeFReader(tmp_path, database=f"{server}/test/bedside/1.0.0/control.duckdb") as reader:
        assert [record.external_id for record in reader.iter_records(batch_size=2)] == [
            "record-000",
            "record-001",
            "record-002",
        ]
        assert [task.external_id for task in reader.iter_tasks(batch_size=2)] == [
            "diagnosis-000",
            "diagnosis-001",
            "diagnosis-002",
        ]


def test_a_quote_in_a_remote_path_is_not_the_end_of_the_sql_literal(published, tmp_path):
    """DuckDB takes no parameter for an ATTACH target, so a path with a quote has to be escaped."""
    root, _, _ = published
    served = tmp_path / "served"
    (served / "o'brien").mkdir(parents=True)
    shutil.copy(root / "test/bedside/1.0.0" / CONTROL_DB_FILE, served / "o'brien" / CONTROL_DB_FILE)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/o'brien/{CONTROL_DB_FILE}"
        with TimeFReader(tmp_path, database=url) as reader:
            assert reader.record_ids() == ["record-000", "record-001", "record-002"]
    finally:
        httpd.shutdown()
        httpd.server_close()
