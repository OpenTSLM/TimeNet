import re
import subprocess
import sys

import pytest

from timenet.builders import BuildOptions, installed_builder
from timenet.client import TimeNet
from timenet.errors import (
    ConnectorCardWarning,
    TimeFValidationError,
    TimeNetDatasetNotFoundError,
    TimeNetInvalidCardError,
    TimeNetVersionUnavailableError,
)
from timenet.manifest import Manifest, PayloadSource
from timenet.manifest.files import FilePart
from timenet.progress import BytesMoved, ShardWritten
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry, open_local_version
from timenet.testing import make_dataset
from timenet.types.references import DatasetRef
from timenet.values_backends import ValuesBackend
from timenet.writer.progress import ProgressStage, WriteProgressEvent
from timenet_connectors import datasets as datasets_module
from timenet_connectors.builder import backend as backend_module
from timenet_connectors.builder.backend import ConnectorBuilder
from timenet_connectors.discovery import known_ids
from timenet_connectors.download.progress import DownloadProgress, current_sink, progress_sink, report_progress


HELLO = DatasetRef.parse("timenet/hello-world@1.0.0")


def _unexpected_isolation(*args, **kwargs):
    raise AssertionError(f"run_isolated_build should not have run: {args} {kwargs}")


def _unexpected_pipeline(*args, **kwargs):
    raise AssertionError(f"run_pipeline should not have run: {args} {kwargs}")


def _pins():
    pinned = FilePart(path="control.duckdb", checksum="sha256:" + "0" * 64, size=1)
    return {DatasetRef.parse("test/reader@1.0.0"): PayloadSource(release=HELLO, files=(pinned,))}


def test_loading_the_builder_backend_imports_no_cli_library():
    # The SDK loads the backend through its entry point to list dataset IDs and to read cards. The
    # libraries of the CLI have no use there.
    cli_modules = ("typer", "timenet.cli.ui", "timenet_connectors.builder.cli")
    code = f"import sys, timenet_connectors.builder.backend; print([m for m in {cli_modules!r} if m in sys.modules])"

    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    assert result.stdout.strip() == "[]"


def test_connector_builder_lists_its_dataset_ids_without_importing_a_connector(monkeypatch):
    leaf = "timenet_connectors.datasets.physionet.ecg_qa_cot"
    monkeypatch.delitem(sys.modules, leaf, raising=False)

    dataset_ids = list(ConnectorBuilder().dataset_ids())

    assert {"timenet/hello-world", "physionet/ecg-qa-cot"} <= set(dataset_ids)
    assert "nope/nothing" not in dataset_ids
    assert leaf not in sys.modules


def test_a_card_that_cannot_be_read_raises_and_only_a_missing_connector_gives_none(tmp_path, monkeypatch):
    assert ConnectorBuilder().card("nope/nothing") is None
    (tmp_path / "dataset.yaml").write_text("dataset_id: [unclosed\n")
    monkeypatch.setattr(backend_module, "connector_dir", lambda dataset_id: tmp_path)

    with pytest.raises(TimeNetInvalidCardError, match="could not read dataset card"):
        ConnectorBuilder().card("test/broken")


def test_the_card_answers_only_to_the_dataset_id_that_it_declares(tmp_path, monkeypatch):
    # The module path of the connector accepts the spelling with "_", but a dataset ID must match its card.
    assert ConnectorBuilder().card("timenet/hello-world") is not None
    assert ConnectorBuilder().card("timenet/hello_world") is None
    monkeypatch.setattr(backend_module, "run_isolated_build", _unexpected_isolation)

    with pytest.raises(TimeNetDatasetNotFoundError, match="timenet/hello_world"):
        ConnectorBuilder().build(
            DatasetRef.parse("timenet/hello_world@1.0.0"),
            LocalRegistry(tmp_path),
            force=False,
            pinned_by={},
            options=BuildOptions(),
        )


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("dataset_id: [unclosed\n", "could not read dataset card {}"),
        ("- a list\n", "dataset card {} must be a YAML mapping"),
        ("title: no id\n", "dataset card {} must declare a dataset_id"),
    ],
)
def test_a_card_that_cannot_be_read_is_skipped_with_a_warning(tmp_path, monkeypatch, text, message):
    (tmp_path / "test" / "broken").mkdir(parents=True)
    (tmp_path / "test" / "broken" / "dataset.yaml").write_text(text)
    monkeypatch.setattr(datasets_module, "__path__", [*datasets_module.__path__, str(tmp_path)])
    expected = re.escape(message.format(tmp_path / "test" / "broken" / "dataset.yaml"))

    with pytest.warns(ConnectorCardWarning, match=expected):
        assert "timenet/hello-world" in known_ids()
    # The installed entry point loads the builder backend of this package, which lists its dataset IDs.
    with pytest.warns(ConnectorCardWarning, match=expected):
        found = installed_builder("timenet/hello-world")
    assert found is not None
    assert found.card.dataset_id == "timenet/hello-world"


def test_build_in_process_round_trips_the_connector_default_zarr_backend(tmp_path, monkeypatch):
    pytest.importorskip("zarr")
    connector_cls = backend_module.resolve("timenet/hello-world")
    monkeypatch.setattr(connector_cls, "values_backend", ValuesBackend.ZARR)
    monkeypatch.setenv("TIMENET_ISOLATION", "off")
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    store = LocalRegistry(tmp_path / "registry")

    ConnectorBuilder().build(HELLO, store, force=False, pinned_by={}, options=BuildOptions())

    version_dir = tmp_path / "registry/timenet/hello-world/1.0.0"
    assert Manifest.model_validate_json((version_dir / "manifest.json").read_text()).values_backend == "zarr"
    with TimeFReader(open_local_version(version_dir)) as reader:
        dataset = reader.read()
        assert len(dataset.records) == 3
        assert len(dataset.records[0].signals[0].to_numpy()) == 16


# Only BuildOptions.quiet silences the child. An SDK build without a progress callback still prints.
@pytest.mark.parametrize("quiet", [False, True])
def test_build_runs_an_isolated_child_with_the_release_and_the_pins(tmp_path, monkeypatch, quiet):
    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    monkeypatch.setattr(backend_module, "run_pipeline", _unexpected_pipeline)
    calls = []
    monkeypatch.setattr(
        backend_module, "run_isolated_build", lambda release, root, **options: calls.append((release, root, options))
    )

    ConnectorBuilder().build(
        HELLO,
        LocalRegistry(tmp_path),
        force=True,
        pinned_by=_pins(),
        options=BuildOptions(keep_cache=True, quiet=quiet),
    )

    assert calls == [(HELLO, tmp_path, {"force": True, "keep_cache": True, "quiet": quiet, "pinned_by": _pins()})]


def test_build_rejects_another_version(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_ISOLATION", "on")
    monkeypatch.setattr(backend_module, "run_isolated_build", _unexpected_isolation)
    monkeypatch.setattr(backend_module, "run_pipeline", _unexpected_pipeline)
    release = DatasetRef.parse("timenet/hello-world@9.9.9")

    with pytest.raises(TimeNetVersionUnavailableError) as caught:
        ConnectorBuilder().build(release, LocalRegistry(tmp_path), force=False, pinned_by={}, options=BuildOptions())

    # It is a TimeFValidationError too, so a handler of the old error still catches it.
    assert isinstance(caught.value, TimeFValidationError)
    assert (caught.value.release, str(caught.value.declared)) == (release, "1.0.0")


def test_build_reports_downloads_and_shards(tmp_path, monkeypatch):
    class Connector:
        pass

    monkeypatch.setenv("TIMENET_ISOLATION", "off")
    monkeypatch.setattr(backend_module, "resolve", lambda dataset_id: Connector)
    received = {}

    def pipeline(connector, root, *, progress_cb, **options):
        received.update(root=root, **options)
        report_progress(DownloadProgress("https://host/records.zip", 5, 10))
        progress_cb(WriteProgressEvent(ProgressStage.TIME_SERIES, 3, None))
        progress_cb(WriteProgressEvent(ProgressStage.SHARD_FINALIZED, 1, None))
        return root / "timenet/hello-world/1.0.0"

    monkeypatch.setattr(backend_module, "run_pipeline", pipeline)
    events = []

    ConnectorBuilder().build(
        HELLO,
        LocalRegistry(tmp_path),
        force=True,
        pinned_by=_pins(),
        options=BuildOptions(keep_cache=True, progress=events.append),
    )

    assert events == [BytesMoved("https://host/records.zip", 5, 10), ShardWritten(HELLO, 1)]
    expected = {"root": tmp_path, "force": True, "keep_cache": True, "pinned_by": _pins()}
    assert received == expected


def test_build_without_progress_installs_no_sink(tmp_path, monkeypatch):
    monkeypatch.setenv("TIMENET_ISOLATION", "off")
    seen = {}

    def pipeline(connector, root, *, progress_cb, **options):
        seen.update(progress_cb=progress_cb, sink=current_sink())
        return root / "timenet/hello-world/1.0.0"

    monkeypatch.setattr(backend_module, "run_pipeline", pipeline)

    # An outer sink must not reach the build either: the build installs its own, empty one.
    with progress_sink(print):
        ConnectorBuilder().build(
            HELLO, LocalRegistry(tmp_path), force=False, pinned_by={}, options=BuildOptions(progress=None)
        )

    assert seen == {"progress_cb": None, "sink": None}


def test_a_client_build_lets_the_isolated_child_print(tmp_path, monkeypatch):
    # The client sends no progress callback and leaves quiet unset. So the child shows its build lines
    # and downloads on the inherited stderr.
    monkeypatch.setenv("TIMENET_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("TIMENET_ISOLATION", raising=False)
    calls = []

    def child(release, store_root, *, quiet, **_):
        calls.append((release, quiet))
        LocalRegistry(store_root).put_dataset(make_dataset())

    monkeypatch.setattr(backend_module, "run_isolated_build", child)

    with TimeNet(tmp_path / "registry").open("timenet/hello-world") as reader:
        assert reader.read().records

    assert calls == [(HELLO, False)]
