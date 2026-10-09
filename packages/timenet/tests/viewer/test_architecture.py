"""Service isolation, observability, and registry contracts."""

import asyncio
from threading import get_ident
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from timenet.client import TimeNet
from timenet.errors import TimeFValidationError
from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.testing import make_dataset
from timenet.viewer.app import create_app
from timenet.viewer.inspection import ViewerInspection
from timenet.viewer.jobs import JobManager
from timenet.viewer.path_registry import PathRegistry
from timenet.viewer.schemas import RecordDetailQuery, TaskDetailQuery
from timenet.viewer.workers import ReaderWorker
from timenet.writer import TimeFWriter


@pytest.fixture
def reader(tmp_path):
    dataset = make_dataset()
    with TimeFWriter(tmp_path, dataset) as writer:
        writer.write()
    with TimeNet(tmp_path).open_reader(dataset.metadata.dataset_id) as opened:
        yield opened


def test_inspection_services_work_without_http(reader):
    inspection = ViewerInspection(reader)
    detail = inspection.record_detail(RecordDetailQuery(record_id="record-0"))
    assert detail["record_id"] == "record-0"
    assert inspection.overview()["identity"]
    task = inspection.task_detail(TaskDetailQuery(task_id="task-answer-0"))
    assert task is not None
    assert task["parent_task_ids"] == ["task-cls-0"]
    assert inspection.task_detail(TaskDetailQuery(task_id="absent")) is None


def test_worker_owns_creation_calls_and_close(reader, monkeypatch):
    calls = []
    original_init, original_close = TimeFReader.__init__, TimeFReader.close

    def init(self, *args, **kwargs):
        calls.append(("create", get_ident()))
        original_init(self, *args, **kwargs)

    def close(self):
        calls.append(("close", get_ident()))
        original_close(self)

    monkeypatch.setattr(TimeFReader, "__init__", init)
    monkeypatch.setattr(TimeFReader, "close", close)
    worker = ReaderWorker(reader, "test-owner")
    try:
        with pytest.raises(TimeFValidationError, match="owning worker"):
            worker.run(ViewerInspection.overview)

        def query(inspection):
            calls.append(("query", get_ident()))
            return inspection.records_page(limit=1)

        async def requests():
            return await asyncio.gather(worker.call(query), worker.call(query))

        assert all(page.items for page in asyncio.run(requests()))
    finally:
        worker.close()
    assert [name for name, _ in calls] == ["create", "query", "query", "close"]
    assert len({thread for _, thread in calls}) == 1
    assert calls[0][1] != get_ident()


def test_unexpected_job_failure_logs_correlation_without_leaking_http_details(reader, caplog):
    manager = JobManager()

    def fail():
        raise RuntimeError("internal storage failure")

    try:
        job = manager.submit(fail, operation="test-scan")
        assert job.future is not None
        job.future.result(timeout=5)
        assert job.state == "failed"
        assert job.error == "operation failed"
        record = next(record for record in caplog.records if getattr(record, "job_id", None) == job.job_id)
        assert record.operation == "test-scan"
        assert record.exc_info is not None
        assert "internal storage failure" in caplog.text
    finally:
        manager.close()


def test_path_registry_enumerates_only_root_and_delegates_pinned_parents(reader, tmp_path):
    local = LocalRegistry(tmp_path)
    root = local.open_version(reader.metadata.dataset_id, str(reader.metadata.dataset_version))
    parents = Mock()
    parents.list_datasets.side_effect = AssertionError("parent catalog must not be enumerated")
    overlay = PathRegistry(root, parents)
    assert overlay.list_datasets() == [root.manifest.metadata]
    assert overlay.open_version(root.manifest.dataset_id) is root
    assert overlay.get_manifest(root.manifest.dataset_id) is root.manifest
    assert overlay.open_version("parent/data", "1.0.0") is parents.open_version.return_value
    parents.open_version.assert_called_once_with("parent/data", "1.0.0")
    assert overlay.get_manifest("parent/data", "1.0.0") is parents.get_manifest.return_value
    assert overlay.open_file("parent/data", "1.0.0", "control.duckdb") is parents.open_file.return_value


def test_frontend_shell_css_and_missing_objects(reader):
    with TestClient(create_app(reader, token="test", port=8000), base_url="http://127.0.0.1:8000") as client:
        shell = client.get("/")
        assert shell.status_code == 200
        assert "<style>" not in shell.text
        assert "/static/app.css" in shell.text
        assert client.get("/static/app.css").headers["content-type"].startswith("text/css")
        assert client.get("/static/unknown.js").status_code == 404
        assert "frame-ancestors 'none'" in shell.headers["content-security-policy"]
        missing = client.post(
            "/api/v1/tasks/detail", json={"task_id": "absent"}, headers={"Authorization": "Bearer test"}
        )
        assert missing.status_code == 404


def test_view_rejects_a_registry_path_with_actionable_cli_guidance():
    from typer.testing import CliRunner  # noqa: PLC0415

    from timenet.cli.app import app  # noqa: PLC0415

    result = CliRunner().invoke(app, ["view", "registry/chengsenwang/tsqa", "--no-browser"])
    assert result.exit_code == 2
    assert "org/name" in result.output
    assert "--registry" in result.output
    assert "--path" in result.output
    assert "Traceback" not in result.output
    assert "ValidationError" not in result.output
