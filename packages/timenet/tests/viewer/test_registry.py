"""Catalog search and dataset selection against a real local registry."""

import json

from timenet.reader import TimeFReader
from timenet.registry import LocalRegistry
from timenet.types import Domain, ForecastingTask, License
from timenet.viewer.workers import ReaderWorker


def test_open_failure_preserves_selection_and_close_releases_it(registry_client, stored, monkeypatch):
    readers, closed_readers, workers, closed_workers = [], [], [], []
    reader_init, reader_close = TimeFReader.__init__, TimeFReader.close
    worker_init, worker_close = ReaderWorker.__init__, ReaderWorker.close

    def init_reader(self, *args, **kwargs):
        reader_init(self, *args, **kwargs)
        readers.append(self)

    def close_reader(self):
        reader_close(self)
        closed_readers.append(self)

    def init_worker(self, *args, **kwargs):
        worker_init(self, *args, **kwargs)
        workers.append(self)

    def close_worker(self):
        worker_close(self)
        closed_workers.append(self)

    monkeypatch.setattr(TimeFReader, "__init__", init_reader)
    monkeypatch.setattr(TimeFReader, "close", close_reader)
    monkeypatch.setattr(ReaderWorker, "__init__", init_worker)
    monkeypatch.setattr(ReaderWorker, "close", close_worker)
    _, dataset_id = stored
    assert registry_client.get("/api/v1/session").json()["mode"] == "registry"
    opened = registry_client.post("/api/v1/registry/open", json={"dataset_id": dataset_id})
    assert opened.status_code == 200
    assert registry_client.get("/api/v1/session").json()["dataset_id"] == dataset_id
    assert registry_client.post("/api/v1/registry/open", json={"dataset_id": "missing/dataset"}).status_code == 404
    assert registry_client.get("/api/v1/session").json()["dataset_id"] == dataset_id
    assert registry_client.get("/api/v1/overview").status_code == 200
    assert len(readers) == 2
    assert len(workers) == 1
    assert not closed_readers and not closed_workers
    assert registry_client.post("/api/v1/registry/close", json={}).status_code == 200
    assert len(closed_readers) == len(readers)
    assert set(closed_readers) == set(readers)
    assert closed_workers == workers
    assert registry_client.get("/api/v1/session").json()["mode"] == "registry"
    assert registry_client.post("/api/v1/records/query", json={}).status_code == 422


def test_catalog_search_and_facets_include_later_pages(registry_client, stored):
    directory, dataset_id = stored
    template = LocalRegistry(directory).get_manifest(dataset_id)
    for index in range(51):
        identity = f"search/item-{index:02}"
        metadata = template.metadata.model_copy(update={"dataset_id": identity, "name": f"Dataset {index}"})
        if index == 50:
            metadata = metadata.model_copy(update={"domains": (Domain.AUDIO,), "license": License.MIT})
        manifest = template.model_copy(update={"dataset_id": identity, "metadata": metadata})
        if index == 50:
            manifest = manifest.model_copy(
                update={"dataset_schema": template.dataset_schema.model_copy(update={"tasks": (ForecastingTask,)})}
            )
        path = directory / identity / str(metadata.dataset_version) / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(manifest.model_dump_json())
    first = registry_client.get("/api/v1/registry/datasets").json()
    second = registry_client.get("/api/v1/registry/datasets", params={"after": first["next_cursor"]}).json()
    found = registry_client.get("/api/v1/registry/datasets", params={"query": "Dataset 50"}).json()
    assert len(first["items"]) == 50
    assert len(second["items"]) == 2
    assert second["next_cursor"] is None
    expected_facets = {
        "domains": sorted({*(domain.value for domain in template.metadata.domains), "audio"}),
        "tasks": sorted({*(str(task.task_type) for task in template.dataset_schema.tasks), "forecasting"}),
        "licenses": sorted({str(template.metadata.license), "MIT"}),
    }
    assert first["facets"] == second["facets"] == found["facets"] == expected_facets
    assert [item["dataset_id"] for item in found["items"]] == ["search/item-50"]
    assert found["total_matches"] == 1


def test_catalog_reports_corruption_and_ignores_staging_versions(registry_client, stored):
    directory, dataset_id = stored
    original = directory / dataset_id / "1.0.0" / "manifest.json"
    raw = json.loads(original.read_text())
    for version in ("9.0.0.tmp-build", ".hidden"):
        target = original.parent.parent / version
        target.mkdir()
        (target / "manifest.json").write_text(json.dumps(raw))
    broken = directory / "test/broken/1.0.0"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text("{}")
    response = registry_client.get("/api/v1/registry/datasets")
    assert response.status_code == 200
    page = response.json()
    assert [(item["dataset_id"], item["version"]) for item in page["items"]] == [(dataset_id, "1.0.0")]
    assert [item["dataset_id"] for item in page["issues"]] == ["test/broken"]
