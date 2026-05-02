# Client

The **client** is the single Python entry point for everyone who uses TimeNet from code. It wraps the registry, the query engine, the reader, and the explorer into one surface.

---

## At a glance

```python
from pathlib import Path
from collections.abc import Callable
from typing import Literal

from timenet.models import DatasetDescriptor, DatasetCollection, DownloadProgressEvent, DownloadReport, QueryCriteria
from timenet.sdk.dataset import Dataset
from timenet.timef.schema import Sample
from timenet.domains import Domain
from timenet.tasks import Task
from timenet.licenses import License


class TimeNet:

    def __init__(self, config: str | Path | None = None) -> None: ...

    def list(self, criteria: QueryCriteria | None = None) -> list[DatasetDescriptor]: ...

    def query(
        self,
        domains: list[Domain] | None = None,
        tasks: list[Task] | None = None,
        license: License | None = None,
        signals: list[str] | None = None,
        min_length_s: float | None = None,
        source: str | None = None,
        dataset_ids: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> DatasetCollection: ...

    def download(
        self,
        collection: DatasetCollection,
        target_dir: Path | None = None,
        workers: int = 4,
        force: bool = False,
        progress_cb: Callable[[DownloadProgressEvent], None] | None = None,
        convert_executor: Literal["auto", "thread", "process"] = "auto",
    ) -> DownloadReport: ...

    def datasets(self, data_root: Path | None = None) -> list[Dataset]: ...

    def query_samples(
        self,
        dataset_id: str,
        version: str,
        data_root: Path | None = None,
        task: Task | None = None,
        domains: list[Domain] | None = None,
    ) -> list[Sample]: ...

    class explorer:
        @staticmethod
        def serve(
            host: str = "127.0.0.1",
            port: int = 8000,
            data_root: Path | None = None,
        ) -> None: ...
```

How the client is built and used:

```python
from timenet import TimeNet, Domain, Task

client = TimeNet()                        # default catalog only
client = TimeNet(config="datasets.yaml")  # default + custom catalog

collection = client.query(domains=[Domain.CARDIOLOGY], tasks=[Task.CLASSIFICATION])
report     = client.download(collection, target_dir="~/.timenet/processed")

```

---

## Methods

### `__init__(config)` { data-toc-label='init()' }

Builds the registry from the library's default YAML catalog, optionally extended by a custom YAML. See [Registry](registry.md)

### `list(criteria)` { data-toc-label='list()' }

Returns every registered dataset as a list of `DatasetDescriptor`. If `criteria` is provided, filters before returning.

### `query(domains, tasks, license, signals, min_length_s, source, dataset_ids, tags)` { data-toc-label='query()' }

Builds a `QueryCriteria` from the given parameters and runs it against the registry. Returns a `DatasetCollection`.

### `download(collection, target_dir, workers, resume, force, progress_cb, convert_executor)` { data-toc-label='download()' }

Runs the engine over every dataset in `collection`. Returns a `DownloadReport` with each dataset's outcome. Files land at `<target_dir>/<dataset_id>/<version>/`.

### `datasets(data_root)` { data-toc-label='datasets()' }

Scans `data_root` for `manifest.json` files and returns a `Dataset` wrapper for each one found. Manifests are loaded into memory; signal data is not read until iterated.

### `query_samples(dataset_id, version, data_root, task, domains)` { data-toc-label='query_samples()' }

Filters the contents of one local dataset by annotation task or domain and returns matching `Sample` objects.

### `explorer.serve(host, port, data_root)` { data-toc-label='explorer.serve()' }

Starts the Explorer FastAPI server on `host:port` bound to `data_root`. The `timenet explorer` CLI calls this directly; a downstream service can call it to embed the catalog browser in its own process.
