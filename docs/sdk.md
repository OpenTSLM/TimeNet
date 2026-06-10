# Client

Single Python entry point for using TimeNet from code. Wraps the registry, engine, reader, and explorer into one surface.

---

## At a glance

```python
from pathlib import Path
from collections.abc import Callable
from typing import Literal

from timenet.models import DatasetCollection, DownloadProgressEvent, DownloadReport, QueryCriteria
from timenet.timef.metadata import DatasetMetadata
from timenet.sdk.dataset import Dataset
from timenet.timef.dataset import Sample
from timenet.domains import Domain
from timenet.tasks import Task
from timenet.licenses import License


class TimeNet:

    def __init__(self, config: str | Path | None = None) -> None: ...

    def list(self, criteria: QueryCriteria | None = None) -> list[DatasetMetadata]: ...

    def query(
        self,
        domains: list[Domain] | None = None,
        tasks: list[type[Task]] | None = None,
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
        force: bool = False,
        io_workers: int = 16,
        cpu_workers: int = 4,
        progress_cb: Callable[[DownloadProgressEvent], None] | None = None,
    ) -> DownloadReport: ...

    def datasets(self, data_root: Path | None = None) -> list[Dataset]: ...

    def query_samples(
        self,
        dataset_id: str,
        version: str,
        data_root: Path | None = None,
        task: type[Task] | None = None,
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

```python
from timenet import TimeNet, Domain
from timenet.tasks import ClassificationTask

client = TimeNet()                        # default catalog only
client = TimeNet(config="datasets.yaml")  # default + custom catalog

collection = client.query(domains=[Domain.CARDIOLOGY], tasks=[ClassificationTask])
report     = client.download(collection, target_dir="~/.timenet/processed")
```

---

## Methods

### `__init__()`

```python
def __init__(self, config: str | Path | None = None) -> None: ...
```

Builds the registry from the library's built-in YAML catalog, optionally extended by a custom catalog. See [Registry](registry.md).

**Parameters**

| Name     | Type                  | Default | Description                                                                            |
| -------- | --------------------- | ------- | -------------------------------------------------------------------------------------- |
| `config` | `str \| Path \| None` | `None`  | Path to a custom YAML catalog. Merged after the default, cannot redefine existing IDs. |

**Raises**

| Exception           | Condition                                                                                                        |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `FileNotFoundError` | `config` is provided and the file does not exist.                                                                |
| `yaml.YAMLError`    | A YAML file is malformed.                                                                                        |
| `ImportError`       | The module portion of a connector's `class:` value cannot be imported.                                           |
| `AttributeError`    | The class portion of a `class:` value is not present on the imported module.                                     |
| `TypeError`         | A connector class requires constructor arguments.                                                                |
| `ValueError`        | A connector's `metadata().dataset_id` does not match its YAML key, or a duplicate ID is found across both files. |

---

### `list()`

```python
def list(self, criteria: QueryCriteria | None = None) -> list[DatasetMetadata]: ...
```

Returns every registered dataset. If `criteria` is provided, filters before returning.

**Parameters**

| Name       | Type                    | Default | Description                                        |
| ---------- | ----------------------- | ------- | -------------------------------------------------- |
| `criteria` | `QueryCriteria \| None` | `None`  | Filter specification. `None` returns all datasets. |

**Returns:** `list[DatasetMetadata]`, one entry per registered dataset matching `criteria`.

---

### `query()`

```python
def query(
    self,
    domains: list[Domain] | None = None,
    tasks: list[type[Task]] | None = None,
    license: License | None = None,
    signals: list[str] | None = None,
    min_length_s: float | None = None,
    source: str | None = None,
    dataset_ids: list[str] | None = None,
    tags: list[str] | None = None,
) -> DatasetCollection: ...
```

Builds a `QueryCriteria` from the given filters and runs it against the registry.

**Parameters**

| Name           | Type                       | Default | Description                                                                                     |
| -------------- | -------------------------- | ------- | ----------------------------------------------------------------------------------------------- |
| `domains`      | `list[Domain] \| None`     | `None`  | Keep datasets that include any of these domains.                                                |
| `tasks`        | `list[type[Task]] \| None` | `None`  | Keep datasets that support any of these task types (pass the class, e.g. `ClassificationTask`). |
| `license`      | `License \| None`          | `None`  | Keep datasets with this exact license.                                                          |
| `signals`      | `list[str] \| None`        | `None`  | Keep datasets that declare all of these `SignalSpec.spec_id` values.                            |
| `min_length_s` | `float \| None`            | `None`  | Keep datasets whose minimum recording length meets this threshold (seconds).                    |
| `source`       | `str \| None`              | `None`  | Keep datasets from this source label.                                                           |
| `dataset_ids`  | `list[str] \| None`        | `None`  | Keep only these specific dataset IDs.                                                           |
| `tags`         | `list[str] \| None`        | `None`  | Keep datasets that carry all of these tags.                                                     |

**Returns:** `DatasetCollection` containing the matching descriptors. Pass directly to `download()`.

**Example**

```python
from timenet.tasks import ClassificationTask

collection = client.query(
    domains=[Domain.CARDIOLOGY],
    tasks=[ClassificationTask],
    license=License.CC_BY_4,
)
```

---

### `download()`

```python
def download(
    self,
    collection: DatasetCollection,
    target_dir: Path | None = None,
    io_workers: int = 16,
    cpu_workers: int = 4,
    force: bool = False,
    progress_cb: Callable[[DownloadProgressEvent], None] | None = None,
) -> DownloadReport: ...
```

Runs the engine over every dataset in `collection`, downloading and converting each one.

**Parameters**

| Name          | Type                                              | Default  | Description                                                                        |
| ------------- | ------------------------------------------------- | -------- | ---------------------------------------------------------------------------------- |
| `collection`  | `DatasetCollection`                               | required | Output of `query()` or `list()`.                                                   |
| `target_dir`  | `Path \| None`                                    | `None`   | Root output directory. Defaults to `$TIMENET_DATA_ROOT` or `~/.timenet/processed`. |
| `io_workers`  | `int`                                             | `16`     | Workers for downloading datasets.                                                  |
| `cpu_workers` | `int`                                             | `4`      | Workers for converting datasets.                                                   |
| `force`       | `bool`                                            | `False`  | Re-download and re-convert even when an up-to-date manifest already exists.        |
| `progress_cb` | `Callable[[DownloadProgressEvent], None] \| None` | `None`   | Optional progress callback. Invoked from worker threads, must be thread-safe.      |

**Returns:** `DownloadReport`. One entry per dataset with `dataset_id`, `version`, `status` (`"ok" | "skipped" | "failed"`), `output_path`, and optional `error`.

**Side effects:** Writes to `<target_dir>/<dataset_id>/<version>/`. Creates parent directories as needed.

**Raises**

| Exception             | Condition                                       |
| --------------------- | ----------------------------------------------- |
| `RegistryLookupError` | A descriptor in `collection` is not registered. |

**Example**

```python
report = client.download(collection, io_workers=16, cpu_workers=4)

for entry in report:
    if entry.status == "failed":
        print(entry.dataset_id, entry.error)
```

---

### `datasets()`

```python
def datasets(self, data_root: Path | None = None) -> list[Dataset]: ...
```

Scans `data_root` for `manifest.json` files and returns a `Dataset` wrapper for each one found.

**Parameters**

| Name        | Type           | Default | Description                                                                    |
| ----------- | -------------- | ------- | ------------------------------------------------------------------------------ |
| `data_root` | `Path \| None` | `None`  | Directory to scan. Defaults to `$TIMENET_DATA_ROOT` or `~/.timenet/processed`. |

**Returns:** `list[Dataset]`. Manifests are loaded into memory; signal data is not read until accessed.

**Raises**

| Exception           | Condition                   |
| ------------------- | --------------------------- |
| `FileNotFoundError` | `data_root` does not exist. |

---

### `query_samples()`

```python
def query_samples(
    self,
    dataset_id: str,
    version: str,
    data_root: Path | None = None,
    task: type[Task] | None = None,
    domains: list[Domain] | None = None,
) -> list[Sample]: ...
```

Filters the samples of one local dataset by task or domain.

**Parameters**

| Name         | Type                   | Default  | Description                                                                                                                                                       |
| ------------ | ---------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_id` | `str`                  | required | ID of the dataset to query.                                                                                                                                       |
| `version`    | `str`                  | required | Version string of the dataset.                                                                                                                                    |
| `data_root`  | `Path \| None`         | `None`   | Root directory to look in. Defaults to `$TIMENET_DATA_ROOT` or `~/.timenet/processed`.                                                                            |
| `task`       | `type[Task] \| None`   | `None`   | Keep samples that have at least one annotation whose `task` is an instance of this class (pass the class, e.g. `ClassificationTask`).                             |
| `domains`    | `list[Domain] \| None` | `None`   | Keep samples whose dataset declares any of these domains in its `metadata().domains`. The filter is dataset-level, applied to every sample of a matching dataset. |

**Returns:** `list[Sample]` matching all provided filters.

**Raises**

| Exception              | Condition                                                          |
| ---------------------- | ------------------------------------------------------------------ |
| `DatasetNotFoundError` | `<data_root>/<dataset_id>/<version>/manifest.json` does not exist. |

---

### `explorer.serve()`

```python
@staticmethod
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    data_root: Path | None = None,
) -> None: ...
```

Starts the Explorer web server bound to `data_root`.

**Parameters**

| Name        | Type           | Default       | Description                                                                        |
| ----------- | -------------- | ------------- | ---------------------------------------------------------------------------------- |
| `host`      | `str`          | `"127.0.0.1"` | Interface to bind to.                                                              |
| `port`      | `int`          | `8000`        | Port to listen on.                                                                 |
| `data_root` | `Path \| None` | `None`        | Dataset root to serve. Defaults to `$TIMENET_DATA_ROOT` or `~/.timenet/processed`. |

**Raises**

| Exception           | Condition                                      |
| ------------------- | ---------------------------------------------- |
| `FileNotFoundError` | `data_root` does not exist.                    |
| `OSError`           | The `host:port` combination is already in use. |

---

## Types

The SDK exposes three runtime data types beyond what is defined elsewhere in the spec. They are returned or consumed by the methods above. For the input filter type `QueryCriteria` see [Registry](registry.md#querycriteria).

---

### `DatasetCollection`

The output of `query()` and `list()`, and the input to `download()`. A typed bag of `DatasetMetadata` entries.

```python
from collections.abc import Iterator
from dataclasses import dataclass

from timenet.timef.metadata import DatasetMetadata


@dataclass(frozen=True)
class DatasetCollection:
    items: tuple[DatasetMetadata, ...]

    def __iter__(self) -> Iterator[DatasetMetadata]: ...
    def __len__(self) -> int: ...
```

---

### `DownloadProgressEvent`

Payload passed to the `progress_cb` callback during `download()`. Emitted multiple times per dataset as it moves through its stages.

```python
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DownloadProgressEvent:
    dataset_id: str
    stage: Literal["download", "convert", "write"]
    completed: int
    total: int
    message: str | None = None
```

| Field                 | Description                                                                                                            |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `dataset_id`          | Which dataset the event is about.                                                                                      |
| `stage`               | Pipeline stage. `download` = fetching bytes, `convert` = parsing into a `TimeFDataset`, `write` = serializing to disk. |
| `completed` / `total` | Stage-specific units of work. For `download` they are bytes; for `convert` and `write` they are samples.               |
| `message`             | Optional human-readable detail (e.g. current file name, error retry message).                                          |

---

### `DownloadReport`

The return value of `download()`. One entry per dataset in the input collection, including those that failed.

```python
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class DownloadEntry:
    dataset_id: str
    version: str
    status: Literal["ok", "skipped", "failed"]
    output_path: Path | None
    error: str | None = None


@dataclass(frozen=True)
class DownloadReport:
    entries: tuple[DownloadEntry, ...]

    def __iter__(self) -> Iterator[DownloadEntry]: ...
    def __len__(self) -> int: ...
    def successes(self) -> tuple[DownloadEntry, ...]: ...
    def failures(self) -> tuple[DownloadEntry, ...]: ...
```

**Status values**

| Status      | Meaning                                                   | `output_path`          | `error` |
| ----------- | --------------------------------------------------------- | ---------------------- | ------- |
| `"ok"`      | The dataset was downloaded and converted in this run.     | set                    | `None`  |
| `"skipped"` | An up-to-date manifest already existed and `force=False`. | points to existing dir | `None`  |
| `"failed"`  | Something went wrong during download or conversion.       | `None`                 | set     |

Per-dataset errors are surfaced here, not raised. The whole `download()` call only raises for setup-level errors like an unregistered descriptor.

---

## `Dataset`

A class wrapping a single dataset that has already been **downloaded to disk**. Returned by `datasets()`.

```python
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from timenet.timef.dataset import Sample, Annotation
from timenet.timef.metadata import DatasetMetadata


class Dataset:
    metadata: DatasetMetadata
    root: Path

    def samples(self) -> Iterator[Sample]: ...
    def annotations(self) -> Iterator[Annotation]: ...

    def signal(
        self,
        sample_id: str,
        spec_id: str,
        channel: str,
    ) -> pd.DataFrame: ...
```

`Dataset.samples()` yields `Sample` objects; each carries the `annotation_ids` linking it to entries from `Dataset.annotations()`.

`Dataset.signal(sample_id, spec_id, channel)` reconstructs the values for a single `(sample_id, spec_id, channel)` triple. It filters `signal_index.parquet` by that key, reads each referenced chunk from the appropriate shard, and concatenates them in `chunk_idx` order. Returns a DataFrame with columns `t_s: float64` and `value: float32`; timestamps are derived from `t_start_s + i / sampling_rate_hz` unless the chunk carries an explicit `timestamps` column.

**Raises**

| Exception  | Condition                                                                                             |
| ---------- | ----------------------------------------------------------------------------------------------------- |
| `KeyError` | `sample_id` is not in `samples.parquet`, or `(spec_id, channel)` is not one of its `signals` entries. |
