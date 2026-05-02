# Connectors

A **connector** is a self-contained Python class that knows what its dataset is, how to fetch it, and how to convert it into TimeF. It knows nothing about the rest of the system. It is the only way TimeNet learns about a data source.

Two design rules:

1. **One connector class is one dataset.** The class is constructed with no arguments and reports a single dataset id
2. **Self-contained.** The constructor reads the connector's own configuration (environment variables, defaults, the testing flag) and never receives parameters from the registry.

---

## At a glance

```python
import os
from pathlib import Path

from timenet.connectors.base import DatasetConnector, ConnectorExecutionContext
from timenet.timef.builder import TimeFDataset
from timenet.models import (
    DatasetDescriptor, SignalDescriptor,
    QueryCriteria, RawDatasetRef,
)
from timenet.domains import Domain
from timenet.licenses import License
from timenet.tasks import Task


class HolterConnector(DatasetConnector):
    connector_version = "1.0"

    def __init__(self) -> None:
        self.testing = os.environ.get("TIMENET_TESTING") == "1"
        self.bucket = os.environ.get(
            "HOLTER_BUCKET",
            "s3://datasets-prod/holter/",
        )
        self.id = "internal_holter_corpus"

    def id(self) -> str:
        return self.id

    def metadata(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            dataset_id=self.id(),
            version="1.0.0",
            source="internal/holter",
            domains=[Domain.CARDIOLOGY],
            license=License.MIT,
            signals=[SignalDescriptor(
                signal_id="ecg",
                name="ECG",
                sampling_rate=360.0,
                unit_sampling_rate="Hz",
                unit_timestamp="s",
                unit="mV",
                channels=["MLII", "V1"],
            )],
        )

    def download(
        self,
        cache_dir: Path,
        ctx: ConnectorExecutionContext,
    ) -> RawDatasetRef | None:
        if self.testing:
            return _serve_from_fixture(self.id(), cache_dir)
        # …fetch from self.bucket, write into cache_dir…
        return RawDatasetRef(
            dataset_id=self.id(),
            local_path=cache_dir,
            source_fingerprint="…",
        )

    def convert_to_timef(
        self,
        raw_ref: RawDatasetRef | None,
        dataset: TimeFDataset,
        ctx: ConnectorExecutionContext,
    ) -> None:
        for record in iter_records(raw_ref):
            sample = dataset.add_sample(signals={"ecg": record.frame})
            for beat in record.beats:
                sample.annotate(
                    task=Task.QUESTION_AND_ANSWER,
                    domains=[Domain.CARDIOLOGY],
                    signals=["ecg"],
                    question="What type of beat occurs in this window?",
                    answer=beat.symbol,
                    windows=[(beat.t_s - 0.21, beat.t_s + 0.21)],
                )
```

---

## The contract

### `connector_version: str`

A class attribute separate from the `metadata().version` (which is the dataset's version).

### `__init__(self) -> None` { data-toc-label='init()' }

No arguments. The constructor's only job is to read whatever environment variables the connector needs and put them on `self`. Two important values for every connector:

- `self.testing = os.environ.get("TIMENET_TESTING") == "1"`
- Source location(s): `self.url`

### `id(self) -> str` { data-toc-label='id()' }

Returns the dataset id, snake-cased string (e.g. `"mit_bih_arrhythmia"`, `"synthetic_ecg"`). Must be identical to:

- the YAML key under which the connector is registered, and
- `metadata().dataset_id`.

### `metadata(self) -> DatasetDescriptor` { data-toc-label='metadata()' }

Returns the descriptor used by the Registry. The descriptor declares what the dataset is: id, version, source label, domains, tasks, license, tags, and the full signal list.

`metadata()` must be derivable from constructor state alone. The registry calls it at startup, before any `download` has run, so it cannot depend on data observed at runtime.

### `download(self, cache_dir, ctx) -> RawDatasetRef | None` { data-toc-label='download()' }

The I/O-bound stage. Runs in a thread on the engine's I/O pool. Does I/O only: fetching bytes, writing them to `cache_dir`, checking the cache. No parsing, no array work, no format decoding.

Two return shapes:

| Return               | Meaning                                                                        |
| -------------------- | ------------------------------------------------------------------------------ |
| `RawDatasetRef(...)` | Connector kept a raw cache on disk                                             |
| `None`               | Connector streams directly to `convert_to_timef` without persisting raw bytes. |

Connectors emit progress through `ctx.progress(...)` during downloads. These events flow back to the CLI/Explorer.

### `convert_to_timef(self, raw_ref, dataset, ctx) -> None` { data-toc-label='convert_to_timef()' }

Runs in a process on the engine's CPU pool.

The connector receives a `TimeFDataset` and populates it via `dataset.add_sample(...)` and `sample.annotate(...)`. See [TimeFDataset](timef-builder.md) for the full builder contract.

The connector stores what the source publishes, **one source recording = one stored sample**. For MIT-BIH that is one sample per `.hea` record. For PSG, one sample per overnight `.edf`. TODO:Slicing will be handled later by the data-loader layer.

`raw_ref` is whatever `download()` returned. If `download()` returned `None`, this method is responsible for producing data on the fly.

---

## Testing mode

When `TIMENET_TESTING=1` is set, `__init__` records the flag and the connector chooses test-mode behavior on every call.
The most common pattern is a fixture under `E2E_CACHE_DIR`:

```python
def download(self, cache_dir, ctx):
    if self.testing:
        fixture = E2E_CACHE_DIR / self.id()
        return RawDatasetRef(
            dataset_id=self.id(),
            local_path=fixture,
            source_fingerprint="fixture",
        )
    # …real download…
```

Other valid choices:

- Skip `download()` entirely (return `None`) and have `convert_to_timef` synthesize a tiny dataset.

---
