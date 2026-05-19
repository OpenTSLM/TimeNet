# Connectors

The unit of dataset integration. One connector class = one dataset (for the moment, in the future this might change). A connector fetches raw data and converts it into a `TimeFDataset`. It has no knowledge of the registry, the engine, or any other connector.

---

## At a glance

```python
from abc import ABC, abstractmethod
from pathlib import Path
from collections.abc import Callable
from typing import Generic, TypeVar

from timenet.timef.dataset import TimeFDataset
from timenet.timef.metadata import DatasetMetadata
from timenet.timef.writer import TimeFWriter, WriteProgressEvent

TRaw = TypeVar("TRaw")


class BaseConnector(ABC, Generic[TRaw]):

    def __init__(self) -> None: ...

    @abstractmethod
    def metadata(self) -> DatasetMetadata: ...

    @abstractmethod
    def download(self, cache_dir: Path) -> list[TRaw]: ...

    @abstractmethod
    def convert(self, raw_refs: list[TRaw]) -> TimeFDataset: ...

    def store(
        self,
        dataset: TimeFDataset,
        root: Path,
        *,
        progress_cb: Callable[[WriteProgressEvent], None] | None = None,
    ) -> None: ...
```

Minimal example, 12-lead ECG dataset:

```python
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from timenet.connectors.base import BaseConnector
from timenet.domains import Domain
from timenet.licenses import License
from timenet.tasks import ClassificationTask
from timenet.timef.dataset import TimeSeries, TimeFDataset
from timenet.timef.metadata import (
    AnnotationSpec, DatasetMetadata, DeviceSpec, TimeSeriesSpec,
)
from timenet.units import Frequency, SamplingRateUnit, TimestampUnit, ValueUnit
from timenet.version import Version
from timenet.views import View


@dataclass(frozen=True)
class Recording:
    recording_file_name: str
    patient_id: str
    path: Path
    leads: tuple[str, ...]
    sampling_rate: float


def read_lead(path: Path, lead: str) -> np.ndarray:
    # parse the EDF and return the channel as a float32 1-D array
    ...


# Modality + device contracts. Define the TimeSeriesSpec subclass first, the
# DeviceSpec subclass second (listing its spec classes), then back-patch the
# spec's typed `device` ref once both classes exist.
class ECGLeadSpec(TimeSeriesSpec):
    spec_id = "ecg_lead"
    name = "ECG Lead"
    unit_sampling_rate = SamplingRateUnit.HZ
    unit_timestamp = TimestampUnit.SECONDS
    unit_value = ValueUnit.MILLIVOLT


class HolterX(DeviceSpec):
    device_id = "holter_x"
    name = "Holter Monitor X"
    manufacturer = "Acme"


class ECGConnector(BaseConnector[Recording]):

    METADATA = DatasetMetadata(
        dataset_id="ecg_dataset",
        version=Version(1, 0, 0),
        description="100 patients, one 12-lead ECG recording each.",
        license=License.CC_BY_4,
        domains=(Domain.CARDIOLOGY,),
        time_series_specs=(ECGLeadSpec,),
        device_specs=(HolterX,),
        annotation_specs=(
            AnnotationSpec(spec_id="rhythm_cls", task=ClassificationTask),
        ),
    )

    def metadata(self) -> DatasetMetadata:
        return self.METADATA

    def download(self, cache_dir: Path) -> list[Recording]:
        # I/O only: fetch files, return lightweight references
        ...

    def convert(self, raw_refs: list[Recording]) -> TimeFDataset:
        dataset = TimeFDataset(
            dataset_id=self.METADATA.dataset_id,
            version=self.METADATA.version,
        )
        for rec in raw_refs:
            sample = dataset.add_sample(
                sample_id=rec.recording_file_name,
                subject_ids=(rec.patient_id,),
                time_series=tuple(
                    TimeSeries(
                        spec=ECGLeadSpec(channel=lead),
                        source_id=rec.recording_file_name,
                        sampling_rate=Frequency.Hz(rec.sampling_rate),
                        reader=lambda p=rec.path, c=lead: read_lead(p, c),
                    )
                    for lead in rec.leads
                ),
                view=View.FULL,
            )
            dataset.add_annotation(
                sample,
                ClassificationTask(label="normal_sinus_rhythm"),
                spec_id="rhythm_cls",
            )
        return dataset
```

---

## Methods

### `__init__()`

No arguments. Reads environment variables and stores them on `self`. The registry instantiates every connector with `Cls()`. A connector that requires constructor arguments is rejected at registry load time.

Standard env vars to read in `__init__`:

| Variable              | Purpose                                                           |
| --------------------- | ----------------------------------------------------------------- |
| `TIMENET_TESTING`     | Set to `"1"` to enable testing mode (see below).                  |
| Dataset-specific vars | Source location, credentials, etc. Read with a hardcoded default. |

```python
def __init__(self) -> None:
    self.testing = os.environ.get("TIMENET_TESTING") == "1"
    self.bucket = os.environ.get("ECG_BUCKET", "s3://datasets/ecg/")
```

---

### `metadata()`

```python
@abstractmethod
def metadata(self) -> DatasetMetadata: ...
```

Returns the dataset's identity and classification. Called by the registry at startup, before any download has run.

**Returns:** `DatasetMetadata`. See [DatasetMetadata](#datasetmetadata) below.

**Constraints**

- Must be derivable from constructor state alone, no I/O, no runtime data.
- `metadata().dataset_id` must equal the YAML key under which the connector is registered. The registry asserts this on load.

---

### `download()`

```python
@abstractmethod
def download(self, cache_dir: Path) -> list[TRaw]: ...
```

I/O-only stage. Fetches or discovers raw source files and returns lightweight references to them. No parsing, no array work, no format decoding.

**Parameters**

| Name        | Type   | Default  | Description                                                                       |
| ----------- | ------ | -------- | --------------------------------------------------------------------------------- |
| `cache_dir` | `Path` | required | Directory to write downloaded files into. Created by the engine before this call. |

**Returns:** `list[TRaw]`. A list of raw references: paths, dataclass handles, HDF5 refs, or any type the connector defines. Passed directly to `convert()`.

**Constraints**

- Must be idempotent: re-running with the same `cache_dir` must produce the same result.
- In testing mode (`self.testing`), return fixtures from `cache_dir` rather than hitting the network.

---

### `convert()`

```python
@abstractmethod
def convert(self, raw_refs: list[TRaw]) -> TimeFDataset: ...
```

CPU-bound stage. Parses raw references and populates a `TimeFDataset`. No network I/O.

**Parameters**

| Name       | Type         | Default  | Description                        |
| ---------- | ------------ | -------- | ---------------------------------- |
| `raw_refs` | `list[TRaw]` | required | The list returned by `download()`. |

**Returns:** A `TimeFDataset`.

**Constraints**

- `dataset_id` and `version` on the returned `TimeFDataset` must match `metadata().dataset_id` and `metadata().version`.
- To share time-series data across samples, attach the **same** `TimeSeries` instance to each sample. The writer dedupes by Python object identity (`id()`).
- Each `TimeSeries.reader` should be a closure that captures whatever it needs (file path, file handle, S3 key, …), `store()` calls it with no arguments.
- In testing mode (`self.testing`), build readers that pull from fixtures instead of remote sources.

---

### `store()`

```python
def store(
    self,
    dataset: TimeFDataset,
    root: Path,
    *,
    progress_cb: Callable[[WriteProgressEvent], None] | None = None,
) -> None: ...
```

Walks `dataset.samples`, dedupes `TimeSeries` instances by identity, and streams them through a `TimeFWriter`. Commits the dataset to disk on success.

`store()` has a default implementation on `BaseConnector`. Most connectors do not need to override it.

**Parameters**

| Name          | Type                                           | Default  | Description                                                 |
| ------------- | ---------------------------------------------- | -------- | ----------------------------------------------------------- |
| `dataset`     | `TimeFDataset`                                 | required | The populated dataset returned by `convert()`.              |
| `root`        | `Path`                                         | required | Parent directory passed through to `TimeFWriter`.           |
| `progress_cb` | `Callable[[WriteProgressEvent], None] \| None` | `None`   | Forwarded to `TimeFWriter`. Called once per progress event. |

**Default body**

```python
def store(
    self,
    dataset: TimeFDataset,
    root: Path,
    *,
    progress_cb: Callable[[WriteProgressEvent], None] | None = None,
) -> None:
    with TimeFWriter(root, dataset, progress_cb=progress_cb) as writer:
        writer.write()
```

---

## `DatasetMetadata`

The connector's self-description. Declares identity, classification, and the specs that govern the dataset's contents (time series, annotations).

```python
from timenet.domains import Domain
from timenet.licenses import License
from timenet.timef.metadata import (
    AnnotationSpec, DeviceSpec, TimeSeriesSpec,
)
from timenet.version import Version


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    version: Version
    description: str
    license: License
    time_series_specs: tuple[type[TimeSeriesSpec], ...] = ()
    annotation_specs: tuple[AnnotationSpec, ...] = ()
    device_specs: tuple[type[DeviceSpec], ...] = ()
    domains: tuple[Domain, ...] = ()
    source_url: str | None = None
    tags: tuple[str, ...] = ()
```

**Fields**

| Name                | Type                               | Required | Description                                                                                                                                                                |
| ------------------- | ---------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_id`        | `str`                              | yes      | Snake-cased unique identifier. Must match the YAML registry key.                                                                                                           |
| `version`           | `Version`                          | yes      | Semantic version (`major.minor.patch`). See [Version](types.md#version).                                                                                                   |
| `description`       | `str`                              | yes      | One-sentence human-readable description.                                                                                                                                   |
| `license`           | `License`                          | yes      | Data license.                                                                                                                                                              |
| `time_series_specs` | `tuple[type[TimeSeriesSpec], ...]` | no       | Modality **types** the dataset records (channel, units, sampling rate, device). `TimeSeriesSpec` subclasses, not instances. See [TimeSeriesSpec](types.md#timeseriesspec). |
| `annotation_specs`  | `tuple[AnnotationSpec, ...]`       | no       | Task types the dataset annotates and their label schemas. See [AnnotationSpec](types.md#annotationspec).                                                                   |
| `device_specs`      | `tuple[type[DeviceSpec], ...]`     | no       | Device **types** that produced the modalities. `DeviceSpec` subclasses, not instances. See [DeviceSpec](types.md#devicespec).                                              |
| `domains`           | `tuple[Domain, ...]`               | no       | Clinical or application domains (e.g. `Domain.CARDIOLOGY`).                                                                                                                |
| `source_url`        | `str \| None`                      | no       | Canonical URL of the source dataset.                                                                                                                                       |
| `tags`              | `tuple[str, ...]`                  | no       | Free-form labels for filtering.                                                                                                                                            |

---

## Testing mode

When `TIMENET_TESTING=1`, `download()` must not hit the network. The standard pattern is to serve fixtures from `cache_dir`:

```python
def download(self, cache_dir: Path) -> list[Recording]:
    if self.testing:
        return [
            Recording(
                recording_file_name="fixture_001",
                patient_id="p_001",
                path=cache_dir / "fixture_001.edf",
                leads=LEADS,
                sampling_rate=500.0,
            )
        ]
    # real download ...
```

Fixture files live under `E2E_CACHE_DIR/<dataset_id>/` by convention.

---

## Design rules

- **One connector class, one dataset.**
- **No constructor arguments.** Configuration comes from environment variables with hardcoded defaults.
- **The connector decides what a sample is.** For a given dataset there is one connector and it is authoritative: which recordings become samples, which variables to expose, which views to create.
- **Time series are references, not data.** `convert()` builds `TimeSeries` instances with lazy `reader` callables. Bytes are pulled by the writer during `store()`, never held in memory by the dataset.
- **Share data by reusing instances.** Two samples that read identical bytes must reference the **same** `TimeSeries` object, the writer dedupes by Python identity.

PD: this design rule will be changed when we support multiple connectors per dataset
