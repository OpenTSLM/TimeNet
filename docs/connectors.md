# Connectors

The unit of dataset integration. One connector class = one dataset (for the moment, in the future this might change). A connector fetches raw data and converts it into a `TimeFDataset`. It has no knowledge of the registry, the engine, or any other connector.

---

## At a glance

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from timenet.timef.dataset import TimeFDataset
from timenet.timef.metadata import DatasetMetadata

TRaw = TypeVar("TRaw")


class BaseConnector(ABC, Generic[TRaw]):

    def __init__(self) -> None: ...

    @abstractmethod
    def metadata(self) -> DatasetMetadata: ...

    @abstractmethod
    def download(self, cache_dir: Path) -> list[TRaw]: ...

    @abstractmethod
    def convert(self, raw_refs: list[TRaw]) -> TimeFDataset: ...
```

Minimal example, 12-lead ECG dataset:

```python
from dataclasses import dataclass
from pathlib import Path

from timenet.connectors.base import BaseConnector
from timenet.domains import Domain
from timenet.licenses import License
from timenet.tasks import ClassificationTask
from timenet.timef.dataset import TimeFDataset, SignalRef
from timenet.timef.metadata import (
    AnnotationSpec, DatasetMetadata, SignalSpec, ViewSpec,
)
from timenet.units import SamplingRateUnit, TimestampUnit, ValueUnit
from timenet.version import Version


@dataclass(frozen=True)
class Recording:
    recording_id: str
    patient_id: str
    path: Path
    leads: tuple[str, ...]


class ECGConnector(BaseConnector[Recording]):

    METADATA = DatasetMetadata(
        dataset_id="ecg_dataset",
        version=Version(1, 0, 0),
        description="100 patients, one 12-lead ECG recording each.",
        license=License.CC_BY_4,
        domains=(Domain.CARDIOLOGY,),
        signal_specs=(
            SignalSpec(
                spec_id="ecg_12lead",
                name="ECG",
                channels=("I", "II", "III", "aVR", "aVL", "aVF",
                          "V1", "V2", "V3", "V4", "V5", "V6"),
                unit_sampling_rate=SamplingRateUnit.HZ,
                unit_timestamp=TimestampUnit.SECONDS,
                unit_value=ValueUnit.MILLIVOLT,
            ),
        ),
        annotation_specs=(
            AnnotationSpec(spec_id="rhythm_cls", task=ClassificationTask),
        ),
        view_specs=(
            ViewSpec(name="full", description="Whole recording with all available leads."),
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
                sample_id=rec.recording_id,
                subject_ids=(rec.patient_id,),
                source_ids=(rec.recording_id,),
                signals=(SignalRef(spec_id="ecg_12lead", channels=rec.leads),),
                view="full",
            )
            sample.annotate(
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

**Returns:** A fully-populated `TimeFDataset`. See [TimeFDataset](timef-dataset.md).

**Constraints**

- One source recording = one stored sample. Do not slice; slicing is the data-loader's responsibility.
- `dataset_id` and `version` on the returned `TimeFDataset` must match `metadata().dataset_id` and `metadata().version`.

---

## `DatasetMetadata`

The connector's self-description. Declares identity, classification, and the specs that govern the dataset's contents (signals, annotations, views).

```python
from timenet.domains import Domain
from timenet.licenses import License
from timenet.timef.metadata import (
    AnnotationSpec, SignalSpec, ViewSpec,
)
from timenet.version import Version


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    version: Version
    description: str
    license: License
    signal_specs: tuple[SignalSpec, ...] = ()
    annotation_specs: tuple[AnnotationSpec, ...] = ()
    view_specs: tuple[ViewSpec, ...] = ()
    domains: tuple[Domain, ...] = ()
    source_url: str | None = None
    tags: tuple[str, ...] = ()
```

**Fields**

| Name               | Type                         | Required | Description                                                                                                                   |
| ------------------ | ---------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `dataset_id`       | `str`                        | yes      | Snake-cased unique identifier. Must match the YAML registry key.                                                              |
| `version`          | `Version`                    | yes      | Semantic version (`major.minor.patch`). See [Version](enums-and-spec.md#version).                                             |
| `description`      | `str`                        | yes      | One-sentence human-readable description.                                                                                      |
| `license`          | `License`                    | yes      | Data license.                                                                                                                 |
| `signal_specs`     | `tuple[SignalSpec, ...]`     | no       | Modalities the dataset records (channels, units, sampling rates). See [SignalSpec](enums-and-spec.md#signalspec).                      |
| `annotation_specs` | `tuple[AnnotationSpec, ...]` | no       | Task types the dataset annotates and their label schemas. See [AnnotationSpec](enums-and-spec.md#annotationspec).                      |
| `view_specs`       | `tuple[ViewSpec, ...]`       | no       | Sample views the connector emits. Every `Sample.view` must reference one of these by name. See [ViewSpec](enums-and-spec.md#viewspec). |
| `domains`          | `tuple[Domain, ...]`         | no       | Clinical or application domains (e.g. `Domain.CARDIOLOGY`).                                                                   |
| `source_url`       | `str \| None`                | no       | Canonical URL of the source dataset.                                                                                          |
| `tags`             | `tuple[str, ...]`            | no       | Free-form labels for filtering.                                                                                               |

---

## Testing mode

When `TIMENET_TESTING=1`, `download()` must not hit the network. The standard pattern is to serve fixtures from `cache_dir`:

```python
def download(self, cache_dir: Path) -> list[Recording]:
    if self.testing:
        return [
            Recording("fixture_001", "p_001", cache_dir / "fixture_001.edf", LEADS)
        ]
    # real download ...
```

Fixture files live under `E2E_CACHE_DIR/<dataset_id>/` by convention.

---

## Design rules

- **One connector class, one dataset.**
- **No constructor arguments.** Configuration comes from environment variables with hardcoded defaults.
- **The connector decides what a sample is.** For a given dataset there is one connector and it is authoritative: which recordings become samples, which variables to expose, which views to create.

PD: this design rule will be changed when we support multiple connectors per dataset
