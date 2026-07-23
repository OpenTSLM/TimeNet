# Connector anatomy

Deep reference for the `add-dataset-connector` skill. Everything here is verified against the code on
this branch. Anchor files:

- Contract: `packages/timenet/src/timenet/connectors/base.py`
- Bases: `packages/timenet-connectors/src/timenet_connectors/bases/{huggingface,physionet}.py`
- Discovery: `packages/timenet-connectors/src/timenet_connectors/discovery.py`
- Curate CLI: `packages/timenet-connectors/src/timenet_connectors/curate/cli.py`
- Worked examples: the `chengsenwang/tsqa`, `physionet/ecg_qa_cot`, and `timenet/hello_world` connectors

## The `BaseConnector` contract

`BaseConnector(ABC, Generic[TRaw])` in `timenet.connectors`. `TRaw` is whatever `download` hands to
`convert` (a dict per Hub row, a dataclass ref per PhysioNet record, etc.).

- `download(self, cache_dir: Path) -> list[TRaw]` (abstract): fetch/discover raw source files, return
  lightweight refs. I/O only, no parsing, idempotent for a given `cache_dir`.
- `convert(self, raw_refs: list[TRaw]) -> TimeFDataset` (abstract): parse refs into a `TimeFDataset`.
  CPU only, no network. Attach values as lazy loaders, never materialized arrays.
- `metadata(self) -> DatasetMetadata` (**concrete**, do not override): loads and validates the card via
  `DatasetMetadata.from_yaml`. By convention the card is `dataset.yaml` beside the connector module;
  set the `CARD` class var to point elsewhere. (Some docs call `metadata` abstract; it isn't.)
- `store(...)` (concrete): derives the schema if missing and streams the dataset through `TimeFWriter`.
  Most connectors never override it.

Connectors take **no constructor arguments** (configuration comes from the environment). End with a
module-level `CONNECTOR = <YourClass>`.

## Discovery and the folder layout

A connector is a package folder, not a flat file:

```
packages/timenet-connectors/src/timenet_connectors/datasets/<org>/<name>/
  __init__.py      # re-exports CONNECTOR (and the class) from connector.py
  connector.py     # the BaseConnector subclass; ends with CONNECTOR = <YourClass>
  dataset.yaml     # the dataset card, read by metadata()
```

`discovery.resolve(dataset_id)` imports only the one module and reads its `CONNECTOR`.
`discovery._module_name` maps the id to the module path: org lowercased, leaf hyphens to underscores, so
`chengsenwang/tsqa -> ...datasets.chengsenwang.tsqa` and `physionet/ecg-qa-cot -> ...datasets.physionet.ecg_qa_cot`.
The org folder needs its own `__init__.py` (a namespace package that exposes no `CONNECTOR`).

## The dataset card (`dataset.yaml`)

```yaml
# yaml-language-server: $schema=https://docs.timenet.ai/schemas/dataset-card-v1.schema.json
yaml_schema_version: 1
dataset_id: chengsenwang/tsqa
dataset_version: 1.0.0
name: TSQA
description: "Time-series question answering: a series plus a question/answer per sample."
license: Apache-2.0
domains:
  - general
tags:
  - qa
  - time-series
  - huggingface
```

`dataset_id` must equal the id the connector is curated under. `license` must be a valid `License`;
`domains` valid `Domain` values. The card is validated on load (needs the `curation` extra, which
`timenet[curation]` pulls in); errors raise `InvalidCardError`.

## Base connectors to reuse

### `BaseHuggingFaceConnector` (`bases/huggingface.py`)
`BaseHuggingFaceConnector(BaseConnector[dict[str, Any]])`. Set `HF_REPO` to the external Hub repo id
and implement `convert`. `download` is inherited: it reads the Hub's auto-converted parquet on
`refs/convert/parquet` and returns one dict per row, so any source format is handled uniformly.
`huggingface_hub` is imported lazily (install the `huggingface` extra); `HF_TOKEN` is read from the
environment, so gated datasets work. Fully private repos have no auto-parquet ref and aren't supported.

### `BasePhysioNetConnector` (`bases/physionet.py`)
`BasePhysioNetConnector(BaseConnector[TRaw])`. Implement `download` and `convert`. Helpers (all lazy
import `wfdb`/`requests` behind the `physionet` extra):
- `_ensure_archive(url, cache_dir, sentinel) -> Path`: download + extract a zip once (idempotent).
- `_stream_download(url, dest)`: chunked download for multi-GB files.
- `_read_header(record_base)`: WFDB header (`fs`, `sig_len`, `sig_name`), no signal decode.
- `_lead_loader(record_base, lead_idx) -> Callable[[], pa.Array]`: lazy float32 loader for one lead's
  physical signal.

## Building the dataset in `convert`

Populate a `TimeFDataset` (`from timenet.dataset import TimeFDataset, TimeSeries`):

- `TimeSeries(spec=..., channel=..., sampling_rate_hz=..., loader=..., time_series_id=..., t_start_s=..., t_end_s=..., source_id=...)`.
  `loader` is a `Callable[[], pa.Array]` returning a **float32** Arrow array. `time_series_id` is the
  dedupe key: reuse the same id (and the same `TimeSeries`) to share one series across samples.
- `spec` is a `TimeSeriesSpec(spec_type=..., name=..., unit_sampling_rate=ureg.hertz, unit_timestamp=ureg.second, unit_value=ureg.<unit>, data_source=...)`.
  Units come from the shared pint registry `ureg` (`from timenet.types import ureg`). Optional
  `data_source=DataSource(data_source_type=..., name=..., provider=...)`.
- `sample = dataset.add_sample(time_series=<tuple of TimeSeries>, view=View.FULL, sample_id=...)`.
  Use `View.WINDOW` for a windowed view.
- `sample.add_annotation(StaticAnnotation(key=..., value=..., id=...))`. Annotation shapes:
  `StaticAnnotation` (whole-sample), `PointAnnotation`, `IntervalAnnotation`.
- `dataset.add_task(sample, <Task>(...))`. Compose derived tasks with `from_tasks=(...)`.

## Task types (`timenet.types.tasks`)

The task **class** is the type tag (used by `search(task=...)`); the instance carries the payload.

| Task | Payload (besides `id`, set automatically) |
| --- | --- |
| `ClassificationTask` | `label`, optional `label_schema` |
| `LabelingTask` | `label`, optional `label_schema`, `time_series_ids`, `windows_s` |
| `CaptioningTask` | `answer` |
| `QATask` | `question`, `answer` |
| `ForecastingTask` | `context_sample_ids`, `target_sample_id` |
| `ReasoningTask` | `question`, `answer`, optional `rationale` (the chain of thought) |

## Worked example: `chengsenwang/tsqa` (HuggingFace, QA)

`connector.py`:

```python
from collections.abc import Callable
import json
from typing import Any

import numpy as np
import pyarrow as pa

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.types import QATask, StaticAnnotation, TimeSeriesSpec, View, ureg
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector

_SPEC = TimeSeriesSpec(
    spec_type="tsqa_series",
    name="TSQA Series",
    unit_sampling_rate=ureg.hertz,
    unit_timestamp=ureg.second,
    unit_value=ureg.dimensionless,
)

def _loader(values: list[float]) -> Callable[[], pa.Array]:
    def load() -> pa.Array:
        return pa.array(np.asarray(values, dtype=np.float32))
    return load

class TSQAConnector(BaseHuggingFaceConnector):
    """Connector for the TSQA time-series QA dataset."""

    HF_REPO = "ChengsenWang/TSQA"  # external Hub repo id, keeps its own casing

    def convert(self, raw_refs: list[dict[str, Any]]) -> TimeFDataset:
        dataset = TimeFDataset(metadata=self.metadata())
        for index, row in enumerate(raw_refs):
            series = json.loads(row["Series"])
            channels = series if series and isinstance(series[0], list) else [series]
            time_series = tuple(
                TimeSeries(
                    spec=_SPEC,
                    channel=f"c{channel}",
                    sampling_rate_hz=1.0,
                    loader=_loader(values),
                    time_series_id=f"row-{index}-c{channel}",
                    t_start_s=0.0,
                    t_end_s=float(len(values)),
                )
                for channel, values in enumerate(channels)
            )
            sample = dataset.add_sample(time_series=time_series, view=View.FULL, sample_id=f"row-{index}")
            sample.add_annotation(StaticAnnotation(key="task", value=row["Task"], id=f"task-{index}"))
            if row.get("Label"):
                sample.add_annotation(StaticAnnotation(key="label", value=row["Label"], id=f"label-{index}"))
            dataset.add_task(sample, QATask(question=row["Question"], answer=row["Answer"], id=f"qa-{index}"))
        return dataset

CONNECTOR = TSQAConnector
```

`__init__.py`:

```python
from timenet_connectors.datasets.chengsenwang.tsqa.connector import (
    CONNECTOR as CONNECTOR,
    TSQAConnector as TSQAConnector,
)
```

## PhysioNet notes: `physionet/ecg_qa_cot`

Subclasses `BasePhysioNetConnector[EcgQaCotRef]` where `EcgQaCotRef` is a frozen dataclass ref.
`download` calls `_ensure_archive` / `_stream_download` and returns refs; `convert` shares the 12-lead
ECG across rows on the same recording (`leads_by_ecg` cache keyed by a stable `time_series_id`), attaches
`StaticAnnotation`s (split, question_type, template_id, clinical_context, answer_options), and adds a
`ReasoningTask(question=..., rationale=<CoT>, answer=<label>)`. `_leads_for` reads the WFDB header for
`fs`/`sig_len`/`sig_name` and builds one lazy `TimeSeries` per lead. See its `connector.py` for the full
pattern, including sharing a series across many samples.

## Fixture-based test pattern

Mirror `packages/timenet-connectors/tests/test_tsqa.py`: check a tiny raw sample into
`tests/fixtures/`, then call `convert()` on it directly and assert on samples, tasks, annotations, and
parsed values. No network, no env-var toggles.
