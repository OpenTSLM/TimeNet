# Connector anatomy

Deep reference for the `add-dataset-connector` skill. Everything here is verified against the code on
this branch. Anchor files:

- Contract: `packages/timenet/src/timenet/connectors/base.py`
- Bases: `packages/timenet-connectors/src/timenet_connectors/bases/{huggingface,physionet}.py`
- Discovery: `packages/timenet-connectors/src/timenet_connectors/discovery.py`
- Build CLI: `packages/timenet-connectors/src/timenet_connectors/builder/cli.py`
- Worked examples: the `chengsenwang/tsqa`, `physionet/ecg_qa_cot`, and `timenet/hello_world` connectors

## The `BaseConnector` contract

`BaseConnector(ABC, Generic[TRaw])` in `timenet.connectors`. `TRaw` is whatever `download` hands to
`convert` (a dict per Hub row, a dataclass ref per PhysioNet record, etc.).

- `download(self, cache_dir: Path) -> list[TRaw]` (abstract): fetch/discover raw source files, return
  lightweight refs. I/O only, no parsing, idempotent for a given `cache_dir`.
- `convert(self, raw_refs: list[TRaw]) -> TimeFDataset` (abstract): parse refs into a `TimeFDataset`.
  CPU only, no network.
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
  requirements.txt # the libraries this connector needs, installed into the
                   # environment its build runs in (optional)
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
description: "Time-series question answering: a series plus a question/answer per record."
license: Apache-2.0
domains:
  - general
tags:
  - qa
  - time-series
  - huggingface
```

`dataset_id` must equal the id the connector is built under. `license` must be a valid `License`;
`domains` valid `Domain` values. The card is validated on load (needs the `build` extra, which
`timenet[build]` pulls in); errors raise `TimeNetInvalidCardError`.

## Base connectors to reuse

### `BaseHuggingFaceConnector` (`bases/huggingface.py`)
`BaseHuggingFaceConnector(BaseConnector[dict[str, Any]])`. Set `HF_REPO` to the external Hub repo id
and implement `convert`. `download` is inherited: it reads the Hub's auto-converted parquet on
`refs/convert/parquet` and returns one dict per row, so any source format is handled uniformly.
The connector imports `huggingface_hub` lazily and declares it in its `requirements.txt`. The lazy
import keeps `--no-isolation` usable while you write the connector. `HF_TOKEN` is read from the
environment, so gated datasets work. Fully private repos have no auto-parquet ref and aren't supported.

### `BasePhysioNetConnector` (`bases/physionet.py`)
`BasePhysioNetConnector(BaseConnector[TRaw])`. Implement `download` and `convert`. Its two helpers read
records through `wfdb`, which the base imports lazily so that `--no-isolation` stays usable while you
write the connector:
- `_read_header(record_base)`: WFDB header (`fs`, `sig_len`, `sig_name`), no signal decode.
- `_lead_loader(record_base, lead_idx) -> Callable[[], pa.Array]`: lazy float32 loader for one lead's
  physical signal.

The base does not fetch the archive. Call `ensure_archive` / `download_files` from
`timenet_connectors.download`. The connector's `requirements.txt` then declares `wfdb` plus whatever
its download path needs. `physionet/ecg_qa_cot` pulls PTB-XL from `s3://physionet-open/`, which goes
through boto3, so it names `boto3` too.

## Building the dataset in `convert`

Populate a `TimeFDataset` (`from timenet.dataset import TimeFDataset, TimeSeries`):

- `TimeSeries.from_values(values, *, spec, signal, time_axis, source_id=None, time_series_id=None)`
  is the shortcut when you already hold the values in memory: it wraps them in a **float32** loader and
  takes `n_values` from the array's own length. When the source has one arbitrary time offset per point,
  use `TimeSeries.from_irregular(values, *, time_offsets_us, spec, signal, ...)` instead, which derives
  the axis from the stream. Use the raw `TimeSeries(..., loader=<Callable[[], pa.Array]>, ...)`
  constructor only for genuinely lazy sources (files, remote shards). `time_series_id` is the dedupe key:
  reuse the same id (and the same `TimeSeries`) to share one series across records.
- `spec` is a `TimeSeriesSpec(spec_type=..., name=..., unit_value=ureg.<unit>, data_source=...)`.
  Units come from the shared pint registry `ureg` (`from timenet.types import ureg`). Optional
  `data_source=DataSource(data_source_type=..., name=..., provider=...)`.
- `record = dataset.add_record(time_series=<tuple of TimeSeries>, record_id=...)`. A windowed record
  says so through its axis: `RegularAxis.at_index(...)` moves the origin into the recording.
- `record.add_annotation(Annotation(key=..., value=..., id=...))` attaches one and returns it;
  `record.add_annotations([...])` takes an iterable and returns a tuple. One class: its shape comes from
  its `span`. No span means whole-record; `span=TimePoint.seconds(...)` a time offset;
  `span=TimeInterval.seconds(...)` a region.
- `dataset.add_task(record, <Task>(...))` registers one and returns it; `dataset.add_tasks(record, [...])`
  takes an iterable and registers the batch all-or-nothing. Set `scope` and `from_tasks` on the task
  itself, not the call; a batch may derive from its own members in any order.
- Name any annotation or task you reference later and read its `id` off it. Never repeat an id literal in
  `input_annotation_ids`, `target_annotation_ids`, or `from_tasks`.

## Task types (`timenet.types.tasks`)

The task **class** is the type tag (used by `search(task=...)`); the instance carries the payload.

Every task shares one frame on the `Task` base — `record_ids`, `prompt`, `scope` (a `Span` narrowing the
input), `input_annotation_ids`, `target` / `target_annotation_ids`, `rationale`, `from_tasks` — so the
type only says what *kind* of answer it is.

| Task | Answer | Extra payload |
| --- | --- | --- |
| `ClassificationTask` | `target: str` (a label) | optional `target_schema` |
| `AnswerTask` | `target: str` (free text; a caption when there is no `prompt`) | — |
| `ScalarPredictionTask` | `target: float` | optional `unit`, `target_name` |
| `TemporalLocalizationTask` | `target: tuple[Span, ...]` | `mode` (`SPARSE` / `EXHAUSTIVE`) |
| `ForecastingTask` | the produced series | `context_record_ids`, `target_record_id` |
| `TSEditingTask` | the produced series | `source_record_id`, `target_record_id` |
| `TSGenerationTask` | the produced series | `target_record_id` |
| `TSCorrespondenceTask` | `target: tuple[str, ...]` (record ids) | `candidate_record_ids` |

The three series-output tasks set `answer_is_record` and locate their answer by record id instead of
filling `target`. Every other task needs exactly one of `target` or `target_annotation_ids` (the latter
points at stored annotations instead of copying them into the task row); `add_task` enforces that, plus
the bounds of every `Span` the task carries.

## Worked example: `chengsenwang/tsqa` (HuggingFace, QA)

`connector.py`:

```python
import json
from typing import Any

from timenet.dataset import TimeFDataset, TimeSeries
from timenet.dataset.axis import OrdinalAxis
from timenet.types import Annotation, AnswerTask, TimeSeriesSpec, ureg
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector

_SPEC = TimeSeriesSpec(
    spec_type="tsqa_series",
    name="TSQA Series",
    unit_value=ureg.dimensionless,
)

class TSQAConnector(BaseHuggingFaceConnector):
    """Connector for the TSQA time-series QA dataset."""

    HF_REPO = "ChengsenWang/TSQA"  # external Hub repo id, keeps its own casing

    def convert(self, raw_refs: list[dict[str, Any]]) -> TimeFDataset:
        dataset = TimeFDataset(metadata=self.metadata())
        for index, row in enumerate(raw_refs):
            series = json.loads(row["Series"])
            signals = series if series and isinstance(series[0], list) else [series]
            time_series = tuple(
                TimeSeries.from_values(
                    values,
                    spec=_SPEC,
                    signal=f"c{signal}",
                    time_axis=OrdinalAxis(),
                    time_series_id=f"row-{index}-c{signal}",
                )
                for signal, values in enumerate(signals)
            )
            record = dataset.add_record(time_series=time_series, record_id=f"row-{index}")
            record.add_annotation(Annotation(key="task", value=row["Task"], id=f"task-{index}"))
            if row.get("Label"):
                record.add_annotation(Annotation(key="label", value=row["Label"], id=f"label-{index}"))
            dataset.add_task(record, AnswerTask(prompt=row["Question"], target=row["Answer"], id=f"qa-{index}"))
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
`download` calls `ensure_archive` / `download_files` and returns refs; `convert` shares the 12-lead
ECG across rows on the same recording (`leads_by_ecg` cache keyed by a stable `time_series_id`), attaches
whole-record `Annotation`s (split, question_type, template_id, clinical_context, answer_options), and adds a
`AnswerTask(prompt=..., rationale=<CoT>, target=<label>)`. `_leads_for` reads the WFDB header for
`fs`/`sig_len`/`sig_name` and builds one lazy `TimeSeries` per lead. See its `connector.py` for the full
pattern, including sharing a series across many records.

## Fixture-based test pattern

Tests live beside their connector, in `<org>/<name>/tests/`. Mirror the `chengsenwang/tsqa` one at
`packages/timenet-connectors/src/timenet_connectors/datasets/chengsenwang/tsqa/tests/test_connector.py`:
check a tiny raw sample into `tests/fixtures/`, then call `convert()` on it directly and assert on
records, tasks, annotations, and parsed values. No network, no env-var toggles.
