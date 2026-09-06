# Connector anatomy

Deep reference for the `add-dataset-connector` skill. Everything here is verified against the code on
this branch. Anchor files:

- Contract: `packages/timenet/src/timenet/connectors/base.py`
- Bases: `packages/timenet-connectors/src/timenet_connectors/bases/{huggingface,physionet}.py`
- Discovery: `packages/timenet-connectors/src/timenet_connectors/discovery.py`
- Build CLI: `packages/timenet-connectors/src/timenet_connectors/builder/cli.py`
- Worked examples: the `chengsenwang/tsqa`, `physionet/ecg_qa_cot`, `physionet/sleep_edfx`, and
  `timenet/hello_world` connectors

This file is the API surface. Its siblings hold the rules: `fidelity.md` for what a connector may do
to its source, `layout.md` for how its modules divide, `discovery.md` for how to read a release
before designing against it.

## Contents

- [The `BaseConnector` contract](#the-baseconnector-contract)
- [Discovery and the folder layout](#discovery-and-the-folder-layout)
- [The dataset card (`dataset.yaml`)](#the-dataset-card-datasetyaml)
- [Base connectors to reuse](#base-connectors-to-reuse)
- [Building the dataset in `convert`](#building-the-dataset-in-convert)
- [Task types (`timenet.types.tasks`)](#task-types-timenettypestasks)
- [Worked example: `chengsenwang/tsqa` (HuggingFace, QA)](#worked-example-chengsenwangtsqa-huggingface-qa)
- [PhysioNet notes: `physionet/ecg_qa_cot`](#physionet-notes-physionetecgqacot)
- [Fixture-based test pattern](#fixture-based-test-pattern)

## The `BaseConnector` contract

`BaseConnector(ABC, Generic[TRaw])` in `timenet.connectors`. `TRaw` is whatever `download` hands to
`convert` (a dict per Hub row, a dataclass ref per PhysioNet record, etc.).

- `download(self, cache_dir: Path) -> list[TRaw]`: fetch/discover raw source files, return
  lightweight refs. I/O only, no parsing, idempotent for a given `cache_dir`. **Not abstract** — its
  default drives `download_async` to completion, because a connector is called synchronously.
  Override it only for a genuinely synchronous connector.
- `download_async(self, cache_dir: Path) -> list[TRaw]`: the async form of the same step. Implement
  this one when the fetch is I/O-bound and can overlap; `physionet/sleep_edfx` does. Implement one of
  the two, not both.
- `convert(self, raw_refs: list[TRaw]) -> TimeFDataset` (abstract): parse refs into a `TimeFDataset`.
  CPU only, no network.
- `metadata(self) -> DatasetMetadata` (**concrete**, do not override): loads and validates the card via
  `DatasetMetadata.from_yaml`. By convention the card is `dataset.yaml` beside the connector module;
  set the `CARD` class var to point elsewhere. (Some docs call `metadata` abstract; it isn't.)
- There is **no `store` hook**, and nothing else to implement. What runs your connector takes the
  dataset `convert` returns and stores it. How that happens is not a connector's concern.

`list[TRaw]` does not mean one entry per record. A connector that would otherwise build millions of
refs returns a **single handle** that `convert` walks, yielding one record at a time.
`SleepEdfxSource` is one such handle: it carries the study directories and the table paths, and no
row of any table. `discovery.md` says how to choose between the two shapes.

Connectors take **no constructor arguments** (configuration comes from the environment). End with a
module-level `CONNECTOR = <YourClass>`.

### Where the contract ends

You implement `download` and `convert`. Nothing below them is yours to know, with one exception, and
it is the exception three rules elsewhere depend on:

> **`convert` returns a description, not data. The values and the task stream are read afterwards, by
> something you do not call.**

That single fact is the whole of the contract's laziness, and everything else follows from it: a
loader must still work when it is called later, whatever it captured stays alive until then, and a
task stream may be read more than once. You never need to know what does the reading.

## Discovery and the folder layout

A connector is a package folder, not a flat file:

```
packages/timenet-connectors/src/timenet_connectors/datasets/<org>/<name>/
  __init__.py      # re-exports CONNECTOR (and the class) from connector.py
  connector.py     # the BaseConnector subclass; ends with CONNECTOR = <YourClass>
  dataset.yaml     # the dataset card, read by metadata()
  README.md        # the assumptions, the inconsistencies, the warnings
  requirements.txt # the libraries this connector needs, installed into the
                   # environment its build runs in (optional)
  tests/           # one test module per module; no fixture files
```

That is the smallest connector. `heads.py`, one `head()` per raw file type, is a new convention that
no connector ships yet; `discovery.md` says what it is for. A release that ships more than one kind
of file divides further:
`tables.py` and `metadata.py` give meaning, `specs.py` holds the signal map, `keys.py` holds the
annotation keys. The half that **opens** a file is a base, not a connector module — `sleep_edfx`
ships no reader of its own and imports `bases.edf.reader` and `bases.excel`. See `layout.md`.
`physionet/sleep_edfx` is the worked example of the divided shape, `chengsenwang/tsqa` of the
undivided one.

`discovery.resolve(dataset_id)` imports only the one module and reads its `CONNECTOR`.
`discovery._module_name` maps the id to the module path, lowercasing and turning hyphens into
underscores in **both** segments, so
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
  takes an iterable and registers the batch all-or-nothing. When a dataset holds far more tasks than
  records, neither fits: `dataset.set_task_stream(task_types, source)` streams them instead, and does
  not validate them the way `add_task` does. `fidelity.md` says how the count decides which, and the
  four rules a streamed task must obey. Set `scope` and `from_tasks` on the task
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
filling `target`. `ForecastingTask` has a second form: `target_span`, a region inside the record the
task is attached to, exclusive with `target_record_id`. Use it when the future to predict lies in the
same record rather than in another one. Every other task needs exactly one of `target` or `target_annotation_ids` (the latter
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

Subclasses `BasePhysioNetConnector[EcgQaCotSource]`, where `EcgQaCotSource` is a frozen handle over
the release rather than one ref per row. `download` calls `ensure_archive` / `download_files` and
returns that handle; `convert` walks it.

**One record is one recording, not one QA row.** The 12-lead ECG becomes a record, and every
question asked of that recording becomes a task on it. An earlier version made one record per QA row
and cached the shared leads to avoid duplicating them; that was replaced, because a record per row
duplicated the recording in the dataset's own model rather than only in memory.

Read `connector.py` for the current shape. Where this file and the code disagree, the code wins.

## Fixture-based test pattern

Tests live beside their connector, in `<org>/<name>/tests/`, one module per module they cover.

**The repo ships no dataset bytes, and no `fixtures/` directory exists.** A test builds what it
needs, synthetically.

For a single-module connector, mirror the `chengsenwang/tsqa` one at
`packages/timenet-connectors/src/timenet_connectors/datasets/chengsenwang/tsqa/tests/test_connector.py`:
hand-write rows shaped exactly like the source's, in the test module, with a comment saying they are
not derived from the real dataset. Then call `convert()` on them directly and assert on records,
tasks, annotations, and parsed values. No network, no env-var toggles.

For a file-shaped source, mirror `sleep_edfx`, which writes a synthetic release into `tmp_path` from
a fixture factory. That is also what lets it test the cases a real release would not hand you: a
recording with no scoring beside it, one with two, one missing a scored signal.

For a divided connector, mirror `physionet/sleep_edfx`, which has `test_connector.py`,
`test_metadata.py`, `test_tables.py` and `test_tasks.py`. The split pays off here: `test_tables.py`
passes literal tuples and needs no fixture at all, because `tables.py` does no I/O. Decide in the
plan which modules get that treatment.

A synthetic binary fixture must still be a valid file of its format — an EDF the reader accepts is a
well-formed header plus its records, written by the test, not bytes copied from a real recording.
