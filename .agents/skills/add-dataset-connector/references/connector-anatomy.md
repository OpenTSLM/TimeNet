# Connector anatomy

Deep reference for the `add-dataset-connector` skill. Everything here is verified against the code on
this branch. Anchor files:

- Contract: `packages/timenet/src/timenet/connectors/base.py`
- Bases: `packages/timenet-connectors/src/timenet_connectors/bases/{huggingface,physionet}.py`
- Discovery: `packages/timenet-connectors/src/timenet_connectors/discovery.py`
- Build CLI: `packages/timenet-connectors/src/timenet_connectors/builder/cli.py`
- Worked examples: the `chengsenwang/tsqa`, `physionet/ecg_qa_cot`, `physionet/sleep_edfx`, and
  `timenet/hello_world` connectors

This file is the API surface. It states what a connector is made of. It does not state how to build
one: the rules for that are the skill's own, and they live in its other references.

## Contents

- [The `BaseConnector` contract](#the-baseconnector-contract)
- [Discovery and the folder layout](#discovery-and-the-folder-layout)
- [The dataset card (`dataset.yaml`)](#the-dataset-card-datasetyaml)
- [Base connectors to reuse](#base-connectors-to-reuse)
- [Building the dataset in `convert`](#building-the-dataset-in-convert)
- [Where an answer, an annotation and a task can live](#where-an-answer-an-annotation-and-a-task-can-live)
- [Task types (`timenet.types.tasks`)](#task-types-timenettypestasks)
- [Worked example: `chengsenwang/tsqa` (HuggingFace, QA)](#worked-example-chengsenwangtsqa-huggingface-qa)
- [PhysioNet notes: `physionet/ecg_qa_cot`](#physionet-notes-physionetecgqacot)
- [Where the tests live](#where-the-tests-live)

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
- Those four are the whole contract. What runs a connector takes the dataset `convert` returns and
  stores it, and that is not a connector's concern.

`list[TRaw]` does not mean one entry per sample. A connector that would otherwise build millions of
refs returns a **single handle** that `convert` walks, yielding one sample at a time.
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

That is the smallest connector. **One rule decides whether a file belongs in that folder: a file
that `download` or `convert` imports and calls is part of the connector and is committed with it.
Every other file used to build the connector stays out.** A `head()` is the common case of the
second half, and the census script is another.

A connector that reads more than one kind of file divides further. `tables.py` and `metadata.py`
give meaning, `specs.py` holds the channel map, and `keys.py` holds the annotation keys.
`physionet/sleep_edfx` is shaped that way; `chengsenwang/tsqa` is one `connector.py`.

The half that **opens** a file is a base, not a connector module. `sleep_edfx` ships no reader of
its own and imports `bases.edf.reader` and `bases.excel`.

`discovery.resolve(dataset_id)` imports only the one module and reads its `CONNECTOR`. The org
folder needs its own `__init__.py`, a namespace package that exposes no `CONNECTOR`.

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

- `TimeSeries.from_values(values, *, spec, channel, time_axis, source_id=None, time_series_id=None)`
  is the shortcut when you already hold the values in memory: it wraps them in a **float32** loader and
  takes `n_values` from the array's own length. When the source has one arbitrary time offset per point,
  use `TimeSeries.from_irregular(values, *, time_offsets_us, spec, channel, ...)` instead, which derives
  the axis from the stream. Use the raw `TimeSeries(..., loader=<Callable[[], pa.Array]>, ...)`
  constructor only for genuinely lazy sources (files, remote shards). `time_series_id` is the dedupe key:
  reuse the same id (and the same `TimeSeries`) to share one series across samples.
- `spec` is a `TimeSeriesSpec(spec_type=..., name=..., unit_value=ureg.<unit>, data_source=...)`.
  Units come from the shared pint registry `ureg` (`from timenet.types import ureg`). Optional
  `data_source=DataSource(data_source_type=..., name=..., provider=...)`.
- `sample = dataset.add_sample(time_series=<tuple of TimeSeries>, sample_id=...)`. A windowed sample
  says so through its axis: `RegularAxis.at_index(...)` moves the origin into the recording.
- `sample.add_annotation(Annotation(key=..., value=..., id=...))` attaches one and returns it;
  `sample.add_annotations([...])` takes an iterable and returns a tuple. One class: its shape comes from
  its `span`. No span means whole-sample; `span=TimePoint.seconds(...)` a time offset;
  `span=TimeInterval.seconds(...)` a region.
- `dataset.add_task(sample, <Task>(...))` registers one and returns it; `dataset.add_tasks(sample, [...])`
  takes an iterable and registers the batch all-or-nothing. When a dataset holds far more tasks than
  samples, neither fits: `dataset.set_task_stream(task_types, source)` streams them instead, and does
  not validate them the way `add_task` does. Set `scope` and `from_tasks` on the task itself, not
  the call; a batch may derive from its own members in any order.
- Name any annotation or task you reference later and read its `id` off it. Never repeat an id literal in
  `input_annotation_ids`, `target_annotation_ids`, or `from_tasks`.

## Where an answer, an annotation and a task can live

Each row below is a choice, not a rule, and each one changes what the built dataset costs and what a
reader can do with it. Pick one per connector and **say in the plan which you picked and why**. A
facility you did not know about is not a choice you made.

| the choice | reach for it when | already done in |
| --- | --- | --- |
| `dataset.add_task(sample, task)` | the tasks are few and you want the cross-task validation | `chengsenwang/tsqa/connector.py:123` |
| `dataset.add_tasks(sample, tasks)` | one batch belongs to one sample, all of it or none of it | `sleep_edfx/tasks.py` |
| `dataset.set_task_stream(task_types, source)` | there are far more tasks than samples, or more than fit in memory | `physionet/ecg_qa_cot/connector.py:297` |
| `sample.add_annotation` / `add_annotations` | the annotation belongs to one sample and is read back through it | `sleep_edfx/connector.py` |
| `dataset.register_annotations` (`dataset/dataset.py:186`) | a task references an annotation that no sample carries, and many tasks reference the same one. It dedupes by id | `physionet/ecg_qa_cot/connector.py:227` |
| `Task.target` | the answer is a short value that belongs to this one task | `chengsenwang/tsqa/connector.py:123` |
| `Task.target_annotation_ids` (`types/tasks.py:118`) | the answer **is** stored annotations: store the text one time and point many tasks at it, instead of copying it into every task row | nothing yet — the path is written and tested only for validation |
| `Task.input_annotation_ids` | the task is asked *about* stored annotations rather than answered by them | `physionet/ecg_qa_cot/connector.py:306` |
| `Task.from_tasks` | this task is derived from other tasks, and a reader has to be able to follow it back | — |

**A task sets `target` or `target_annotation_ids`, never both and never neither.** `add_task`
raises `TimeFValidationError` when a task sets both, and when it sets neither
(`dataset/dataset.py:525-533`). Set `scope` and `from_tasks` on the task itself, not on the call.

**Copying an answer into every task is the mistake this table exists to prevent.** A release with
four captions per sample and 600 000 samples writes 2.4 million copies of text it could have stored
one time. `register_annotations` plus `target_annotation_ids` is the pair that stores it once.

## Task types (`timenet.types.tasks`)

The task **class** is the type tag (used by `search(task=...)`); the instance carries the payload.

Every task shares one frame on the `Task` base — `sample_ids`, `prompt`, `scope` (a `Span` narrowing the
input), `input_annotation_ids`, `target` / `target_annotation_ids`, `rationale`, `from_tasks` — so the
type only says what *kind* of answer it is.

| Task | Answer | Extra payload |
| --- | --- | --- |
| `ClassificationTask` | `target: str` (a label) | optional `target_schema` |
| `AnswerTask` | `target: str` (free text; a caption when there is no `prompt`) | — |
| `ScalarPredictionTask` | `target: float` | optional `unit`, `target_name` |
| `TemporalLocalizationTask` | `target: tuple[Span, ...]` | `mode` (`SPARSE` / `EXHAUSTIVE`) |
| `ForecastingTask` | the produced series | `context_sample_ids`, `target_sample_id` |
| `TSEditingTask` | the produced series | `source_sample_id`, `target_sample_id` |
| `TSGenerationTask` | the produced series | `target_sample_id` |
| `TSCorrespondenceTask` | `target: tuple[str, ...]` (sample ids) | `candidate_sample_ids` |

The three series-output tasks set `answer_is_sample` and locate their answer by sample id instead of
filling `target`. `ForecastingTask` has a second form: `target_span`, a region inside the sample the
task is attached to, exclusive with `target_sample_id`. Use it when the future to predict lies in the
same sample rather than in another one. Every other task needs exactly one of `target` or `target_annotation_ids` (the latter
points at stored annotations instead of copying them into the task row); `add_task` enforces that, plus
the bounds of every `Span` the task carries.

## Worked example: `chengsenwang/tsqa` (HuggingFace, QA)

This is the smallest connector in the tree, and it answers the two questions a row-shaped release
raises first. Its corpus states no sample id and no sampling rate. So it builds the id from the
row's position — `sample_id=f"row-{index}"` at line 53, `time_series_id=f"row-{index}-c{channel}"` at
line 49 — and gives every series an `OrdinalAxis()` at line 48 rather than inventing a rate.

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
            channels = series if series and isinstance(series[0], list) else [series]
            time_series = tuple(
                TimeSeries.from_values(
                    values,
                    spec=_SPEC,
                    channel=f"c{channel}",
                    time_axis=OrdinalAxis(),
                    time_series_id=f"row-{index}-c{channel}",
                )
                for channel, values in enumerate(channels)
            )
            sample = dataset.add_sample(time_series=time_series, sample_id=f"row-{index}")
            sample.add_annotation(Annotation(key="task", value=row["Task"], id=f"task-{index}"))
            if row.get("Label"):
                sample.add_annotation(Annotation(key="label", value=row["Label"], id=f"label-{index}"))
            dataset.add_task(sample, AnswerTask(prompt=row["Question"], target=row["Answer"], id=f"qa-{index}"))
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

**One sample is one recording, not one QA row.** The 12-lead ECG becomes a sample, and every
question asked of that recording becomes a task on it. An earlier version made one sample per QA row
and cached the shared leads to avoid duplicating them; that was replaced, because a sample per row
duplicated the recording in the dataset's own model rather than only in memory.

Read `connector.py` for the current shape. Where this file and the code disagree, the code wins.

## Where the tests live

A connector's tests sit in `<org>/<name>/tests/`, one module per module they cover. A connector
divided across several modules has several test modules: `sleep_edfx` has `test_connector.py`,
`test_metadata.py`, `test_tables.py` and `test_tasks.py`.

The repo ships no dataset bytes, and no `fixtures/` directory exists anywhere in it. Every test
builds what it needs at run time. `layout.md § Tests` holds the rules for writing one.
