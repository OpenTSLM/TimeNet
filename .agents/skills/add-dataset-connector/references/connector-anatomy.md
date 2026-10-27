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
`BaseHuggingFaceConnector(BaseConnector[HubFileRef])` downloads files from a Hub dataset at a pinned
commit. Set:

- `HF_REPO`: the Hub repo id.
- `SOURCE_REVISION`: a 40-character commit, not a branch, so every rebuild reads the same files.
- `SOURCE_FILES`: part name -> file path in the repo.

`download` is inherited and returns one `HubFileRef(path, part)` per file. Declare
`huggingface_hub>=1.1` in the connector's `requirements.txt`, and set `HF_TOKEN` for gated or
private datasets. The base checks the refs before your `convert` runs: they must be nonempty and
name each declared part at most once. Do not call `check_refs` yourself.

To read the Hub's parquet conversion instead of the original files, pin the commit that
`refs/convert/parquet` points to and read the files with `iter_parquet_rows`
(`timenet_connectors.sources.parquet`).

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

Populate a `TimeFDataset` (`from timenet.dataset import Record, Signal, Source, TimeFDataset`):

- `Signal.from_values(values, *, spec, name, time_axis)`
  is the shortcut when you already hold the values in memory. It checks them against the spec,
  keeps them in memory, and takes `n_values` from the array length. If the source has one
  arbitrary time offset per point, use
  `Signal.from_irregular(values, *, time_offsets_us, spec, name)` instead. It derives the axis
  from the stream. Use `Signal.from_loader(..., loader=<Callable[[], pa.Array]>, n_values=...)`
  only for lazy sources (files, remote shards).
- `spec` is a `TimeSeriesSpec(spec_type=..., name=..., unit_value=ureg.<unit>, data_source=...)`.
  Units come from the shared pint registry `ureg` (`from timenet.types import ureg`). Optional
  `data_source=DataSource(data_source_type=..., name=..., provider=...)`.
- Group signals under a `Source(name=..., signals=...)`. Then
  `dataset.add_record(record=Record(sources=(source,)))` registers the record and returns it. A
  windowed record says so through its axis: `RegularAxis.at_index(...)` moves the origin into the
  recording.
- `record.annotate(Annotation(key=..., value=...))` attaches one annotation and returns it.
  `record.add_annotations([...])` takes an iterable and returns a tuple. There is one annotation
  class, and its `span` gives its shape. No span means the whole record.
  `span=TimePoint.seconds(...)` means a time offset. `span=TimeInterval.seconds(...)` means a region.
- `dataset.add_task(task=<Task>(inputs=(record,), ...))` registers one task and returns it.
  `dataset.add_tasks(tasks=[...])` takes an iterable and registers the batch all-or-nothing. The task
  names its input records itself. Set `scope` and `from_tasks` on the task too. A batch can derive
  from its own members in any order.
- You cannot pass ids. The dataset gives every record, signal, and task a read-only int `id` when
  you add it. Sources, axes, and annotations have no id. Put upstream identifiers (a recording
  name, an `ecg_id`) in `Record.metadata` or in an annotation, for example
  `metadata={"ecg_id": 17}`. A span takes Signal objects in `time_series`, not ids.
- Keep a Python reference to any annotation or task you reference later and pass that object. When
  you look objects up by an upstream key, keep a local dict keyed by that key, not by a derived id
  string.

## Task types (`timenet.types.tasks`)

The task **class** is the type tag (used by `search(task=...)`); the instance carries the payload.

Every task shares one set of fields on the `Task` base: `inputs`, `prompt`, `scope` (a `Span` that
narrows the input), `input_annotations`, `targets` / `target_annotations`, `rationale`, `from_tasks`.
So the type only says what *kind* of answer it is.

| Task | Answer | Extra payload |
| --- | --- | --- |
| `ClassificationTask` | `target: str` (a label) | optional `target_schema` |
| `AnswerTask` | `targets: tuple[str, ...]` (free text, which is a caption when there is no `prompt`) | — |
| `ScalarPredictionTask` | `target: float` | optional `unit`, `target_name` |
| `TemporalLocalizationTask` | `target: tuple[Span, ...]` | `mode` (`SPARSE` / `EXHAUSTIVE`) |
| `ForecastingTask` | the produced series | `context_record_ids`, `target_record_id` |
| `TSEditingTask` | the produced series | `source_record_id`, `target_record_id` |
| `TSGenerationTask` | the produced series | `target_record_id` |
| `TSCorrespondenceTask` | `targets: tuple[Record, ...]` | `candidate_records` |

The three series-output tasks set `answer_is_record` and locate their answer by record id instead of
filling `target`. Every other task needs exactly one of `target` or `target_annotation_ids` (the latter
points at stored annotations instead of copying them into the task row); `add_task` enforces that, plus
the bounds of every `Span` the task carries.

## Worked example: `chengsenwang/tsqa` (HuggingFace, QA)

`connector.py`:

```python
import json
from collections.abc import Mapping
from typing import ClassVar

from timenet.dataset import Record, Signal, Source, TimeFDataset
from timenet.dataset.axis import OrdinalAxis
from timenet.types import Annotation, AnswerTask, TimeSeriesSpec, ureg
from timenet_connectors.bases.huggingface import BaseHuggingFaceConnector, HubFileRef
from timenet_connectors.sources.parquet import iter_parquet_rows


_SPEC = TimeSeriesSpec(
    spec_type="tsqa_series",
    name="TSQA Series",
    unit_value=ureg.dimensionless,
)


class TSQAConnector(BaseHuggingFaceConnector):
    """Connector for the TSQA time-series QA dataset (Hub repo ``ChengsenWang/TSQA``)."""

    HF_REPO = "ChengsenWang/TSQA"  # the external Hub repo id (keeps its own casing)
    # The commit refs/convert/parquet pointed to: the Hub's parquet conversion of the CSV.
    SOURCE_REVISION = "22518209e3ee6302be4b4957f2692b3e8de849c3"
    SOURCE_FILES: ClassVar[Mapping[str, str]] = {"train": "default/train/0000.parquet"}

    def convert(self, raw_refs: list[HubFileRef]) -> TimeFDataset:
        """Build one record per row: the parsed series plus its question-and-answer task.

        Args:
            raw_refs: The parquet files from :meth:`download`.

        Returns:
            The populated :class:`~timenet.dataset.TimeFDataset`.
        """
        dataset = TimeFDataset(metadata=self.metadata())
        for ref in raw_refs:
            for row in iter_parquet_rows(ref.path):
                series = json.loads(row["Series"])
                signals = series if series and isinstance(series[0], list) else [series]
                time_series = tuple(
                    Signal.from_values(
                        values,
                        spec=_SPEC,
                        name=f"c{signal}",
                        time_axis=OrdinalAxis(),
                    )
                    for signal, values in enumerate(signals)
                )
                record = Record(sources=(Source(name="TSQA series", signals=time_series),))
                dataset.add_record(record=record)
                record.annotate(Annotation(key="task", value=row["Task"]))
                if row.get("Label"):
                    record.annotate(Annotation(key="label", value=row["Label"]))
                dataset.add_task(
                    task=AnswerTask(
                        inputs=(record,),
                        prompt=row["Question"],
                        targets=(row["Answer"],),
                    )
                )
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
`download` calls `ensure_archive` / `download_files` and returns refs. `convert` builds one record per
recording and keeps the records in a local dict keyed by the PTB-XL `ecg_id`. It also stores the
`ecg_id` in `Record.metadata`. It attaches whole-record `Annotation`s (split, question_type,
template_id, clinical_context, answer_options). It adds an
`AnswerTask(inputs=(record,), prompt=..., rationale=<CoT>, targets=(<label>,))`. `_leads_for` reads
the WFDB header for `fs`/`sig_len`/`sig_name` and builds one lazy `Signal` per lead. All tasks for a
recording point at its one record. Records do not share a series. See its `connector.py` for the
full pattern.

## Fixture-based test pattern

Tests live beside their connector, in `<org>/<name>/tests/`. Mirror the `chengsenwang/tsqa` one at
`packages/timenet-connectors/src/timenet_connectors/datasets/chengsenwang/tsqa/tests/test_connector.py`:
check a tiny raw sample into `tests/fixtures/`, then call `convert()` on it directly and assert on
records, tasks, annotations, and parsed values. No network, no env-var toggles.
