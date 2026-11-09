# Connector anatomy

Deep reference for the `add-dataset-connector` skill. Everything here is verified against the code on
this branch. Anchor files:

- Contract: `packages/timenet/src/timenet/connectors/base.py`
- Bases: `packages/timenet-connectors/src/timenet_connectors/bases/{huggingface,physionet}.py`
- Discovery: `packages/timenet-connectors/src/timenet_connectors/discovery.py`
- Build CLI: `packages/timenet-connectors/src/timenet_connectors/builder/cli.py`
- Implementations: see [Connector examples](#connector-examples) below.

## The `BaseConnector` contract

`BaseConnector(ABC, Generic[TRaw])` in `timenet.connectors`. `TRaw` is whatever `download` hands to
`convert` (a dict per Hub row, a dataclass ref per PhysioNet record, etc.).

- `download(self, cache_dir: Path) -> list[TRaw]`, or `async download_async(self, cache_dir)`:
  implement one of them. Fetch or find the raw source files and return lightweight refs. I/O only,
  no parsing, idempotent for a given `cache_dir`. The default `download` runs `download_async`.
- `convert(self, raw_refs: list[TRaw]) -> TimeFDataset`: parse refs into a `TimeFDataset`. Read
  local files and build the dataset without network access. A child dataset implements `compose(self, raw_refs, context)` instead, and reads its
  parents from `context.parents`. A connector class that has neither raises `TypeError` when you
  create an instance.
- `metadata(self) -> DatasetMetadata` (**concrete**, do not override): loads and validates the card via
  `DatasetMetadata.from_yaml`. By convention the card is `dataset.yaml` beside the connector module;
  set the `CARD` class var to point elsewhere. (Some docs call `metadata` abstract; it isn't.)
- The engine, not the connector, derives the schema and writes the release. There is no `store`
  method to call or override.

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

See the "Connector layout" section of `docs/connectors.md` for organizing code within this package.

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
- `_lead_loader(record_base, header, lead_idx) -> Callable[[], pa.Array]`: lazy float32 loader for
  one lead's physical signal.

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
  when the values stay in a file that a reader can open for one signal (WFDB, EDF, a large array),
  so that `convert` does not keep every signal in memory.
- `spec` is a `TimeSeriesSpec(spec_type=..., name=..., unit_value=ureg.<unit>, data_source=...)`.
  Units come from the shared pint registry `ureg` (`from timenet.types import ureg`). Optional
  `data_source=DataSource(data_source_type=..., name=..., provider=...)`.
- A spec allows no missing timestep by default, so a `None` value fails the build. If the source
  has gaps, say why on the spec: `missing=MissingValues(reason=MissingReason.SENSOR_GAP)`. The
  writer records how many timesteps each signal misses, as `Signal.null_count`. Keep a null a
  null: never fill a gap.
- Group signals under a `Source(name=..., signals=...)`. Then
  `dataset.add_record(record=Record(sources=(source,)))` registers the record and returns it. A
  windowed record says so through its axis: `RegularAxis.at_index(...)` moves the origin into the
  recording.
- `record.annotate(Annotation(key=..., value=...))` attaches one annotation and returns it.
  `record.add_annotations([...])` takes an iterable and returns a tuple. There is one annotation
  class, and its `span` gives its shape. No span means the whole record.
  `span=TimePoint.seconds(...)` means a time offset. `span=TimeInterval.seconds(...)` means a region.
- For text that the connector reads from a source file, use `LazyText(function)` as the
  annotation value. Import it from `timenet.types`. The function returns the string. If you know
  its number of characters, pass it as `length=n` and every read checks the result against it.
  The writer stores this text in a separate payload file so readers can load it on request. For a
  list of strings, use `LazyTextList`, with the number of strings as `length` if you know it.
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
| `ClassificationTask` | `target: str` (a label) | Required `label_set`: a `LabelSet` that defines the allowed labels, their descriptions, and how many labels a target can contain. |
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

## Connector examples

The "Example connectors" section of `docs/connectors.md` lists the connectors to read. Each one is
in `packages/timenet-connectors/src/timenet_connectors/datasets/<org>/<name>/connector.py`.

## PhysioNet notes: `physionet/ptb_xl` and `physionet/ecg_qa_cot`

`physionet/ptb_xl` subclasses `BasePhysioNetConnector`. Its `download_async` fetches the files, and
`convert` builds one record per row of `ptbxl_database.csv`. For each record, `convert` reads the
WFDB header with `_read_header` and builds one lazy `Signal` per lead with
`Signal.from_loader(..., loader=self._lead_loader(base, header, position))`. The record keeps the
upstream `ecg_id` in `Record.metadata` and the patient in a `subject_ids` annotation.

`physionet/ecg_qa_cot` declares PTB-XL as a parent in its card and implements `compose` instead of
`convert`. It imports the PTB-XL records and adds the questions as tasks. `docs/connectors.md`
describes this in "Compose a child dataset".

## Fixture-based test pattern

Tests live beside their connector, in `<org>/<name>/tests/`. Mirror the `chengsenwang/tsqa` one at
`packages/timenet-connectors/src/timenet_connectors/datasets/chengsenwang/tsqa/tests/test_connector.py`:
check a tiny raw sample into `tests/fixtures/`, then call `convert()` on it directly and assert on
records, tasks, annotations, and parsed values. Also write the release and read it back, as the
"Connector layout" section of `docs/connectors.md` asks. No network, no env-var toggles.
