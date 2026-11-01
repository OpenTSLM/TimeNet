# RATS40K

`time-ra/rats40k` holds RATS40K, the benchmark of TIME-RA ([arXiv:2507.15066][paper]): univariate
and multivariate series from ten real-world domains, each with an anomaly category, a binary
label, and a diagnostic explanation that language models generated and GPT-4 ranked.

The connector reads the two JSON files of the [Hugging Face release][card] at commit
`dc968d6a88f396215be50023c93c3928e06096df`. The source license is Apache-2.0. Access requires
accepting the Hugging Face gate; use an existing login or an externally supplied `HF_TOKEN`. The
authors' code is [yyysjz1997/Time-RA at `200b244`][code].

[paper]: https://arxiv.org/abs/2507.15066
[card]: https://huggingface.co/datasets/Time-RA/RATs40K
[code]: https://github.com/yyysjz1997/Time-RA/tree/200b244f75b134348fad39c9ba140d9b2f1e1279

## The source data

Each file maps a split, `TSAD_train` or `TSAD_test`, to rows keyed by a decimal string:

| Original file | Train | Test | Total | Null entries excluded |
| --- | ---: | ---: | ---: | ---: |
| `RATs-Uni-TSImage_Reason.json` | 30,266 | 6,034 | 36,300 | 11 test entries |
| `RATs-Multi-TSImage_Reason.json` | 2,728 | 546 | 3,274 | 0 |
| Total | 32,994 | 6,580 | 39,574 | 11 |

The eleven null entries are test keys that the authors' `labeling/rank_models.py` starts as
`None` and never fills. The paper counts them: its 36,311 univariate instructions are the 36,300
rows here plus the eleven. Downloads must match the pinned split counts and null keys in
`release.PINNED_INVENTORY`.

A row holds these fields:

| Field | Content |
| --- | --- |
| `Observation` | Univariate: a list of numbers, 32, 64, or 128 long. Multivariate: a string of `"name": [values]` pairs separated by `;`, with 3 or 9 channels, 16, 32, or 64 long |
| `Source` | The domain and data source, such as `Healthcare-ECG` |
| `FigurePath` | The rendered plot of the series in `figures.zip` or `figures_multi.zip` |
| `Action` | The anomaly category name |
| `ActionID` | The category's code |
| `Label` | `Normal` or `Anomaly` |
| `Thought` | The generated explanation |
| `Rank` | Test rows only: the GPT-4 judge's ordering of the four labelling models |

The release gives no units, sample rates, or timestamps, and no row names the upstream series or
file it was cut from. The card names UCR, NASA, Yahoo, MITDB, and AIOps challenges as sources, but
no mapping from rows to them is published.

### Generated labels

Four models (GPT-4o, Gemini, DeepSeek-R1, and Llama-3.3-70B) each labelled the rows, and GPT-4
ranked their outputs. The published `Thought` and `Action` come from the first-ranked model. The
labelling script passes the source benchmark's `Label` to the model and parses `Thought` and
`Action` from the same response, so `Thought` argues for `Action`, and `Label` is the source
benchmark's label, not the generated one.

## The TimeNet dataset

### Records

Each usable entry becomes one Record with one Source: a single signal, or the named multivariate
channels, whose original order is the Source's `channel_order` metadata, since TimeF does not keep
signal order on read. Signals are lazy float64 with unknown units and ordinal axes, since the
release gives neither units nor sampling cadence. TimeNet numbers Records and tasks itself, so the
source row is in `Record.metadata` as `variant`, `split`, and `key`, the fields of
`records.RATS40KRecordMetadata`. Keys repeat across splits, so all three name the row.
`records.record_metadata(record)` reads them back as a typed object. The release names no upstream
series or subject, so the shared `subject_id` is unset. Nothing else is in record metadata, so
nothing derived from a row's fields travels with the input. The domain (`source_domain`) is a
Record annotation, since the task uses it as an input annotation.

### Tasks

Each record has one `ClassificationTask`:

| Field | Value |
| --- | --- |
| `prompt` | The variant's evaluation prompt (see [Prompt](#prompt)) |
| `input_annotations` | `source_domain` |
| `targets` | The published category name, `Action` |
| `target_schema` | `uni_category_vocabulary` or `multi_category_vocabulary`, the key of the vocabulary annotation |
| `rationale` | The published explanation, `Thought` |
| `annotations` | `label`, the published `Label`, and `action_id`, the published `ActionID` |
| `metadata` | `target_in_vocabulary`, `figure_path`, and `Rank` on test rows |
| `split` | The source split, `train` or `test`, with no validation split |

This mirrors the authors, who ask one question per row and read the category and the explanation
from the same answer. The target is the category *name* because that is what the prompt asks the
model to answer, so an answer is compared with the target directly, with no code mapping.

`label` and `action_id` are task annotations so the authors' detection and code-based metrics
remain reproducible. They are in `task.annotations`, never in `input_annotations`, so they are
never model input. `figure_path` is task metadata, not record metadata, because in univariate test
its extension gives away which model wrote the row: the 1,552 paths that end in `.pdf` are exactly
the rows `gpt-4o` ranked first.

### Dataset annotations

| Annotation | Value |
| --- | --- |
| `source_repository` | `Time-RA/RATs40K` |
| `source_revision` | The pinned commit |
| `uni_category_vocabulary` | The 15 univariate category names, in the authors' code order (0–14) |
| `multi_category_vocabulary` | The 8 multivariate category names, in the authors' code order (0–7) |
| `uni_request_template` | The authors' univariate evaluation prompt, as their `str.format` template |
| `multi_request_template` | The authors' multivariate evaluation prompt, as their `str.format` template |

### Prompt

`prompts/uni.txt` and `prompts/multi.txt` hold the authors' [evaluation prompts][prompts]:
`USER_DETECTION_PROMPT` from `prompt_llama_anoclf_reason.py` (univariate) and
`prompt_llama_multi_ano.py` (multivariate). Every evaluation script uses them, for zero-shot and
fine-tuned models alike. Each states the role, the category definitions, worked examples, and the
`\boxed1{}`/`\boxed2{}` answer format, and asks for a `Thought` and a category name. The files keep
the authors' `str.format` escapes (`{{}}`) and their two placeholders, `{our_source}` and
`{our_observation}`. Wording and typos are unchanged. The authors' text ends 15 univariate and 2
multivariate lines with trailing spaces, which carry no meaning; the repository's
`trailing-whitespace` hook strips them from the stored files.

Each task's `task.prompt` is its variant's prompt without the domain clause ("The oberservation is
from [the] domain of {our_source}.") and the `Observation:` line, since the task's
`source_domain` input annotation and its Record supply them. TimeF stores the prompt once per
task, and DuckDB does not compress strings this long. The two distinct 6 KB prompts take 249 MB of
the 270 MB full build; that cost is accepted.

**Rebuilding the authors' request.** The dataset annotations `uni_request_template` and
`multi_request_template` hold the full prompts. Fill a template as the authors' scripts do:
`our_source` is the task's `source_domain`, and `our_observation` is the series as the authors
print it. A univariate series is Python's `str()` of its float list. Multivariate channels follow
the Source's `channel_order`, not the signal order, each as `"name": [values]`, joined by `"; "`.
`request.authors_request` does this:

```python
from timenet.client import TimeNet
from timenet_connectors.datasets.time_ra.rats40k.request import (
    SYSTEM_PROMPT,
    authors_request,
)

dataset = TimeNet(registry="/tmp/rats40k-registry").load(
    "time-ra/rats40k@1.0.0"
)
task = next(dataset.iter_tasks())
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": authors_request(task, dataset.annotations)},
]
```

On the full build, `authors_request` equals the authors' `USER_DETECTION_PROMPT.format(...)` of
the raw row for all 39,574 tasks, apart from those trailing spaces. The system message (`You are a
helpful assistant.`) is not stored in the dataset. The authors' API clients send it, and their
local Hugging Face models drop it.

### Answer format

The prompt asks for two boxed fields, and each task supplies one of them:

```text
Thought: \boxed1{<the task's rationale>}
Action: \boxed2{<the task's target, a category name>}
```

`request.authors_answer(task)` formats a task this way. It is the target the authors fine-tune on
([`train/train_sft_parser.py`][sft]). TimeNet ships no parser or metric, because evaluation is out
of its scope, but a score that wants to match the authors has to parse as they do. Their
evaluation scripts (`eval_anomaly_multiclf_reason*.py`) do the following:

- They read the text after `Action:` to the end of that line, then strip a leading `\boxedN{`, a
  leading code such as `12.`, and a trailing `}`.
- They look the name up in the variant's `ANOMALY_DICT`, whose order is the vocabulary's: a name's
  code is its index in `<variant>_category_vocabulary`.
- They skip a name that is not in the dictionary. That row gets no prediction, and `eval2file.py`
  scores only rows with a prediction, so it is left out rather than counted wrong. A model that
  answers the 3,790 out-of-vocabulary targets exactly is never scored on them.
- They keep only the first line after `Thought:` for the text metrics.

A predicted code is the predicted name's index, which is not always the target's `action_id`
(see [Comparing with the paper](#comparing-with-the-paper)). On the full build, a predictor that
answers every target exactly parses back to its target in all 35,784 in-vocabulary tasks and is
skipped on the 3,790 others. Scored with the authors' metrics, it gets:

| Variant and split | Scored | `action_id` match | Macro F1 | Binary F1 |
| --- | ---: | ---: | ---: | ---: |
| Univariate train | 30,266 | 1.000 | 1.000 | 0.978 |
| Univariate test | 2,244 | 1.000 | 1.000 | 0.978 |
| Multivariate train | 2,728 | 0.412 | 0.500 | 1.000 |
| Multivariate test | 546 | 0.418 | 0.500 | 1.000 |

Binary F1 is below 1 where `Label` disagrees with `Action`, and multivariate macro F1 is 0.5
because three categories are stored one code below the prompt's number.

### How the connector follows the authors

The [dataset card][card] frames RATS40K as a multi-objective diagnosis: anomaly detection,
fine-grained classification, and diagnostic reasoning. The authors' [evaluation][eval] asks one
question per row and derives all three scores from the answer: binary F1 against `Label`, macro F1
against `ActionID`, and text similarity against `Thought`. The connector mirrors that setup:

| Authors' setup | Connector choice |
| --- | --- |
| One question per row, three scores from the answer | One task per row; task annotations keep the scoring fields |
| Evaluation prompt per variant | That prompt as `task.prompt` |
| Domain and series filled into the prompt | `source_domain` input annotation and the Record; `authors_request` rebuilds the request |
| Model answers a category name and a `Thought` | Target `Action`, rationale `Thought` |
| `Label` and `ActionID` scored from the answer | Both kept as task annotations, `label` and `action_id` |
| `TSAD_train` and `TSAD_test` | The same two splits, with no validation split |
| Series rendered as figures for multimodal models | Numeric series only; `FigurePath` as task metadata |
| Gated release | `credentialed` access, pinned commit and inventory |

Within TimeNet, each source row is one Record carrying its series once. The category vocabularies
are dataset annotations that the task names through `target_schema`, which is how TimeNet attaches
class lists to classification targets. Source quirks are kept in the data as published and
documented under [Data limitations](#data-limitations) instead of being silently fixed.

## Comparing with the paper

The connector keeps the release exactly as the authors evaluate it: every test row, with
`ActionID` and `Label` as published, in the task annotations `action_id` and `label`. Their
[evaluation][eval] scores all keys shared by the prediction and ground-truth files, with no
filtering or deduplication. It computes macro F1 of the predicted code against `action_id`, and
binary F1 of `code != 0` against `label`. To reproduce it, parse each answer as described in
[Answer format](#answer-format), map the predicted name to its vocabulary index, and score every
test task with a prediction.

On univariate test, macro F1 on `action_id` merges categories. The 3,790 rows labelled with
multivariate names carry multivariate codes, which are also univariate codes: 1,336 `Joint Space
Anomaly` rows have `action_id` 4, the univariate code of `Change Point Anomaly`, and so on for five
more pairs. A univariate answer on such a row is scored against a code its prompt gives another
meaning. The filters below remove these rows.

Two choices in that setup penalise answers that follow the prompt, so a separate filtered score
is worth reporting alongside it. Neither is applied by the connector:

- **Drop out-of-vocabulary targets.** Keep test tasks whose `target_in_vocabulary` is `true`.
  This removes the 3,790 univariate rows labelled with multivariate names.
- **Drop identical inputs with conflicting targets.** Group rows by prompt variant, domain, and
  named float64 series, across both splits, ignoring keys, figure paths, and channel order. Drop
  every member of a group with more than one category target; nothing tells which is right. For
  example, univariate train key 545 and test key 5747 have identical
  inputs but targets `Nonlinear Pattern Anomaly` and `Sudden Spike Anomaly`. A model sees the same
  input in both cases, so it can match at most one of them.
- **Score binary detection from the category.** Compare against `Normal` for `Normal Sequence`
  and `Anomaly` otherwise, instead of `Label`, which disagrees with `Action` in 66 test rows.

On the full build, the two filters keep 2,779 of 6,580 test tasks: they exclude 3,790
out-of-vocabulary targets and 11 more conflicting-input tasks. Filtering does not certify the
remaining generated labels or remove train/test overlap with matching targets, and only exact
duplicates are caught; see [Data limitations](#data-limitations) for the overlapping windows it
leaves. Report which filters were applied and the retained count with any score.

**Category vocabularies.** Each vocabulary lists the category names its prompt defines, in the
authors' code order: 15 for univariate (0–14) and 8 for multivariate (0–7). The data does not
always follow the prompt:

- 3,790 of the 6,034 univariate test rows (63%) carry a *multivariate* category name, so their
  target is outside the univariate vocabulary and a model following the prompt cannot match it.
  The authors' univariate labelling script imports the multivariate prompt module, the likely
  cause; the `Thought` of these rows reasons with the multivariate categories. The rows are kept
  as published, and their task metadata sets `target_in_vocabulary` to `false`, so a score can
  exclude them with one filter.
- Multivariate `ActionID` uses 0–6, matching the six anomaly types on the dataset card, while the
  prompt numbers its categories 0–7. Its class 4 (`Local Structural Jump Anomaly`) never occurs,
  and `Joint Space`, `Principal Component Space`, and `Collinearity Shift Anomaly` are stored as
  4, 5, and 6 where the prompt numbers them 5, 6, and 7 (1,923 of the 3,274 rows). The name target
  avoids this; it affects only comparisons by `ActionID`.

**Label/Action disagreements.** 1,184 rows have a `Label` that disagrees with whether `Action` is
`Normal Sequence`: 1,117 univariate train, 66 univariate test, and one multivariate train. Both are
kept as published (see [Generated labels](#generated-labels)).

## Data limitations

The connector keeps the release as the authors publish and evaluate it. These issues are in the
source data and are documented here, not corrected:

- **Overlapping windows across splits.** Many rows are windows cut from the same upstream series
  at different offsets, and the authors' splits don't keep them apart. 1,935 of the 6,034
  univariate test rows and 124 of the 546 multivariate test rows share a run of 32 consecutive
  values, with at least 8 distinct values, with some train row. Most are `Healthcare-ECG` (887),
  `Environment-Sensor` (674), `Machine-System` (343), and `Medical-ECG` (116). Counting runs of 16
  values, it is 3,636 and 288. For example, univariate test row 63 and train row 27 are both
  128-value `Healthcare-ECG` series, and test values 12–127 equal train values 0–115. The test row
  is labelled `Normal Sequence` and the train row `Sudden Spike Anomaly`. A model can score well
  on such rows by memorizing its training data, and the filters in
  [Comparing with the paper](#comparing-with-the-paper) remove only exact repeats. The dataset
  keeps the authors' splits, so that results compare with the paper.
- **Repeated series.** 43 univariate and 10 multivariate test series also appear, value for
  value, in train, and 182 univariate and 18 multivariate train series appear more than once.
- **Identical inputs with conflicting targets.** An audit of the full build found 66 groups
  covering 216 tasks with identical series, channel names, domain, and prompt variant but
  different category targets. This includes 25 groups with conflicts within train and eight
  within test. See [Comparing with the paper](#comparing-with-the-paper) for how to exclude them.
- **Univariate test rows with multivariate names.** 3,790 univariate test rows (63%, nearly all
  `Anomaly`) carry a multivariate category name, and their `Thought` often reasons about
  variables or channels. The first model in `Rank` predicts this: no row where `gpt-4o` ranks
  first is affected, against 81–89% where `gemini`, `llama-70b`, or `deepseek-r1` ranks first.
  Their targets are outside the univariate vocabulary, and `target_in_vocabulary` is `false` on
  their tasks. Any test score should state whether these rows are kept.
- **Multivariate class codes.** Three multivariate categories have an `ActionID` one below the
  prompt's numbering, which the authors' evaluation uses (see
  [Comparing with the paper](#comparing-with-the-paper)).
- **Label/Action disagreements.** 1,184 rows. In train, 758 of the 1,117 are `Label` Normal with
  an anomalous `Action`; in univariate test, 45 of the 66 are the reverse.
- **Rationale names the category.** `Thought` names its `Action` in 41% of univariate and 28% of
  multivariate rows. It is the rationale, never a model input.
- **`Rank` is parsed model text.** Multivariate entries always list the four generators.
  Univariate entries list one to four, with occasional malformed names such as `llaama-70b`.
- **No comparison with the upstream series.** No row names the upstream file, series, or offset
  it was cut from, so the release can't be checked against its originals. Whether the authors
  scaled, normalized, or reordered values or channels, and what units they are in, is unknown.
  The connector copies every value exactly as published, and a full read-back of the build
  matches the release value for value, but that shows only that TimeNet holds the release, not
  that the release holds its sources faithfully.

## Data flow and validation

The release has no `refs/convert/parquet` revision, so the connector subclasses
`BaseHuggingFaceConnector`, which downloads the two pinned JSON files. The code follows the stages
of a conversion, one module each:

- `release.py` holds the pin, the two variants and their files, the splits, and each file's pinned
  inventory: rows per split and the null entry keys.
- `rows.py` reads the published JSON. It walks the nested split/key objects one entry at a
  time and validates each row against a strict model of its fields, `Row`, which reduces the
  observation to its channels' names and lengths. Each `Row` carries its `RowRef`: the file,
  variant, split, key, and byte range it was read from, never taken from the JSON. The values stay
  in the file. `reread_channels` rereads an entry and parses only its observation, and
  `check_channels` raises `TimeFFormatError` if its channels' names or lengths differ from the
  converted row, as when the file changed.
- `records.py` builds one Record per row, whose signals reread only their own entry, so the raw
  files must stay available until writing ends. A row's signals share one reread: each load's
  `ChannelLoader` caches the last row read (`lru_cache(maxsize=1)`, as TRACE's `WindowLoader`).
- `tasks.py` builds one task per row, `RATS40KCategoryTask`, and the `ClassificationTask` it
  becomes.
- `request.py` holds the authors' prompts: the templates, the task prompts derived from them, the
  category names, `authors_request`, and `authors_answer`.
- `keys.py` names every annotation and metadata key, so writing and reading share one spelling.
- `loader.py` reads both files once and checks the inventory. `RATS40KDatasetLoader` is a
  `DatasetLoader`: it gives the vocabulary and template annotations, the Records, and the tasks,
  which `tasks_of(RATS40KCategoryTask, split=...)` selects by split. `connector.py` adds the
  provenance annotations, and the loader's `populate` adds the rest and streams the tasks.

Validation rejects malformed JSON, duplicate splits or keys, unknown, missing, or invalid row
fields, empty, nonnumeric, or nonfinite observations, and duplicate or unequal channels, with file,
split, and key in the error. Only explicit null entries are skipped. Tests cover the download pin
and inventory, the stored prompts and the authors' category order, published names as targets,
the rebuilt request, the answer format against the authors' parser, malformed inputs, a write and
read-back of the whole conversion, and a conversion whose lazy reads follow UTF-8 and CRLF offsets
across small read buffers.

## Open questions

1. **Domain tags.** `dataset.yaml` has no tag for the synthetic `Artificial` series or the
   `Twitter-Ticker_Symbol` counts.

## Local build

After granting access and configuring Hugging Face authentication outside the repository:

```bash
uv run timenet-build build time-ra/rats40k --out /tmp/rats40k-registry
```

```python
from timenet.client import TimeNet

dataset = TimeNet(registry="/tmp/rats40k-registry").load(
    "time-ra/rats40k@1.0.0"
)
dataset.describe()
```

[eval]: https://github.com/yyysjz1997/Time-RA/blob/200b244f75b134348fad39c9ba140d9b2f1e1279/eval2file.py
[sft]: https://github.com/yyysjz1997/Time-RA/blob/200b244f75b134348fad39c9ba140d9b2f1e1279/train/train_sft_parser.py
[prompts]: https://github.com/yyysjz1997/Time-RA/tree/200b244f75b134348fad39c9ba140d9b2f1e1279/prompts
