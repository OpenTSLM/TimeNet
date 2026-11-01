# RATS40K

`time-ra/rats40k` holds RATS40K, the benchmark of TIME-RA ([arXiv:2507.15066][paper]): univariate
and multivariate series from ten real-world domains, each with an anomaly category, a binary
label, and a diagnostic explanation that language models generated and GPT-4 ranked.

The connector reads the two JSON files of the [Hugging Face release][card] at commit
`dc968d6a88f396215be50023c93c3928e06096df`. The source license is Apache-2.0. Access requires
accepting the Hugging Face gate; use an existing login or an externally supplied `HF_TOKEN`. The
authors' code is [yyysjz1997/Time-RA at `200b244`][code].

**Terms.** The Apache-2.0 license covers the Hugging Face release. It does not replace the terms
of the release's own sources. The series come from datasets such as UCR, NASA, Yahoo S5, MITDB
and AIOps, and Yahoo S5 is under the research-use terms of Yahoo Webscope. GPT-4o, Gemini,
DeepSeek-R1 and Llama 3.3 generated the labels and explanations, and the terms of these models can
limit how you use their outputs, for example to train other models. Read these terms before you
use the data.

[paper]: https://arxiv.org/abs/2507.15066
[card]: https://huggingface.co/datasets/Time-RA/RATs40K
[code]: https://github.com/yyysjz1997/Time-RA/tree/200b244f75b134348fad39c9ba140d9b2f1e1279

## The source data

Each file maps a split, `TSAD_train` or `TSAD_test`, to rows keyed by a decimal string. The
connector checks these counts and the null keys against `release.PINNED_INVENTORY`:

| Original file | Train | Test | Total | Null entries excluded |
| --- | ---: | ---: | ---: | ---: |
| `RATs-Uni-TSImage_Reason.json` | 30,266 | 6,034 | 36,300 | 11 test entries |
| `RATs-Multi-TSImage_Reason.json` | 2,728 | 546 | 3,274 | 0 |
| Total | 32,994 | 6,580 | 39,574 | 11 |

The null entries are test keys that the authors' `labeling/rank_models.py` starts as `None` and
never fills. The paper counts them in its 36,311 univariate instructions.

A row holds these fields:

| Field | Content |
| --- | --- |
| `Observation` | Univariate: a list of numbers. Multivariate: a string of `"name": [values]` pairs separated by `;` |
| `Source` | The domain and data source, such as `Healthcare-ECG` |
| `FigurePath` | The rendered plot of the series in `figures.zip` or `figures_multi.zip` |
| `Action` | The anomaly category name |
| `ActionID` | The category's code |
| `Label` | `Normal` or `Anomaly` |
| `Thought` | The generated explanation |
| `Rank` | Test rows only: the GPT-4 judge's ordering of the labelling models |

The release gives no units, sample rates, or timestamps, and no row names the upstream series or
file it was cut from.

Four models labelled the rows, and GPT-4 ranked their outputs. The published `Thought` and
`Action` come from the first-ranked model. The labelling script gives the model the source
benchmark's `Label` and reads `Thought` and `Action` from one response. So `Thought` argues for
`Action`, and `Label` is the source benchmark's label, not a generated one.

## The TimeNet dataset

### Records

Each usable row becomes one Record with one Source: a single signal, or the named multivariate
channels. The Source's `channel_order` metadata keeps the original channel order, since TimeF does
not keep signal order on read. Signals are lazy float64 on ordinal axes, with unknown units.

`Record.metadata` names the source row with `variant`, `split`, and `key`, the fields of
`records.RATS40KRecordMetadata`. Keys repeat across splits, so all three name the row.
`records.record_metadata(record)` reads them back. The release names no subject, so the Records
have no `subject_ids` annotation. The domain, `source_domain`, is a Record annotation, because the
task uses it as an input annotation.

### Tasks

Each Record has one `ClassificationTask`, as the authors ask one question per row:

| Field | Value |
| --- | --- |
| `prompt` | The variant's evaluation prompt (see [Prompt](#prompt)) |
| `input_annotations` | `source_domain` |
| `targets` | The published category name, `Action` |
| `target_schema` | `uni_category_vocabulary` or `multi_category_vocabulary`, the key of the vocabulary annotation |
| `rationale` | The published explanation, `Thought` |
| `annotations` | `label`, the published `Label`, and `action_id`, the published `ActionID` |
| `metadata` | `target_in_vocabulary`, `figure_path`, and `rank` on test rows: the published `Rank` |
| `split` | The source split, `train` or `test`, with no validation split |

The target is the category name, because the prompt asks the model for the name. `label` and
`action_id` are task annotations, so the authors' metrics stay reproducible. They are never in
`input_annotations`, so they are never model input. `figure_path` is task metadata, not Record
metadata, because in univariate test its `.pdf` extension marks the rows that `gpt-4o` ranked
first.

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

`prompts/uni.txt` and `prompts/multi.txt` hold the authors' [evaluation prompts][prompts], the
`USER_DETECTION_PROMPT` of `prompt_llama_anoclf_reason.py` and `prompt_llama_multi_ano.py`. Every
evaluation script uses them. They keep the authors' wording, their `str.format` escapes, and their
two placeholders, `{our_source}` and `{our_observation}`. The repository's `trailing-whitespace`
hook removes the trailing spaces of some lines, which carry no meaning.

Each `task.prompt` is its variant's prompt without the domain clause and the `Observation:` line,
because the task's `source_domain` input annotation and its Record supply them. TimeF stores the
prompt once per task, so the two prompts take most of the build's size.

To rebuild the authors' full request, fill the `<variant>_request_template` annotation as their
scripts do. `request.authors_request` does this:

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

The system message is not stored in the dataset. The authors' API clients send it, and their local
Hugging Face models drop it.

### Answer format

The prompt asks for two boxed fields, and each task supplies one of them:

```text
Thought: \boxed1{<the task's rationale>}
Action: \boxed2{<the task's target, a category name>}
```

`request.authors_answer(task)` formats a task this way. It is the target that the authors
fine-tune on ([`train/train_sft_parser.py`][sft]). They pair it with a shorter prompt of their
own there, not with `USER_DETECTION_PROMPT`, so `task.prompt` with `authors_answer` does not
rebuild their training pairs.

To score as the authors do, parse an answer as their evaluation scripts
(`eval_anomaly_multiclf_reason*.py`) do:

- Read the text after `Action:` to the end of that line. Remove a leading `\boxedN{`, a leading
  code such as `12.`, and a trailing `}`.
- Look the name up in the variant's vocabulary annotation: a name's code is its index there.
- Skip a name that is not in the vocabulary. The authors' [`eval2file.py`][eval] scores only rows
  with a prediction, so a skipped row is left out, not counted wrong.
- Keep only the first line after `Thought:` for the text metrics.

Their evaluation scores macro F1 of the predicted code against `action_id`, and binary F1 of
`code != 0` against `label`, on every test task with a prediction. Two filters give a fairer
score, and the connector applies neither:

- Keep test tasks whose `target_in_vocabulary` is `true`.
- Drop groups of tasks with identical inputs (prompt variant, domain, and series) but different
  targets, because a model can match only one of them.

State which filters you applied, and how many tasks you kept, with any score.

## Data limitations

The connector keeps the release as the authors publish and evaluate it. These issues are in the
source data, and the connector does not correct them:

- **Univariate test rows with multivariate names.** Most univariate test rows carry a
  multivariate category name, outside the univariate vocabulary, so a model that follows the
  prompt cannot match them. Their `target_in_vocabulary` is `false`.
- **Multivariate class codes.** Three multivariate categories have an `ActionID` one below the
  prompt's numbering. This affects only comparisons by `ActionID`, not the name targets.
- **Label/Action disagreements.** Some rows have a `Label` that disagrees with whether `Action` is
  `Normal Sequence`. Both are kept as published.
- **Overlapping windows across splits.** Many rows are windows of the same upstream series at
  different offsets, and the authors' splits do not keep them apart. Some test series also appear
  in train, value for value. A model can score well on such rows by memorizing its training data.
- **Identical inputs with conflicting targets.** Some groups of rows have the same series, domain
  and prompt variant, but different category targets.
- **Rationale names the category.** `Thought` often names its `Action`. It is the rationale,
  never a model input.
- **`Rank` is parsed model text.** Univariate entries list one to four models, with some
  misspelled names.
- **No comparison with the upstream series.** No row names the upstream file, series or offset,
  so the release cannot be checked against its originals. The connector copies every value
  exactly as published.

## Validation

The connector downloads the two pinned JSON files through `BaseHuggingFaceConnector`. It
validates each row against a strict model and rejects malformed JSON, duplicate splits or keys,
unknown, missing or invalid fields, empty, nonnumeric or nonfinite observations, and duplicate or
unequal channels. The error names the file, split and key. Only explicit null entries are skipped.
Signals reread their own entry lazily, so the raw files must stay available until writing ends.
If a reread entry no longer matches the converted row, reading raises `TimeFFormatError`.

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
