# mikeam/time-series-reasoning

Synthetic time series with their descriptions, and the multiple-choice questions that Merrill et
al. ask about them in "Language Models Still Struggle to Zero-shot Reason about Time Series"
([arXiv:2404.11757](https://arxiv.org/abs/2404.11757)). The connector reads the Hugging Face
release `mikeam/time-series-reasoning` at commit `d6a1c7d4`, and follows the authors' loading code,
`behavioral-data/TSandLanguage` at commit `d4e748cb`.

## The source data

### What the release is

Every series is synthetic. The paper (Section 3) says the authors "prompt GPT-4 to generate
descriptions of environments that change over time", then "prompt GPT-4 to produce data generating
functions in the form of Python scripts", whose output is the series. They "filter the scenarios by
removing multivariate time series and those with complex, missing, or infinite values", and
"manually reviewed 50 scenarios". No series is a measurement from the field its scenario describes,
so the dataset card declares the `general` domain only.

The 10 scenario categories, such as `Health and Medical Data`, come from GPT-4 too: the authors
"feed 100 captions into GPT-4 and ask the model to categorize these time series into ten domains",
then "automatically apply these categories to the remaining 8.7k scenarios". They split the
description task by them (see [Splits](#splits)).

Each series comes with a written scenario (`description`, and its shorter `description_short` and
`description_tiny`), a list of `characteristics`, free-text `metadata` such as start, end, units and
frequency, and the `generator` code.

### Files

The connector reads six of the release's files. All are JSON Lines, including the
`TS_Dataset_MCQ/*.json` files.

| File | Rows | Size | Holds |
| --- | --- | --- | --- |
| `TS_Dataset.jsonl` | 8,787 | 75 MB | Every series, with its texts and code |
| `TS_Dataset_MCQ/train.json` | 5,779 | 50 MB | Description questions, train |
| `TS_Dataset_MCQ/val.json` | 1,000 | 9 MB | Description questions, validation |
| `TS_Dataset_MCQ/test.json` | 1,036 | 9 MB | Description questions, test |
| `MCQ_1_TS.jsonl` | 97,330 | 940 MB | Questions about one series |
| `MCQ_2_TS.jsonl` | 132,689 | 2,204 MB | Questions comparing a series with a changed version of it |

Every question row repeats its series' texts, metadata and values. A two-series row adds the
`counterfactual`, a "What if ...?" question, and the changed series with its own `new_generator`.

The connector reads neither `TS_Dataset_MCQ/all.json`, which no config of the authors loads and
whose 924 extra rows are in no split, nor the `*_llava.json` files. Those pose the same description
questions, with the same options, to the authors' image model: each names a plot of the series, uses
another prompt (`Please pick the correct description`), and lists the options in one shuffled order.

## The TimeNet dataset

### Records

Each Record holds one series as one Signal on an ordinal axis. The release states start, end,
units and frequency only as free text, so the Signal has no unit and no timestamps; the text is
kept in the `metadata` annotation, as the authors' prompt prints it. A series written in integers
only is stored as `int32`, and any other as `float64`, so no published value changes.

The release builds 45,006 Records:

| Records | Count | Values from |
| --- | --- | --- |
| Series | 8,787 | `TS_Dataset.jsonl`, one per `uuid` |
| Rounded copies | 6,152 | `TS_Dataset_MCQ/{train,val,test}.json` |
| Changed series | 30,067 | `MCQ_2_TS.jsonl` `new_series`, one per `uuid` and `counterfactual` |

Every question row repeats its series' texts, metadata and values, and the connector checks that
each copy equals the series' row in `TS_Dataset.jsonl`. The description files are the exception:
6,160 of their 7,815 rows hold other values than the series' row. 6,149 are the same values
rounded to about 10 digits, and 11 write `0.0` where the series' row writes `-0.0`. The authors'
description task reads these values, so each distinct copy is a Record of its own. The other 1,655
rows use the series' Record.

### Annotations and metadata

| Where | Key | Value |
| --- | --- | --- |
| Series and rounded copy Records | `description`, `description_short`, `description_tiny`, `characteristics`, `generator` | The published texts, edge whitespace included |
| Series and rounded copy Records | `metadata` | `str()` of the published dict, in its key order, as the authors' prompt prints it |
| Changed series Records | `counterfactual`, `generator` | The change, and its `new_generator` code |
| The dataset | `source_repository`, `source_revision` | `mikeam/time-series-reasoning` and `d6a1c7d4…` |

A Record's metadata, `records.TimeSeriesReasoningRecordMetadata`, names the published series it
holds: its `uuid`, and the `file` and `line` its values are read from. A rounded copy and a changed series
keep the uuid of the series they come from. Metadata is provenance and never model input, and no
annotation is a task's input: the prompt holds everything the authors show a model.

### Tasks

The connector builds one `AnswerTask` per row of the authors' three task files, the way their loader
builds its item: `MultimodalMCQDataset.__getitem__` in `src/models/tasks/multimodal.py`, with the
configs in `configs/tasks/llms_and_ts/`.

| Task | Config | File | Tasks | Inputs | Split |
| --- | --- | --- | --- | --- | --- |
| Description | `ts2desc_mcq` | `TS_Dataset_MCQ/{train,val,test}.json` | 7,815 | The series, as the file holds it | The file's |
| Question | `qa_mcq` | `MCQ_1_TS.jsonl` | 97,330 | The series | None |
| Counterfactual | `qa_counterfactual` | `MCQ_2_TS.jsonl` | 132,689 | The series, then its changed version | None |

A task's prompt is the text the authors' loader gives a model, and its target is the letter of the
right option. For a question:

```text
Answer the following question about the time series Freezer Temperature Time Series
{'start': '2022-03-01 00:00:00', 'end': '2022-03-15 00:00:00', 'units': 'Degrees Celsius', 'frequency': 'Every 6 hours'}
What if the power was restored more slowly after the outage?
A) The timeseries would show a shorter period of temperature increase ...
B) ...
C) ...
D) ...
```

The prefix comes from the config, and it meets `description_tiny` without a separator. The texts
keep their published edge whitespace, and the metadata appears as Python prints the dict. A
question's options keep their published order, and its target is the letter at `answer_index`. A
description task's prompt is `What description best relates to this time series?` followed by the
four lettered descriptions.

**The order of a description task's options.** The release lists the answer first. The authors'
loader shuffles the options with `random.shuffle`, unseeded, each time it loads a row, so their
order changes from run to run. A stored prompt needs one order, so the connector draws one per row
with `random.Random`, seeded by the row's file and line, such as `TS_Dataset_MCQ/train.json:1`.
Every build draws the same order. The seed is the connector's choice, not the authors'.

**The series.** `request.authors_series(task)` returns the values the authors' loader passes with
the prompt. For a two-series question, it joins the two input Records into one list, as the loader
does with `series + new_series`.

**Task metadata**, never model input: the task's `file` and `line`; for a description task, its
`scenario_category`; for a question, its `answer_index`, `answer_text` and `source_issues`; for a
two-series question, also its `category`, `difference` or `similarity`.

### Splits

Only the description task has a split in the release. The question tasks have none, and the
connector gives them none. The authors did split the question files, but they released neither
split, and their splits can't be rebuilt from the release. This section records what their code
shows and what remains unknown. The notebooks cited are in `notebooks/mam/ACL_2024/` of
`behavioral-data/TSandLanguage` at `d4e748cb`.

**The authors' rule.** `train_test_split.ipynb` splits by the series' `scenario_category`, which it
reads from `categorized_output.json` (not released):

- **train**: rows in the 7 categories Sales, Wildlife, Environmental, Energy, Technological,
  Recreational and Transport, shuffled with `random_state=42`.
- **val**: a sample of train rows, removed from train, also with `random_state=42`.
- **test**: rows in the 3 categories Health, Agriculture and Education.
- Rows in any other category are dropped.

Appendix B.1 of the paper states the same held-out categories for the models it trains.

**Description task: the authors' split, as released.** `TS_Dataset_MCQ/{train,val,test}.json` is
the output of that rule with a val of 1,000 rows: the notebook prints 5,779, 1,000 and 1,036 rows,
the counts of the three files. The connector keeps this split. Two artifacts of the rule remain:

- Series `f1e0cf26` and `e830d5f7` are in both val and test. Each has two categories, one train
  and one test, and the notebook's join copies such a row into both splits.
- `TS_Dataset_MCQ/all.json` holds 924 rows that are in no split, probably series whose category is
  outside the 10. The category file isn't released, so this can't be checked.

**Question tasks: no split in the release.** The authors' configs read
`data/processed/QA/{train,val,test}.json` (`qa_mcq.yaml`) and
`data/processed/counterfactual_qa_mcq/{train,val,test}.json` (`qa_counterfactual.yaml`), and their
evaluation scripts test fine-tuned models on the `qa_mcq` test split (`slurm-utils/eval/*.sh`). None
of these files is released.

- **Two-series questions.** `train_test_split.ipynb` applies the rule to
  `data/processed/Counterfactual/CF_Feb_1.json` with a val of 100 rows. Its saved output prints
  131,157 train, 100 val and 19,690 test rows: 150,947 in all. `MCQ_2_TS.jsonl` has 132,689 rows,
  so it isn't the file the authors split, and their split can't be rebuilt.
- **Single-series questions.** `synthetic_datasets.ipynb` cells 26 to 29 apply the rule to
  `v2_MCQ.json` with a val of 1,000 rows, after two filters: only the 10 most common categories,
  and only series of 10 to 1,500 values, all finite and at most 1e9. Three things are unknown:
  - `v2_MCQ.json` isn't released, and isn't `MCQ_1_TS.jsonl` as published: the notebooks filter it
    by a `category` field that `MCQ_1_TS.jsonl` doesn't have.
  - The cells have no saved output, so there are no counts to compare with.
  - Cell 5 of the same notebook renames categories, such as `Agricultural and Food Production` to
    `Agriculture and Food`. If it ran before the split, the rule misses 4 of the 10 categories:
    Environmental, Wildlife and Recreational in train, and Agriculture in test. That contradicts
    Appendix B.1, which lists Agriculture as a test category.

**The rule applied to the release.** For reference only, the connector doesn't apply it. Taking
each series' category from the description files, the only source of categories in the release:

| File | Train category | Test category | No category | Two categories |
| --- | --- | --- | --- | --- |
| `MCQ_1_TS.jsonl` | 75,034 | 11,471 | 10,804 | 21 |
| `MCQ_2_TS.jsonl` | 102,550 | 15,581 | 14,558 | 0 |

With cell 5's renaming, the single-series counts would be 47,125 and 7,341, and 32,039 more rows
would be in neither split. The 621 single-series rows that fail the series filter all have no
category in the release, so the description split likely left their series out for the same
reason. The authors' val rows are a random sample of a file that isn't released, so no
validation split follows from the release.

## Comparing with the paper

**Counts.** The release matches the paper's: "8.7k" series (8,787), "100k single time series MCQs"
(97,330) and "over 130k" two-series questions (132,689), together the "230k" questions it reports
(230,019).

**How the authors score.** Their fine-tuned models score a task by the log-probability of each
option's letter after the context and series, and compare the most likely with the label's
position (`src/models/models/bases.py` and `src/models/eval.py`). A task's target letter is that
label. On a description task, each of the authors' runs drew its own option order; the dataset
fixes one draw per row, so a score compares with theirs over the whole task, not row by row.

**The authors' other format.** The authors evaluate API models such as GPT-4 with another script,
`src/models/gpt_all.py`. It writes `Description:`, `MetaData:` and optionally the series as text,
then several questions per request as `Question #1:` and `Option A:` lines, and asks for
`{'question number': 'correct option'}`. The tasks follow the configs' loader, not this script.

## Data limitations

The connector keeps the release as the authors publish and load it. These issues are in the
source data and are documented here, not corrected.

### Blank questions and options

12 question rows have a blank question or option. The authors' loader keeps them, and formats a
blank option as `C) `. `release.PINNED_BLANK_LINES` lists them, so a blank anywhere else fails
the build.

| File | Lines | What is blank |
| --- | --- | --- |
| `MCQ_1_TS.jsonl` | 71926, 71927 | One option, which is the answer in both rows |
| `MCQ_2_TS.jsonl` | 45584 | The question and all four options |
| `MCQ_2_TS.jsonl` | 71367–71372, 118975, 128859, 128967 | One wrong option |

Line 45584 can't be answered: its target, `A`, stands for a blank option, as do the other three.

### Repeated options

243 questions offer the same text twice: 218 single-series and 25 two-series questions. In 102 of
them, 85 and 17, the answer's text is also a wrong option's. For example, `MCQ_2_TS.jsonl` line
11893 offers `0 to 20`, `0 to 20`, `10 to 10` and `0 to 0 no change`, with `answer_index` 0. The
target is the letter `A`, as the authors score it, so a model that picks `B`, the same text, is
marked wrong. A score by `answer_text` would count both.

## Filtering rows with source issues

Each question task lists its row's issues in its metadata, under `source_issues`. The list is
empty when the row has none. The issues are the values of `issues.SourceIssue`:

| Issue | Rows | Meaning |
| --- | --- | --- |
| `blank_question` | 1 | The question is blank |
| `blank_option` | 12 | At least one option is blank |
| `repeated_option` | 243 | Two options have the same text |
| `answer_text_repeated` | 102 | The answer's text is also a wrong option's |

254 question tasks have at least one issue. The connector keeps them all.
`issues.without_source_issues` leaves them out of a score, and the connector doesn't call it:

```python
from timenet_connectors.datasets.mikeam.time_series_reasoning.issues import (
    SourceIssue,
    without_source_issues,
)

# Leave out every task with an issue.
clean = list(without_source_issues(tasks))
# Leave out only the tasks whose answer can't be told apart by its text.
clear = list(
    without_source_issues(tasks, issues=[SourceIssue.ANSWER_TEXT_REPEATED])
)
```

Report which issues were left out, and how many tasks remain, with any score.

## Data flow and validation

A build reads each file once, in stages, one module each:

1. `rows.py` validates every line as a strict row, and keeps each series' length, value type and
   digest, not its values.
2. `index.py` requires every row's copy of its series to equal `TS_Dataset.jsonl`, groups the
   changed series by counterfactual, and checks the pinned row counts and blank lines in
   `release.py`.
3. `records.py` builds the Records. Their values stay in the files: a Signal rereads its line when
   read, and fails if the line no longer holds the series it was built from.
4. `tasks.py` and `request.py` build each task as the authors' loader builds its item. The loader
   streams them, rereading the two question files, so they are never all held in memory.

The tests read a small fixture release, real lines of the pinned release copied byte for byte
(`tests/release_fixtures.py` names each one). `tests/test_tasks.py` copies the authors'
`MultimodalMCQDataset.__getitem__` line for line and requires every task's prompt, letter and
series to equal its item, and `tests/test_connector.py` writes the conversion and reads it back.

On the full release, run on 2026-10-27: all 237,834 tasks equal the item the authors' loader builds
from the same row, in prompt, letter and series, with no mismatch. A full build stores 248 MB and
reads back 45,006 Records and 237,834 tasks, with the split counts above.

## Local build

The build downloads the six files, 3.3 GB, from the Hub at the pinned commit:

```bash
uv run timenet-build build mikeam/time-series-reasoning --out /tmp/timenet
```

Converting the release took 54 s on 2026-10-27, and the built dataset takes 248 MB. To load it:

```python
from timenet.client import TimeNet

dataset = TimeNet("/tmp/timenet").load("mikeam/time-series-reasoning")
```
