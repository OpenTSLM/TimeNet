# mikeam/time-series-reasoning

Synthetic time series with their descriptions, and the multiple-choice questions that Merrill et
al. ask about them in "Language Models Still Struggle to Zero-shot Reason about Time Series"
([arXiv:2404.11757](https://arxiv.org/abs/2404.11757)). The connector reads the Hugging Face
release `mikeam/time-series-reasoning` at commit `d6a1c7d4`. It follows the authors' loading code,
`behavioral-data/TSandLanguage` at commit `d4e748cb`.

All of the code is in `connector.py`.

## The source data

Every series is synthetic. GPT-4 wrote each scenario, and a Python script that GPT-4 wrote
generated the series. No series is a measurement from the field its scenario names, so the dataset
card declares the `general` domain only.

Each series comes with a written scenario (`description`, and the shorter `description_short` and
`description_tiny`), a list of `characteristics`, free-text `metadata` (such as start, end, units
and frequency), and the `generator` code.

The connector reads six files. All are JSON Lines, including the `TS_Dataset_MCQ/*.json` files. The
row counts below are the dataset card's. This change does not check them on the real files.

| File | Rows | Holds |
| --- | --- | --- |
| `TS_Dataset.jsonl` | 8,787 | Every series, with its texts and code |
| `TS_Dataset_MCQ/train.json` | 5,779 | Description questions, train |
| `TS_Dataset_MCQ/val.json` | 1,000 | Description questions, validation |
| `TS_Dataset_MCQ/test.json` | 1,036 | Description questions, test |
| `MCQ_1_TS.jsonl` | 97,330 | Questions about one series |
| `MCQ_2_TS.jsonl` | 132,689 | Questions that compare a series with a changed version of it |

Every question row repeats the texts, metadata and values of its series. A two-series row adds a
`counterfactual` (a "What if ...?" question) and the changed series with its own `new_generator`.

The connector does not read `TS_Dataset_MCQ/all.json`, which no config of the authors loads, or the
`*_llava.json` files, which pose the description questions to the authors' image model.

## The TimeNet dataset

### Records

Each Record holds one series as one Signal on an ordinal axis. The release gives start, end, units
and frequency only as free text, so the Signal has no unit and no timestamps. The text stays in the
`metadata` annotation. A series written in integers only is stored as `int32`. Any other series is
stored as `float64`. No published value changes.

The connector builds three kinds of Record, in this order:

1. One Record for each row of `TS_Dataset.jsonl`.
2. One Record for each distinct rounded copy of a series in the description files. The description
   files hold some series with other values than the series' own row (for example rounded to fewer
   digits). The authors' description task reads those values, so each distinct copy has its own
   Record. A description row with the same values as its series uses the series' Record.
3. One Record for each changed series, by `uuid` and `counterfactual`.

A Record's metadata holds the `uuid` of the series it comes from, and the `file` and `line` that its
values are read from. A rounded copy and a changed series keep the `uuid` of their series.

| Record | Annotations |
| --- | --- |
| Series and rounded copy | `description`, `description_short`, `description_tiny`, `characteristics`, `generator` as published, edge whitespace included, and `metadata` as `str()` of the published dict in its key order, the way the authors' prompt prints it |
| Changed series | `counterfactual`, and `generator` (the published `new_generator`) |

The dataset has the annotations `source_repository` and `source_revision`.

### Tasks

The connector builds one `AnswerTask` for each row of the three kinds of task file, the way the authors'
loader builds its item (`MultimodalMCQDataset.__getitem__`, with the configs in
`configs/tasks/llms_and_ts/`).

| Task | Config | File | Input Records | Split |
| --- | --- | --- | --- | --- |
| Description | `ts2desc_mcq` | `TS_Dataset_MCQ/{train,val,test}.json` | The series, as the file holds it | The file's |
| Question | `qa_mcq` | `MCQ_1_TS.jsonl` | The series | None |
| Counterfactual | `qa_counterfactual` | `MCQ_2_TS.jsonl` | The series, then its changed version | None |

The prompt is the text the authors' loader gives a model. The target is the letter of the right
option. A question prompt looks like this:

```text
Answer the following question about the time series <description_tiny>
<metadata as printed by Python>
<question>
A) <option 1>
B) <option 2>
C) <option 3>
D) <option 4>
```

The prefix comes from the config, and it meets `description_tiny` without a separator. A question
keeps its options in the published order. Its target is the letter at `answer_index`. A description
prompt is `What description best relates to this time series?` followed by the four lettered
descriptions.

The order of the description options: the release lists the right description first. The authors'
loader shuffles the options with `random.shuffle` and no seed, so the order changes on each load. A
stored prompt needs one order. The connector shuffles the options of each row with `random.Random`,
seeded by the row's file and line (for example `TS_Dataset_MCQ/train.json:1`). Every build gives a
row the same order. The seed is the connector's choice, not the authors'. A score on a description
task compares with the authors' over the whole task, not row by row.

No annotation is a task's input, because the prompt holds everything the authors show a model.

Task metadata is provenance and never model input:

- Every task has `file` and `line`.
- A description task has `scenario_category`.
- A question task has `answer_index`, `answer_text` (the right option) and `source_issues`.
- A counterfactual task also has `category` (`difference` or `similarity`).

### Splits

Only the description task has a split, the one in the file names. The test file holds three
held-out domains (Health and Medical Data, Agricultural and Food Production, and Educational and
Public Services). The question files have no split in the release, so the question tasks have
none.

## Source issues

The connector keeps every row as published. A question task lists the problems of its row in
`source_issues`, or an empty list:

| Issue | Meaning |
| --- | --- |
| `blank_question` | The question is blank |
| `blank_option` | At least one option is blank |
| `repeated_option` | Two options have the same text |
| `answer_text_repeated` | The answer's text is also a wrong option's, so only `answer_index` tells which option the authors count as right |

To leave out the rows with an issue, filter on the task metadata:

```python
clean = [task for task in tasks if not task.metadata["source_issues"]]
```

A description task has no `source_issues` key. Report which issues you left out, and how many tasks
remain, with any score.

## How a build works

Pass one reads each file once. It checks every row, and it keeps only small results: the series' texts,
a digest of each series, and a digest of each file. The Records read their values from the files
when needed. A Record fails with `TimeFFormatError` when its line no longer holds the series that the
build read.

Pass two rereads the five task files to stream the tasks, so the tasks are never all in memory. It
fails with `TimeFFormatError` when a file differs from pass one. The files must stay in place until
writing ends.

The build raises `TimeFValidationError`, with the file and line, for these problems:

- A row is not one JSON object, or it repeats a field.
- A field is missing or has the wrong type.
- A series is empty, holds a value that is not a finite number, or holds an integer that its type
  cannot hold exactly.
- An option list does not have four texts, an `answer_index` is not 0 to 3, or a category is not one
  of the release's.
- A description row's first option is not its description.
- A rounded copy in a description file has another length than its series.
- A row's copy of its series differs from `TS_Dataset.jsonl` in its texts, its metadata (including
  key order) or, for the question files, its values.
- A `uuid` repeats in `TS_Dataset.jsonl` or is not in it.
- Two rows of one counterfactual hold different changed series.

## Tests

The tests read a small fixture release: real lines of the pinned release, copied byte for byte.
`tests/test_connector.py` names each one. They check every Record's values and texts, and they
compare every task with an item built by a line-for-line copy of the authors' loader. They also check
that each failure above raises, and that a changed file fails on reread.

## What this change does not verify

This change does not download or build the real release. The row counts, the number of Records and
tasks, and the number of rows with each source issue come from the dataset card and are not checked
here. A build of the full release (the files are about 3.3 GB) must confirm them.

## Local build

```bash
uv run timenet-build build mikeam/time-series-reasoning --out /tmp/timenet
```
