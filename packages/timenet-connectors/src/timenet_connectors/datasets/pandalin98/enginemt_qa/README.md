# EngineMT-QA

EngineMT-QA asks questions about simulated aero-engine flights from NASA's N-CMAPSS simulator. Each
question is about one 600-step row of 33 signals, or about a group of 10 rows. The authors define four
tasks, one per stage: Understanding, Perception, Reasoning and Decision-Making. The dataset comes
with the ITFormer paper (Wang et al., ICML 2025).
The paper is [arXiv 2506.20093](https://arxiv.org/abs/2506.20093).
The authors' code is [ITFormer-ICML25](https://github.com/Pandalin98/ITFormer-ICML25).

The connector builds one task per labelled question-answer pair. A single-row question reads the
Record of its row. A multi-row question reads one joined Record: the first 60 steps of each of its
10 rows. The authors' code gives its model the same input. The connector lives in one file,
`connector.py`.

## The source data

The release is the Hub dataset
[pandalin98/EngineMT-QA](https://huggingface.co/datasets/pandalin98/EngineMT-QA), pinned at
`ea2051a15512212976c22c1a075ec549c7d48eef`. The Hub metadata declares the MIT license. It holds three
files:

| File | Holds |
| --- | --- |
| `time_series_data.h5` | `seq_data`, float64 shaped (118,921, 600, 33), and `data_ID`, the ids 1 to 118,921 |
| `train_qa.jsonl` | 13,291 lines, 101,565 QA pairs |
| `test_qa.jsonl` | 5,539 lines, 42,477 QA pairs |

### Signals

All signal values are in the one HDF5 file. A QA line names its rows by id, stored as a string such
as `"633"`. Row `i` of `seq_data` has id `i + 1`. So a QA's id `n` is row `n - 1`, which is how the
authors' `TsQaDataset` reads it. The HDF5 file has no attributes and names no channel. The authors'
channel audit
([`docs/ISSUE_23_CHANNEL_AUDIT.md`](https://github.com/Pandalin98/ITFormer-ICML25/blob/b80142f/docs/ISSUE_23_CHANNEL_AUDIT.md))
says not to name the 33 channels without a mapping, so the connector does not.

The release does not say what a row is. We read about 400 rows over HTTP and found:

- A row is a window of one flight. Consecutive ids are consecutive windows: the last step of a row
  joins the first step of the next one. So a row of 600 steps at 1 Hz is the paper's "10-minute (600
  points) time-series segment", and N-CMAPSS records its flights at 1 Hz.
- A flight is 10 to 21 consecutive rows. When a flight ends inside a row, the release fills the rest
  of that row with zeros in all 33 channels. In 171 consecutive rows, 11 rows ended this way, one per
  flight.
- The release does not mark where a flight starts. The zeros at the end of a flight are the only
  sign of a flight boundary.
- The values look standardized, with statistics that the release does not give. Channel 32 is
  constant within each row, and its values do not follow the health answers.

The QA files name only 7,755 of the 118,921 rows. The single-row lines name 5,945 rows. The 8,845
distinct lists of the multi-row lines use 1,900 rows, and 90 rows occur in both. No question names
the other rows.

### QA

Each line has an `id`, a `name` and a `conversations` list of `human` questions. A `gpt` answer
follows each question. A pair is one question and its answer. Each question has a `stage`, 1 to 4,
and an `attribute`, `open` or `close`. A line has one of two shapes:

| Line | `id` | Pairs |
| --- | --- | --- |
| Single-row | one row id | 8 Understanding, then 7 Perception |
| Multi-row | a list of 10 row ids | 2 Reasoning, then 1 Decision-Making |

Every question holds one `<ts>` marker where the series goes.

### Train and test files

The train file is the paper's training set, followed by copies of most test lines. The copies are
one block at the end of the train file. A labelled pair is a pair whose answer is not empty (see
[Empty answers](#empty-answers)). Line numbers count from 1:

| Part | Lines | Pairs | Labelled pairs |
| --- | --- | --- | --- |
| `train_qa.jsonl` lines 1 to 9,252: the paper's training set | 9,252 | 73,248 | 72,341 |
| `train_qa.jsonl` lines 9,253 to 13,291: copies of test lines | 4,039 | 28,317 | 27,955 |
| `test_qa.jsonl`, lines that the train file copies | 4,039 | 28,317 | 27,955 |
| `test_qa.jsonl`, lines only in the test file | 1,500 | 14,160 | 14,022 |

The copies are not a random sample of the test file. They hold 2,689 of the 3,384 multi-row test
lines and 1,350 of the 2,155 single-row test lines. [Source issues](#source-issues) gives the
counts per stage and compares them with the paper.

### The four tasks

The paper defines the tasks in Section 5.1 and Appendix C:

| Stage | Task | Input | Format | Authors' metric |
| --- | --- | --- | --- | --- |
| 1 | Understanding | 1 row | open | BLEU, ROUGE |
| 2 | Perception | 1 row | multiple choice | accuracy, F1 |
| 3 | Reasoning | 10 rows | multiple choice | accuracy, F1 |
| 4 | Decision-Making | 10 rows | open | BLEU, ROUGE |

Understanding asks what a change in one named signal means, such as the altitude or the total
temperature at the LPT outlet. The closed questions use fixed wording. There are nine, and the
release prints each one with one option list:

| Stage | Question | Options |
| --- | --- | --- |
| 2 | Health status of the fan, LPC, HPC, HPT, LPT (5 questions) | a: Health, b: Fault |
| 2 | Fault cause | a to f: HPT, LPT, Fan efficiency modifier, then HPT, LPT, Fan flow modifier |
| 2 | Potential health state | a: Operating Normally, b: Abnormal Degradation, c: Imminent Failure |
| 3 | Qualitative condition of the engine | a to e: Good, Moderate, Poor, Bad, Extremely Poor Condition |
| 3 | Probability range of engine failure | a to e: 1%-10%, 10%-30%, 30%-50%, 50%-70%, 70%-100% |

An answer is one letter. The fault cause is the exception. It can hold several letters joined by
`", "`, such as `a, b, e`.

### The multi-row questions

The paper says that Reasoning and Decision-Making look at "10 consecutive cycles" to judge the
long-term health of the engine. In the one list that we traced, the 10 rows are windows in the middle
of 9 different flights, over about one day of flying. The ids of a list span a median of 145 rows.

The authors' code does not give the model the 10 whole rows. It takes the first 60 steps of each row
and joins them into one (600, 33) array (`TsQaDataset.__getitem__` at `b80142f`). The paper does not
describe this cut. It describes encoding each segment separately.

The three answers of a multi-row line carry about one label. The condition grade sets it. Grade e
never occurs. The failure probability follows the grade: a goes with a or b, b with b, c with c or e,
and d with d. The Decision-Making answer is maintenance advice in varied wording. Its content
follows the grade: preventive maintenance at grade a, urgent repair or replacement at grade d.

## The TimeNet dataset

### Records

The dataset holds two kinds of Record. Both have one Source, `N-CMAPSS engine`, with 33 float64
signals, `channel_00` to `channel_32`, in `seq_data` column order. The values are the published
values, unchanged. They have no unit.

- A row Record holds one row that a labelled single-row pair names: 5,945 of the 118,921 rows. Its
  signals are on a regular 1 Hz axis. If the flight ended inside the row, the signals stop at the last
  real step. The zeros that the release added are not stored. Its metadata holds `row_id`, the row's
  `data_ID`, and `padded_steps`, the number of zero steps left out. In the pinned release, 317 row
  Records end in padding, with fewer than 500 padded steps each.
- A joined Record holds the input of one multi-row question: the first 60 steps of each of its 10
  rows, in published order, 600 steps in all. There is one for each distinct list of 10 ids, 8,845
  in all. The 10 blocks come from different flights, so the signals use an ordinal axis: the steps
  have an order but no clock. If a flight ended within the first 60 steps of a row, the padded steps
  are null, with the missing-value reason `alignment_padding`. In the pinned release, no joined Record
  has a null. The metadata holds `row_ids` and `steps_per_row`, which is 60.

A row that both files or several lines name is one Record, and their tasks share it. To pad a row
Record back to the authors' array, add `padded_steps` zero steps at its end.

### Tasks

Each published QA pair with a label becomes one task of its stage, with one input Record. Its split is
the split of its file: `train_qa.jsonl` is train and `test_qa.jsonl` is test. The tasks come in file
order, train file first.

| Stage | TimeNet type | Input | Targets |
| --- | --- | --- | --- |
| Understanding | `AnswerTask` | row Record | the published answer |
| Perception | `ClassificationTask` | row Record | the published letters |
| Reasoning | `ClassificationTask` | joined Record | the published letter |
| Decision-Making | `AnswerTask` | joined Record | the published answer |

That gives 142,273 tasks: 100,296 train and 41,977 test.

The prompt is the published question, unchanged, with its `<ts>` marker and its options. The connector
adds no wording.

A closed question is a classification over letters. It always offers the same options in the same
order, so a letter has one meaning per question. The authors' model answers with the letters, and
their `closed_question_metrics` scores the set of letters. So the targets are the published letters,
one item per letter, in published order, such as `("a", "b", "e")`.

Each closed task has a label set: the allowed letters, with the option text of each letter as its
description. A `name` annotation names each label set. The five health-status questions share the
label set `component_health`. The other label sets are `fault_cause`, `health_state`,
`engine_condition` and `failure_probability`. Only `fault_cause` is multi-label.

### Task metadata

| Key | Value |
| --- | --- |
| `source_file`, `source_line`, `qa_offset` | where the pair is published. The line and offset are 0-based, as in the authors' evaluation |
| `stage` | the published stage, 1 to 4 |
| `closed_question` | the closed question, such as `hpc_health` or `fault_cause`. Closed tasks only |
| `source_issues` | a list of issue names, empty for most tasks. See [Source issues](#source-issues) |

Error messages count lines from 1, as editors do. `source_line` counts from 0, as the authors do.

### Dataset annotations

The annotation `unlabelled_pairs` holds the count of closed pairs with an empty answer, 1,769. Its
metadata `pairs` lists the place of each one. See [Empty answers](#empty-answers).

## Source issues

The connector keeps every labelled pair as published. It names two problems in the task metadata
`source_issues`, with the same names as the TSAQA connector:

| Issue | Meaning | Tasks |
| --- | --- | --- |
| `repeated_across_splits` | the other file has a pair with the same rows, question and answer | 27,961 train, 27,955 test |
| `repeated_question` | another pair in the same file has the same rows, question and answer | 12 train |

4,039 of the 5,539 test lines are byte copies of a train line. The test file matches the per-stage
counts in the paper's Fig. 8 exactly. The train file holds the paper's train counts plus those 4,039
copied lines:

| Stage | Train file | Copied from test | Paper's train (Fig. 8) |
| --- | --- | --- | --- |
| 1 Understanding | 41,128 | 10,800 | 30,328 |
| 2 Perception | 35,987 | 9,450 | 26,537 |
| 3 Reasoning | 16,300 | 5,378 | 10,922 |
| 4 Decision-Making | 8,150 | 2,689 | 5,461 |

These counts include the empty answers. So the release adds test lines to the paper's training set.
With the released train file, a model reaches a Reasoning test accuracy of 99.3% to 99.8%, as the
authors' issue [#34](https://github.com/Pandalin98/ITFormer-ICML25/issues/34) reports. The connector keeps the
released split. To train on the paper's split, leave out the train tasks whose `source_issues` holds
`repeated_across_splits`.

## Empty answers

1,769 closed answers are empty: the fault cause 1,054 times in train and 422 times in test, the
health state 215 times in train and 78 times in test. The connector treats an empty answer as a
missing label, and the pair gets no task. The authors' current metric raises an error on an empty
reference answer (`utils/metrics.py` at `b80142f`).

An empty fault cause does not mean "no fault". The single-row ids fall into 12 contiguous blocks. In
each block, every row has the same fault cause, whether its components are answered healthy or not.
The blocks with an empty fault cause are those where only HPC, or LPC and HPC, are answered `b:
Fault`. None of the six options names an HPC or LPC modifier, so these rows have no correct option.
The empty health states have no such pattern.

The answers also disagree in one place. In the blocks whose fault cause is `b, e` (LPT efficiency and
flow), the component answered `b: Fault` is the LPC, never the LPT. The connector keeps both answers
as published.

## What the connector checks

If the content is wrong, conversion fails. For a QA line, the message names the file and the line.
Conversion fails in these cases:

- The signals file does not hold exactly the HDF5 datasets `seq_data` and `data_ID`, with `seq_data`
  as float64 shaped (rows, 600, 33) and `data_ID` as the integers 1 to the number of rows.
- A QA file has another number of lines, pairs per stage or empty answers per question than the
  pinned release. This catches a truncated or empty file.
- A QA line is not one JSON object with exactly the keys `id`, `name` and `conversations`, or a turn
  has other keys, values or roles than the release uses. The validated JSON Lines reader raises
  `TimeNetConnectorError` for these.
- A `name` does not repeat its `id`, an id is not a row of the file, or a list does not hold 10
  distinct ids.
- The conversation does not alternate a `human` question and a `gpt` answer.
- The stage, the `open` or `close` attribute and the number of rows disagree, or a question does not
  hold one `<ts>` marker.
- A closed question is not one of the nine, or prints other options than its label set. Other options
  change what a letter means.
- A closed answer is not distinct letters of its options, or holds several letters for a single-label
  question.
- An open answer is empty.
- A named row holds only zeros, or a step of 33 zeros before its last real step. Padding is only at
  the end of a row.

Conversion reads the last step of each named row. If all 33 values are zero, it reads the whole row to
find where the padding starts. The other values stay in the HDF5 file until the writer reads them.
When a signal reads its rows, it can fail. If the file cannot be opened, a row cannot be read, or a
row's padding changed since conversion, it raises `TimeFFormatError`. If a row holds a NaN or an
infinite value, it raises `TimeFValidationError`.

## How the connector follows the authors

| The authors (code at `b80142f`) | The connector |
| --- | --- |
| `_build_index` flattens each line into one sample per pair | one task per labelled pair |
| `seq_data[int(id) - 1]` | the row Record of row id `id` |
| first 60 steps of each of 10 rows, joined | the joined Record of the 10 ids |
| SFT target: the published answer | the target: the answer, or its letters |
| `closed_question_metrics`: the set of letters a to f | a `ClassificationTask` over a label set of letters |
| train on `train_qa.jsonl`, evaluate on `test_qa.jsonl` | `Split.TRAIN`, `Split.TEST` |
| `source_line`, `qa_offset` | `source_line`, `qa_offset` in task metadata |

The connector differs from the authors' input in one way: it does not store the padding zeros. A row
Record stops at the last real step. A joined Record has nulls where the authors' array has zeros.

## Comparing with the paper and the card

- Channels: the paper says 32 channels. The release has 33. The authors' audit cannot tell which
  column is extra, so a dropped column is a guess. The connector keeps all 33.
- Pairs: the paper gives both 11k and 110k pairs, and the Hub card gives 87k train and 25k test
  pairs. The files hold 101,565 train and 42,477 test pairs. See [Source issues](#source-issues) for
  how they relate to Fig. 8.
- Reasoning: the paper says Reasoning predicts "the probability of future failures and the remaining
  useful life" (Section 5.1). The release has no remaining-useful-life question. Reasoning asks for
  the condition and the failure probability.
- Cycles: the paper calls a row a "cycle" and also a "10-minute (600 points)" segment. A row is a
  window of a flight, not a whole flight. The paper says a multi-row question is about "10
  consecutive cycles". Every one of the 11,534 multi-row lists has gaps between its ids, and 256 of
  them mix two ranges of ids.
- Multi-row input: the paper encodes each segment separately. The code joins the first 60 steps of
  each row. The connector follows the code.

## Data limitations

- The train split holds copies of most of the test split. See [Source issues](#source-issues).
- The signals come from a simulator, are scaled, and have no channel names. The questions name
  signals such as altitude, but no Record says which channel is which.
- The paper says an LLM rewrote the questions and answers, and experts reviewed them (Appendix A). The
  Understanding and Decision-Making answers vary in wording for the same content.
- The fault cause describes the fault mode of a whole block of rows, not of one row. See
  [Empty answers](#empty-answers).
- The paper does not say how the authors derived the closed answers from the N-CMAPSS health labels.

## Open questions

- Channel names: which N-CMAPSS variable each of the 33 channels holds, and which one the paper's 32
  leaves out. The authors promise a mapping in issue
  [#9](https://github.com/Pandalin98/ITFormer-ICML25/issues/9).
- Flights: the release does not mark where a flight or an engine unit starts.

## Local build

```bash
uv run timenet-build build pandalin98/enginemt-qa --local-registry .timenet-registry
```

The build downloads 18.8 GB. After the download, it takes about two minutes and writes 1.47 GB.
