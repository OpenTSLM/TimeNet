# EngineMT-QA

EngineMT-QA asks questions about simulated aero-engine flights. Each question is about one 600-step
cycle, or a group of 10 cycles, of 33 signals from NASA's N-CMAPSS simulator. The authors define four
tasks, one per stage: Understanding, Perception, Reasoning and Decision-Making. They released the
dataset with the ITFormer paper (Wang et al., ICML 2025,
[arXiv 2506.20093](https://arxiv.org/abs/2506.20093)) and their code,
[ITFormer-ICML25](https://github.com/Pandalin98/ITFormer-ICML25).

The connector builds one Record per cycle and one task per published question-answer pair, as the
authors' code reads them. It lives in one file, `connector.py`.

## What is verified on the real release

The connector's QA parser ran on the two pinned QA files. It accepts every line, and every count on
this page that comes from the QA files is from that run. The HDF5 header was read over HTTP. It holds
only `seq_data` and `data_ID`, as the table below shows, with no compression and no attributes.

Nobody has run a full build yet, so the 18.8 GB signals file has not been read end to end. Before you
rely on the dataset, run the build in [Local build](#local-build) once.

## The source data

The release is the Hub dataset
[pandalin98/EngineMT-QA](https://huggingface.co/datasets/pandalin98/EngineMT-QA), pinned at
`ea2051a15512212976c22c1a075ec549c7d48eef`. It holds three files:

| File | Holds |
| --- | --- |
| `time_series_data.h5` | `seq_data`: float64 shaped (118,921, 600, 33); `data_ID`: the ids 1 to 118,921 |
| `train_qa.jsonl` | 13,291 lines, 101,565 QA pairs |
| `test_qa.jsonl` | 5,539 lines, 42,477 QA pairs |

### Signals

Row `i` of `seq_data` has id `i + 1`. So a QA's id `n` is row `n - 1`, which is how the authors'
`TsQaDataset` reads it. The HDF5 file names no channel. The authors' channel audit
([`docs/ISSUE_23_CHANNEL_AUDIT.md`](https://github.com/Pandalin98/ITFormer-ICML25/blob/b80142f/docs/ISSUE_23_CHANNEL_AUDIT.md))
says not to name the 33 channels without a mapping, so the connector does not.

The release does not say how the values are scaled. In 24 random rows, each channel has a mean from
-0.29 to 0.28 and a standard deviation from 0.74 to 1.07. So the values look z-scored, with
statistics that the release does not give. This is a measurement, not a statement by the authors.

### QA

Each line has an `id`, a `name` and a `conversations` list of `human` questions. A `gpt` answer
follows each question. A line has one of two shapes:

| Line | `id` | Pairs |
| --- | --- | --- |
| Single-cycle | one row id | 8 Understanding, then 7 Perception |
| Multi-cycle | a list of 10 row ids | 2 Reasoning, then 1 Decision-Making |

Every question holds one `<ts>` marker where the series goes.

### The four tasks

The paper defines the tasks in Section 5.1 and Appendix C:

| Stage | Task | Input | Format | Authors' metric |
| --- | --- | --- | --- | --- |
| 1 | Understanding | 1 cycle | open | BLEU, ROUGE |
| 2 | Perception | 1 cycle | multiple choice | accuracy, F1 |
| 3 | Reasoning | 10 cycles | multiple choice | accuracy, F1 |
| 4 | Decision-Making | 10 cycles | open | BLEU, ROUGE |

The closed questions have fixed wording. There are nine, and the release prints each one with one
option list:

| Stage | Question | Options |
| --- | --- | --- |
| 2 | Health status of the fan, LPC, HPC, HPT, LPT (5 questions) | a: Health, b: Fault |
| 2 | Fault cause | a to f: HPT, LPT, Fan efficiency modifier; HPT, LPT, Fan flow modifier |
| 2 | Potential health state | a: Operating Normally, b: Abnormal Degradation, c: Imminent Failure |
| 3 | Qualitative condition of the engine | a to e: Good, Moderate, Poor, Bad, Extremely Poor Condition |
| 3 | Probability range of engine failure | a to e: 1%-10%, 10%-30%, 30%-50%, 50%-70%, 70%-100% |

An answer is one letter. The fault cause is the exception. It can hold several letters joined by
`", "`, such as `a, b, e`.

## The TimeNet dataset

### Records

The dataset holds one Record for each row that a labelled QA pair names: 7,755 of the 118,921 rows. A
row that both files or several lines name is one Record, and their tasks share it. Each Record has:

- One Source, `N-CMAPSS engine`, with 33 signals, `channel_00` to `channel_32`, in `seq_data` column
  order. The values are the published float64 values, unchanged. They have no unit.
- 600 values per signal on a regular 1 Hz axis. The rate comes from the paper's "10-minute (600
  points) time-series segments" (Section 5.2). The HDF5 file states no rate.
- Metadata `row_id`: the row's `data_ID`, which the QA files use as `id`. The release names no engine
  unit or flight, so the metadata holds nothing else.

The authors' encoder pretraining also reads the rows that no QA pair names. The connector leaves them
out, because they have no task.

### Tasks

Each published QA pair with a label becomes one task of its stage. Its split is the split of its file:
`train_qa.jsonl` is train and `test_qa.jsonl` is test. The tasks come in file order, train file first.

| Stage | TimeNet type | Inputs | Targets |
| --- | --- | --- | --- |
| Understanding | `AnswerTask` | 1 Record | the published answer |
| Perception | `ClassificationTask` | 1 Record | the published letters |
| Reasoning | `ClassificationTask` | 10 Records | the published letter |
| Decision-Making | `AnswerTask` | 10 Records | the published answer |

That gives 142,273 tasks: 100,296 train and 41,977 test.

The prompt is the published question, unchanged, with its `<ts>` marker and its options. The connector
adds no wording.

A closed question is a classification over letters. It always offers the same options in the same
order, so a letter has one meaning per question. The authors' model answers with the letters, and
their `closed_question_metrics` scores the set of letters. So the targets are the published letters,
one item per letter, in published order, such as `("a", "b", "e")`.

Each closed task has a label set: the allowed letters, with the option text of each letter as its
description. The five health-status questions share the label set `component_health`. The other
label sets are `fault_cause`, `health_state`, `engine_condition` and `failure_probability`. Only
`fault_cause` is multi-label.

A Reasoning or Decision-Making task has its 10 Records as inputs, in published order. The release does
not keep the lists sorted. The authors' model reads only the first 60 steps of each row and joins them
into one (600, 33) array (`TsQaDataset.__getitem__`). One task scope cannot cut 10 inputs, so the task
holds the whole Records. Its metadata states `steps_per_cycle: 60`. To build the authors' input, take
the first 60 steps of each Record.

### Task metadata

| Key | Value |
| --- | --- |
| `source_file`, `source_line`, `qa_offset` | where the pair is published. The line and offset are 0-based, as in the authors' evaluation |
| `stage` | the published stage, 1 to 4 |
| `closed_question` | the closed question, such as `hpc_health` or `fault_cause`. Closed tasks only |
| `steps_per_cycle` | 60. Multi-cycle tasks only |
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

The overlap is in the authors' release, not in the connector. The connector keeps the authors' split
as published, and only flags the repeated tasks.

Most of the test file repeats the train file. 4,039 of the 5,539 test lines are byte copies of a train
line. 27,955 of the 41,977 test tasks (67%) are also in train:

| Stage | Test tasks also in train |
| --- | --- |
| 1 Understanding | 10,800 of 17,240 |
| 2 Perception | 9,088 of 14,585 |
| 3 Reasoning | 5,378 of 6,768 |
| 4 Decision-Making | 2,689 of 3,384 |

A model that memorizes train scores well on these tasks. The authors' issue
[#34](https://github.com/Pandalin98/ITFormer-ICML25/issues/34) reports Reasoning test accuracy of 99.3%
to 99.8%. To score on unseen pairs, leave out the test tasks whose `source_issues` holds
`repeated_across_splits`.

## Empty answers

1,769 closed answers are empty: the fault cause 1,054 times in train and 422 times in test, the
health state 215 times in train and 78 times in test. The connector treats an empty answer as a
missing label, and the pair gets no task. The authors trained on these pairs.

An empty fault cause does not mean "no fault". 1,031 of the 1,476 empty fault causes are on lines
where at least one component is answered `b: Fault`. The answers also disagree the other way: 1,821
fault causes that are not empty are on lines where all five components are answered `a: Health`.

## What the connector checks

Conversion fails with `TimeFValidationError` when the content is wrong. For a QA line, the message
names the file and the line. Conversion fails when:

- The signals file does not hold exactly the HDF5 datasets `seq_data` and `data_ID`, with `seq_data`
  as float64 shaped (rows, 600, 33) and `data_ID` as the integers 1 to the number of rows.
- A QA file has another number of lines, pairs per stage or empty answers per question than the
  pinned release. This catches a truncated or empty file.
- A QA line has other keys than `id`, `name` and `conversations`, or a `name` that does not repeat its
  `id`. It also fails for an id that is not a row of the file, or a list that does not hold 10
  distinct ids.
- The conversation does not alternate a `human` question and a `gpt` answer.
- The stage, the `open` or `close` attribute and the number of rows disagree, or a question does not
  hold one `<ts>` marker.
- A closed question is not one of the nine, or prints other options than its label set. Other options
  would change what a letter means.
- A closed answer is not distinct letters of its options, or holds several letters for a single-label
  question.
- An open answer is empty.

The values stay in the HDF5 file until the writer reads them. When a signal reads its row, it fails
with `TimeFFormatError` if the file cannot be opened or the row cannot be read. It fails with
`TimeFValidationError` if the row holds a NaN or an infinite value. The values look z-scored, so they
must be finite.

The task stream reads each QA file a second time. Conversion keeps a 16-byte digest of each line, and
the second read compares each line with its digest. If a line changed or was removed, or the file
cannot be read, the stream fails with `TimeFFormatError`.

## How the connector follows the authors

| The authors (code at `b80142f`) | The connector |
| --- | --- |
| `_build_index` flattens each line into one sample per pair | one task per labelled pair |
| `seq_data[int(id) - 1]` | the Record of row id `id` |
| first 60 steps of each of 10 rows, joined | 10 input Records, `steps_per_cycle: 60` |
| SFT target: the published answer | the target: the answer, or its letters |
| `closed_question_metrics`: the set of letters a to f | a `ClassificationTask` over a label set of letters |
| train on `train_qa.jsonl`, evaluate on `test_qa.jsonl` | `Split.TRAIN`, `Split.TEST` |
| `source_line`, `qa_offset` | `source_line`, `qa_offset` in task metadata |

## Comparing with the paper and the card

- Channels: the paper says 32 channels. The release has 33. The authors' audit cannot tell which
  column is extra, so dropping one would be a guess. The connector keeps all 33.
- Pairs: the paper gives 11k or 110k pairs, and the Hub card gives 87k train and 25k test pairs. The
  files hold 101,565 train and 42,477 test pairs.
- Reasoning: the paper says Reasoning predicts "the probability of future failures and the remaining
  useful life" (Section 5.1). The release has no remaining-useful-life question. Reasoning asks for
  the condition and the failure probability.
- Cycles: the paper says a multi-cycle question is about "10 consecutive cycles" (Section 5.1). Every
  one of the 11,534 multi-cycle lists has gaps between its ids, and 256 of them mix two ranges of ids.
- Time: the paper calls an input both a "10-minute (600 points)" segment and a cycle of "600 time
  steps". N-CMAPSS flights last hours, so a row can be a window of a flight and not a whole flight.
  This is not confirmed.

## Data limitations

- The test split mostly repeats train. See [Source issues](#source-issues).
- The signals come from a simulator, are scaled, and have no channel names. The questions name
  signals such as altitude, but no Record says which channel is which.
- The paper says an LLM rewrote the questions and answers, and experts checked them (Appendix A). The
  Understanding and Decision-Making answers vary in wording for the same content.

## Open questions

- License: the Hub metadata says MIT, but the License section of the card is an unfilled placeholder.
  The paper names no data license. The dataset card here says MIT. Confirm with the authors.
- Channel names: which N-CMAPSS variable each of the 33 channels holds, and which one the paper's 32
  leaves out.
- Sampling rate: whether a row is a 10-minute window at 1 Hz, and where in its flight it starts.

## Local build

```bash
uv run timenet-build build pandalin98/enginemt-qa --out .timenet-registry
```

The build downloads 18.8 GB. It reads the 7,755 named rows, about 1.2 GB of values.
