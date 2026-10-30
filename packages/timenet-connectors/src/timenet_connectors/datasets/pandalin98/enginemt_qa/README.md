# EngineMT-QA

EngineMT-QA asks questions about simulated aero-engine flights. Each question is about one 600-step cycle
or a group of 10 cycles, of 33 signals from NASA's N-CMAPSS simulator. The authors define four tasks, one
per stage: Understanding, Perception, Reasoning and Decision-Making. They release it with the ITFormer
paper (Wang et al., ICML 2025, [arXiv 2506.20093](https://arxiv.org/abs/2506.20093)) and their code,
[ITFormer-ICML25](https://github.com/Pandalin98/ITFormer-ICML25).

The connector builds one Record per cycle and one task per published question-answer pair, as the
authors' code reads them. Each statement below names its source: the paper, the dataset card on the
Hub, the released files, or the authors' code at `b80142f` (2026-07-19).

## The source data

The release is the Hub dataset
[pandalin98/EngineMT-QA](https://huggingface.co/datasets/pandalin98/EngineMT-QA), pinned at
`ea2051a15512212976c22c1a075ec549c7d48eef`. The authors' audit of channel counts
(`docs/ISSUE_23_CHANNEL_AUDIT.md`) checked the same revision and the same SHA-256 digests. It holds
three files:

| File | Size | Holds |
| --- | --- | --- |
| `time_series_data.h5` | 18.8 GB | `seq_data`: float64 shaped (118,921, 600, 33); `data_ID`: the ids 1 to 118,921 |
| `train_qa.jsonl` | 40.5 MB | 13,291 lines, 101,565 QA pairs |
| `test_qa.jsonl` | 17.0 MB | 5,539 lines, 42,477 QA pairs |

**Signals (released files).** Row `i` of `seq_data` has id `i + 1`, so a QA's id `n` is row `n - 1`,
which is how the authors' `TsQaDataset` reads it. The HDF5 file has no attributes and names no channel.
The authors' audit says not to name the 33 channels without a mapping from them, so the connector
doesn't. The values are z-scored: over a sample of 400 rows, every channel has a mean near 0 and a
standard deviation near 1. No row holds NaN or infinity.

**QA (released files).** Each line has an `id`, a `name`, and a `conversations` list of `human`
questions, each followed by its `gpt` answer. Here is one line, shortened to one of its 15 pairs:

```json
{
  "id": "7323",
  "name": "7323.csv",
  "conversations": [
    {"from": "human", "stage": "2", "attribute": "close",
     "value": "What is the health status of HPC of the given engine signal in one cycle?\n <ts> a: Health\n b: Fault\n"},
    {"from": "gpt", "value": "a"}
  ]
}
```

A line has one of two shapes. Every line of the pinned files has one or the other:

| Line | `id` | Pairs | train lines | test lines |
| --- | --- | --- | --- | --- |
| Single-cycle | one row id | 8 Understanding, then 7 Perception | 5,141 | 2,155 |
| Multi-cycle | a list of 10 row ids | 2 Reasoning, then 1 Decision-Making | 8,150 | 3,384 |

Every question holds one `<ts>` marker where the series goes.

**The four tasks (paper Section 5.1, Appendix C, and the card).**

| Stage | Task | Input | Format | Authors' metric |
| --- | --- | --- | --- | --- |
| 1 | Understanding | 1 cycle | open | BLEU, ROUGE |
| 2 | Perception | 1 cycle | multiple choice | accuracy, F1 |
| 3 | Reasoning | 10 cycles | multiple choice | accuracy, F1 |
| 4 | Decision-Making | 10 cycles | open | BLEU, ROUGE |

Understanding asks what the change in a named signal means, such as altitude or the total
temperature at the HPT outlet. Its wording varies from line to line. Decision-Making asks for
maintenance actions. The closed questions have fixed wording. Nine questions exist, and each is
published with exactly one text and one option list (released files):

| Stage | Question | Options |
| --- | --- | --- |
| 2 | Health status of the fan, LPC, HPC, HPT, LPT (5 questions) | a: Health, b: Fault |
| 2 | Fault cause: "caused by efficiency or flow problems?" | a to f: HPT, LPT, Fan efficiency modifier; HPT, LPT, Fan flow modifier |
| 2 | Potential health state | a: Operating Normally, b: Abnormal Degradation, c: Imminent Failure |
| 3 | Qualitative condition of the engine | a to e: Good, Moderate, Poor, Bad, Extremely Poor Condition |
| 3 | Probability range of engine failure | a to e: 1%-10%, 10%-30%, 30%-50%, 50%-70%, 70%-100% |

An answer is one letter, except for the fault cause. Its answers hold 1, 2, 3 or all 6 letters, joined
by `", "`, such as `a, b, e`.

## The TimeNet dataset

### Records

Each HDF5 row becomes one Record:

- One Source, `N-CMAPSS engine`, with 33 signals, `channel_00` to `channel_32`, in `seq_data` column
  order. The values are the published float64 values, unchanged. They are z-scored and have no unit.
- Each signal has 600 values on a regular 1 Hz axis that starts at 0. The paper describes the inputs as
  "10-minute (600 points) time-series segments" (Section 5.2). The HDF5 file states no rate.
- Typed metadata, `EngineMTQARecordMetadata`, which holds `row_id`: the row's `data_ID`, the id the QA
  files use. The release names no engine unit or flight, so the shared `subject_id` is unset.

By default the dataset holds the 7,755 rows that a QA pair names. With `EngineMTQAConnector.ALL_ROWS =
True` it holds all 118,921 rows, which the authors' encoder pretraining reads (`train_pretrain.py` reads
every row of `seq_data`). The extra Records have no task.

A row named by several lines, in train and in test, is one Record that their tasks share.

### Tasks

Each published QA pair with a label becomes one task. There are four kinds, one per stage, and each
task takes its split from its file:

| Kind | TimeNet type | Inputs | Target | train | test |
| --- | --- | --- | --- | --- | --- |
| `UnderstandingTask` | `AnswerTask` | 1 Record | the published answer | 41,128 | 17,240 |
| `PerceptionTask` | `ClassificationTask` | 1 Record | the published letters | 34,718 | 14,585 |
| `ReasoningTask` | `ClassificationTask` | 10 Records | the published letter | 16,300 | 6,768 |
| `DecisionTask` | `AnswerTask` | 10 Records | the published answer | 8,150 | 3,384 |

That is 142,273 tasks: 100,296 train and 41,977 test. The 1,769 Perception pairs with an empty
answer have no task (see [Data limitations](#data-limitations)).

**Prompt.** The prompt is the published question, unchanged, with its `<ts>` marker and its options.
The authors send it as the user message, after the system message `You are a helpful assistant.`
Their `add_adaptive_prompt`, which would append a stage hint, is commented out in `__getitem__`, so the
prompt carries no hint.

**Closed questions are classifications over letters.** Each closed question always offers the same
options in the same order, so a letter has one meaning per question. The authors' model answers with
the letters, and `closed_question_metrics` scores the set of letters. So the targets are the published
letters, one item per letter, in published order, such as `("a", "b", "e")`. `target_schema` names the
question's vocabulary annotation. Task metadata `closed_question` names the question, such as
`hpc_health` or `fault_cause`.

**Multi-cycle inputs.** A Reasoning or Decision-Making task has its 10 Records as inputs, in published
order. The order is the release's, which isn't sorted by id in 256 of the 11,534 lists. The authors'
model reads only the first 60 steps of each row and concatenates them into one (600, 33) array
(`seq_data[int(i) - 1][: 600 // 10]` in `TsQaDataset.__getitem__`). One task scope can't cut 10 inputs,
so the task holds the whole Records, and its metadata states `steps_per_cycle: 60`.
`model_input.authors_series(task)` rebuilds the authors' array from the task's Records.

**Metadata.** Every task names its pair as the authors' evaluation does: `source_file`, `source_line`
(0-based) and `qa_offset` (the 0-based pair within the line).

### Dataset annotations

| Key | Value |
| --- | --- |
| `source_repository`, `source_revision` | the pinned Hub source |
| `component_health_vocabulary` | `["a", "b"]`; metadata `options` maps a to Health, b to Fault |
| `fault_cause_vocabulary` | `["a", ..., "f"]` with the six modifiers |
| `health_state_vocabulary` | `["a", "b", "c"]` with the three states |
| `engine_condition_vocabulary` | `["a", ..., "e"]` with the five conditions |
| `failure_probability_vocabulary` | `["a", ..., "e"]` with the five ranges |

The five health-status questions share `component_health_vocabulary`.

## How the connector follows the authors

| The authors (code at `b80142f`) | The connector |
| --- | --- |
| `_build_index` flattens each line into one sample per pair | one task per labelled pair |
| `seq_data[int(id) - 1]` | the Record of row id `id` |
| first 60 steps of each of 10 rows, concatenated | 10 input Records, `steps_per_cycle: 60`, `authors_series` |
| SFT target: the published answer | the target: the answer, or its letters |
| `closed_question_metrics`: the set of letters a to f | a `ClassificationTask` over letters |
| train on `train_qa.jsonl`, evaluate on `test_qa.jsonl` | `Split.TRAIN`, `Split.TEST` |
| `sample_index`, `source_line`, `qa_offset` | `source_line`, `qa_offset` in task metadata |

TimeNet ships no parser or metric for the answers.

## Comparing with the paper

- **Size.** The paper says "over 110k" QA pairs (Section 1) and "11k QA pairs" (Section 5.1 and Figure
  3). The card says 87k train and 25k test, 113k in all. The released files hold 101,565 and 42,477,
  144,042 in all, which the authors' README and audit state. The connector follows the files.
- **Channels.** The paper says 32 channels (Sections 1 and 5.2, Appendix D). The release has 33. The
  authors' audit says it can't tell which column differs, and that dropping one would be a guess. The
  connector keeps all 33.
- **Reasoning.** The paper says Reasoning predicts "the probability of future failures and the
  remaining useful life" (Section 5.1). The release has no remaining useful life question: Reasoning asks
  for the condition and the failure probability. `utils/metrics.py` has an unused `compute_rul`.
- **Time.** The paper calls an input both a "10-minute (600 points)" segment (Section 5.2) and a cycle
  of "600 time steps" (Appendix D). N-CMAPSS flights last hours. Within a row, channel 0 spans a median
  of 0.28 standard deviations, against -1.89 to 2.28 across 2,000 rows. So a row is a window of a
  flight, not a whole flight resampled. The 1 Hz axis rests on the paper.

## Data limitations

- **Empty closed answers.** 1,769 Perception answers are the empty string: the fault cause in 1,054
  train and 422 test pairs, and the health state in 215 train and 78 test pairs. They are missing labels.
  An empty fault cause appears whether or not all five components are healthy, so it doesn't mean "no
  fault". The health state takes exactly one option. The authors trained on these pairs, and their
  metric before `b80142f` scored an empty answer as an empty set. Their current metric raises on an
  empty reference, so it fails on the test file's Perception references, 500 of which are empty. The connector
  leaves them out and pins their count per question and file.
- **Train and test share rows.** 3,223 of the 7,755 rows appear in both files. They include 1,350 rows
  that a single-cycle line names in each file, and all 1,900 rows of the multi-cycle lists, which both
  files draw from. 2,689 of the 3,384 test lists of 10 ids (79%) are also train lists, and each has the
  same two Reasoning answers in both files. The split is the authors' and is kept. A test score is not a
  score on unseen signals, and for Reasoning it is mostly a score on train questions.
- **Repeated id.** Two train lines, 2,003 and 10,533 (1-based), are both about row id 40034. They ask
  different questions, and both are kept.
- **Simulated, normalized, unnamed.** The signals come from a simulator, are z-scored with statistics
  the release doesn't give, and have no channel names. The questions name signals such as altitude,
  but no Record says which channel is which.
- **Generated text.** The paper says an LLM rewrote the questions and answers (ChatGPT, DeepSeek), and
  experts checked them (Appendix A). The Understanding and Decision-Making answers vary in wording for
  the same content.

## Data flow and validation

```text
time_series_data.h5 --check_layout--> RowLoader --> one lazy Record per row
train_qa.jsonl, test_qa.jsonl --read_pairs--> QAPair --> one task per pair
```

1. `signals.check_layout` checks the HDF5 file. It must hold exactly `seq_data` and `data_ID`, of the
   published shape and dtype, with the ids 1 to 118,921 in order.
2. `qa.read_pairs` validates every line strictly. The keys are exact, and `name` must repeat `id`. A
   list of ids holds 10 distinct ids. The conversation alternates `human` and `gpt` turns. A stage
   decides the format (`open` or `close`) and the number of rows. Each question holds one `<ts>`. A
   closed question must be one of the nine, with its exact text. A closed answer must be distinct
   letters of its options, and one letter where one is expected. An open answer must not be empty.
3. The loader counts each file's lines, pairs per stage, and empty answers by question, and compares
   them with `release.PINNED_RELEASE`. It also compares the number of rows that QAs name. Any
   difference raises `TimeFValidationError`.
4. Each signal reads its row when the writer asks for it, one row per Record. A row that the file no
   longer holds raises `TimeFFormatError`.

## Open questions

- **License.** The Hub metadata says MIT, but the card's License section is an unfilled placeholder.
  The repository's `LICENSE` is MIT and covers the code. The paper names no data license. The card
  here says MIT. Confirm with the authors.
- **Channel names.** Which N-CMAPSS variable each of the 33 channels holds, and which one the paper's
  32 leaves out.
- **Sampling rate.** Whether a row is a 10-minute window at 1 Hz, as the paper says, and where in its
  flight the window starts.
- **Cycles.** Whether the 10 rows of a multi-cycle question are consecutive cycles of one engine unit.
  The release names neither the unit nor the cycle.

## Local build

```bash
uv run timenet-build build pandalin98/enginemt-qa --out .timenet-registry
```

The build downloads 18.8 GB. The default dataset holds 7,755 Records and 142,273 tasks, and takes about
900 MB as TimeF. Writing it takes about a minute after the download. With `ALL_ROWS = True` it holds
118,921 Records and the same tasks, and takes about 13 GB and 12 minutes.
