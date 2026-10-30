# VeriTime

`hayleyzhou1113/veritime` holds TSRBench, the dataset of VeriTime ([arXiv:2602.07830][paper],
ICML 2026). It has time-series reasoning questions, each with a six-step chain of thought (CoT)
that DeepSeek-R1 wrote. Each question also has "process-verifiable" labels for steps 1, 2, 4
and 6, which the authors' reinforcement-learning rewards score.

The connector reads the four JSON Lines files of the [Hugging Face release][card] at commit
`fc88cb1305a96a208093e13b69d766b19c84c99b`. The source license is Apache-2.0. The authors' code is
the [anonymous repository][code] that the paper cites. It has no commit to pin, so the copy read for
this connector is the archive downloaded on 2026-10-29, sha256 `233e6710…5c69d9`.

[paper]: https://arxiv.org/abs/2602.07830
[card]: https://huggingface.co/datasets/HayleyZhou1113/VeriTime
[code]: https://anonymous.4open.science/r/VeriTime-E017

## The source data

The release has two families of rows, the paper's two task categories, and one file per split:

| File | Train | Test | Tasks |
| --- | ---: | ---: | --- |
| `Dataset/Scenario/scenario_shuffled_{train,test}.jsonl` | 2,059 | 461 | Anomaly detection, Scenario attribution, Inferential calculation |
| `Dataset/Knowledge/knowldege_shuffled_train.jsonl`, `knowledge_test.jsonl` | 1,142 | 694 | CTU, ECG, EMG, RCW |
| Total | 3,201 | 1,155 | |

| Task | Train | Test | Answer |
| --- | ---: | ---: | --- |
| Anomaly detection | 1,016 | 180 | `Yes` or `No` |
| Scenario attribution | 855 | 176 | One of the options the question offers |
| Inferential calculation | 188 | 105 | A count |
| CTU (computer type usage) | 131 | 144 | `Desktop` or `Laptop` |
| ECG (rhythm) | 584 | 200 | `normal sinus rhythm`, `atrial fibrillation`, `other cardiac rhythms` or `noise` |
| EMG (neuromuscular condition) | 300 | 150 | `Healthy`, `Myopathy` or `Neuropathy` |
| RCW (right whale call) | 127 | 200 | `Right Whale Present` or `No Right Whale` |

Each row holds these fields. The connector reads them all:

| Field | Content |
| --- | --- |
| `task` | The task's name, as in the table above |
| `question` | The question, with one `<ts><ts/>` marker where the series goes |
| `output` | Scenario rows only: the ChatTS answer that the label was extracted from |
| `label` | The ground-truth answer |
| `cot` | `<THINK>` six steps `</THINK><ANSWER>The answer is …</ANSWER>`, the authors' fine-tuning target |
| `step1_label`, `step2_label`, `step4_label`, `step6_label` | The process labels: task intent, key patterns (separated by `;`), preliminary answer, final answer |
| `timeseries2` | Scenario: a list with one list of numbers. Knowledge: one string of four-decimal numbers separated by `", "` |

The Hugging Face files are the repository's `Dataset/` files without four columns: `id`,
`cot_deepseekr1`, `timeseries` and `timeseries3`. Every row is otherwise equal, field by field
and in the same order. In the repository's copy, one knowledge train row starts with a stray `c`,
and the Hugging Face copy fixes it. The release keeps no id, so no row names its upstream series.

### How the authors built it

The authors' `TSRBench/` scripts describe the pipeline:

- `classify_rule_based.py` sorts ChatTS SFT questions into the three scenario tasks by keyword.
- `extract_label.py` takes the label from the ChatTS answer and rounds the series to four
  decimals.
- `cot_deepseekr1.py` asks DeepSeek-R1 to follow a six-step template for each task.
- `cot_correct.py` keeps the rows whose step 6 matches the label, and extracts the step labels.
- `generate_cot.py` writes `cot`.
- `extract_step2label_from_output.py` has GPT-4o-mini extend `step2_label` with the patterns that
  the ChatTS answer names.

According to the paper, the authors split the scenario rows randomly, about 5:1. The knowledge
rows keep the train and test split of each source dataset. The authors' training code calls them
"TimeBed" data.

**Scenario rows are ChatTS rows.** In the repository's copy, a scenario row's `id` is its line in
ChatTS `sft/train.jsonl`. For all 2,520 scenario rows, `timeseries2` is that line's series rounded
to four decimals. The question and the answer are unchanged in 2,493 rows. The authors edited the
other 27, mostly from "significant" to "abnormal" or from "issue" to "anomaly", which their
anomaly keywords match.

## The TimeNet dataset

### Records

ChatTS asks several questions about one series. So the 2,520 scenario rows hold only 1,259
distinct series, and 617 of those series have more than one row. Rows that publish equal values
share one Record, and each row keeps its own task, as in TRACE. No two knowledge rows share a
series. That gives 3,095 Records: 1,259 scenario, 784 ECG, 450 EMG, 327 RCW and 275 CTU.

Each Record has one Source with one lazy float64 signal. A signal rereads its first row's byte
range when it loads, and checks the values against the converted row:

| Kind | Signal | Spec | Unit | Axis |
| --- | --- | --- | --- | --- |
| Scenario | `series` | `chatts_series` | unknown | ordinal |
| CTU | `consumption` | `electricity_consumption` | unknown | 120 s |
| ECG | `ecg` | `ecg` | unknown | 150 Hz |
| EMG | `emg` | `emg` | mV | ordinal |
| RCW | `audio` | `hydrophone_audio` | unknown | 250 Hz |

**The axes give the rate of the values as published, not the rate the question states.**

- The test file of the repository's copy keeps each knowledge original in `timeseries`.
- Over all 200 ECG test rows, `timeseries2` is every second sample of the 1,000 in `timeseries`,
  so 150 Hz, while the question says 300 Hz.
- Over all 200 RCW test rows, it is every eighth sample of 4,000, so 250 Hz, while the question
  says "sampled at 2kHz, resulting in a time series of length 4000".
- The train files keep no original. Their series have the same lengths (500), so the same
  decimation is assumed.
- CTU equals its original, and the question states its 2-minute cadence.
- EMG equals its original, and the question states no rate.

**Units.** Only the EMG question states a unit (mV). The CTU values are not consumption: every
CTU series has mean 0 and standard deviation 0.9993, so they are z-scores. The RCW values lie
between −10 and 9.9997, and 4 RCW series reach exactly −10.0 (16 values), which looks like
clipping. Scenario series are synthetic, and 227 of the 2,520 rows publish a constant series.

The Record's metadata is `records.VeriTimeRecordMetadata`: the series `kind` and `rows`, which lists
every row that publishes the series by `part` and `row` (its position in the file, from 0).
`records.record_metadata(record)` reads it back as a typed object. The release names no subject,
so the shared `subject_id` is stored as null.

### Tasks

Each row has one task. Each answer kind maps to the TimeNet type that fits it:

| Rows | Task | Target | `target_schema` |
| --- | --- | --- | --- |
| Anomaly detection | `ClassificationTask` | `Yes` or `No` | `anomaly_detection_vocabulary` |
| Scenario attribution | `AnswerTask` | The label as published | None: each question offers its own options |
| Inferential calculation | `ScalarPredictionTask` | The label as a number: an `int` count, or one `float` (see [Source issues](#source-issues-kept-as-published)) | None |
| CTU, ECG, EMG, RCW | `ClassificationTask` | The class name | `ctu_vocabulary`, `ecg_vocabulary`, `emg_vocabulary` or `rcw_vocabulary` |

Every task also has these fields:

| Field | Value |
| --- | --- |
| `prompt` | The question verbatim, with its `<ts><ts/>` marker |
| `inputs` | The Record of the row's series |
| `rationale` | The published `cot`, verbatim: the authors' fine-tuning target |
| `annotations` | `step1_label`, `step2_label`, `step4_label`, `step6_label`, and on scenario rows `source_answer`, the ChatTS `output` |
| `metadata` | `source_task`, `part` and `row`: the row the task comes from |
| `split` | The row's file: `train` or `test`. There is no validation split |

The process labels and `source_answer` are task annotations. They are supervision, which the
authors' rewards score, and no task names them in `input_annotations`. A label published as null
has no annotation: `step1_label` and `step4_label` are null in the same 137 scenario rows.

`loader.VeriTimeDatasetLoader` has one method for each of the seven tasks:
`anomaly_detection_tasks`, `scenario_attribution_tasks`, `inferential_calculation_tasks`,
`ctu_tasks`, `ecg_tasks`, `emg_tasks` and `rcw_tasks`. Its `tasks` yields every task in file order.
After `populate`, `TimeFDataset.get_split` selects the tasks of one split.

### Dataset annotations

| Annotation | Value |
| --- | --- |
| `source_repository` | `HayleyZhou1113/VeriTime` |
| `source_revision` | The pinned commit |
| `anomaly_detection_vocabulary` | `Yes`, `No` |
| `ctu_vocabulary`, `ecg_vocabulary`, `emg_vocabulary`, `rcw_vocabulary` | The task's classes, in the order its question offers them as options a), b), and so on |

### Rebuilding the authors' request

`VeriTime_sft.py`, `VeriTime_grpo.py` and `VeriTime_evaluate.py` send the same messages:

- The system message, `request.SYSTEM_PROMPT`.
- The question, with its marker replaced by the series as text. A scenario series is
  `", ".join(map(str, values))`. A knowledge series is the published string, which
  `f"{value:.4f}"` reproduces for every row.

`request.authors_request(task)` rebuilds the user message. On the full build it equals the
authors' text for all 4,356 tasks. The authors' answer is the task's rationale.

```python
from timenet.client import TimeNet
from timenet_connectors.datasets.hayleyzhou1113.veritime.request import (
    SYSTEM_PROMPT,
    authors_request,
)

dataset = TimeNet(registry="/tmp/veritime-registry").load(
    "hayleyzhou1113/veritime@1.0.0"
)
task = next(dataset.iter_tasks())
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": authors_request(task)},
    {"role": "assistant", "content": task.rationale.read()},
]
```

### Scoring as the authors do

TimeNet ships no parser or metric. The authors' code scores in these ways:

- **Final answer.** `check_final_answer` in `VeriTime_grpo.py` reads the text after
  "The answer is" inside `<ANSWER>`. It counts an answer as correct when the label, lowercased, is
  a substring of that text. So `C) other cardiac rhythms` scores against `other cardiac rhythms`.
- **Process.** `check_process_answer` matches step 1 by containment or fuzzy ratio, step 2 by
  fuzzy phrase matches against `step2_label` split on `;`, and steps 4 and 6 by label
  containment.
- **Selection.** `cot_correct.py` compares counts as integers and other answers by containment,
  after it strips punctuation and spaces.

The system prompt names `<THINK>` and `<ANSWER>`, and the GRPO format reward matches `</THINK>`
case-sensitively. Yet 409 published `cot` values use `<think>` (82 scenario, 327 knowledge).

## How the connector follows the authors

| Authors' setup | Connector choice |
| --- | --- |
| One question per row, scored against `label` | One task per row, with the label as target |
| The user message is the question with the series pasted in | The prompt keeps the marker, the Record holds the series, and `authors_request` rebuilds the message |
| The model is fine-tuned on `cot` | `rationale` is `cot` verbatim |
| The GRPO rewards score steps 1 and 2 against `step1_label` and `step2_label`, and steps 4 and 6 against `label`; the paper describes labels for steps 4 and 6 too | Every step label is a task annotation, never input |
| `timeseries2` is what the model sees | Signals hold `timeseries2`, not the anonymous repository's originals |
| Scenario split at random, knowledge splits from the sources | The files' own splits |

## Source issues, kept as published

- **Counts differ from the paper.** Table 9 gives 4,340 rows: 1,180 anomaly detection, 930
  scenario attribution, 410 inferential calculation, 270 CTU, 780 ECG, 450 EMG and 320 RCW. The
  release has 4,356 rows: 1,196, 1,031, 293, 275, 784, 450 and 327. The train rows, 3,201, match
  the paper's total.
- **Series shared across splits.** 241 scenario series appear in both splits. 326 of the 461
  scenario test rows ask about a series that 571 train rows also ask about, with another question.
  3 (series, question) pairs are in both splits, and 18 repeat a pair within the release. Treat
  the scenario test split as new questions, not new series.
- **Repeated ChatTS rows.** Three ChatTS lines each appear twice in scenario train, with different
  CoTs.
- **Decimated knowledge series.** The ECG and RCW questions state a rate and length that the
  series doesn't have (see [Records](#records)). The kept values are the original samples, so no
  filter ran before the decimation. At 250 Hz, an RCW series holds frequencies up to 125 Hz only,
  and content above that aliases. The question places up-calls at 60–250 Hz.
- **One decimal inferential calculation label.** Scenario train row 1125 asks "how many days … and
  what was the amplitude of this change?". Its label `10641.02` is the amplitude. Its CoT answers
  `40; 10641.02`, and the ChatTS answer says about 38 days. `cot_correct.py` would have rejected
  it. The target is the float `10641.02`.
- **Counts of every point.** Five inferential calculation labels, 256 (four rows) and 726, equal
  the length of their series. Each comes from a ChatTS answer that counts every point of the
  series. Three of those answers state twice the length (512, 512 and 1,452).
- **Scenario attribution labels.** The label is the extracted first sentence of the ChatTS answer.
  230 keep an option letter, such as `A) System maintenance`, and many differ from the option in
  capitalization.
- **CoTs that disagree with their labels.**
  - In 7 scenario rows, `cot` nests its tags or has an `<ANSWER>` that differs from
    `step6_label`.
  - In 15 rows, `step4_label` doesn't contain the label.
  - In 2 scenario attribution rows, `step6_label` doesn't contain the label, and their `cot` fails
    the authors' final-answer reward against it. All other 4,354 `cot` values pass it.
  - The CoT of scenario train row 1782 says the problem is "unsolvable as posed" because it saw
    only the series length.

## Comparing with the paper

The paper's Tables 1 and 2 report accuracy on the test rows of each task. Its counts differ from
the release's, so a result on this build is not the paper's test set exactly.
