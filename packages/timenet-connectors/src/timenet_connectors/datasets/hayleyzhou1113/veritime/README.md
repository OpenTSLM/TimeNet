# VeriTime

`hayleyzhou1113/veritime` holds TSRBench, the dataset of VeriTime ([arXiv:2602.07830][paper],
ICML 2026). It has time-series reasoning questions. Each question has a six-step chain of thought
(CoT) that DeepSeek-R1 wrote, and "process-verifiable" labels for steps 1, 2, 4 and 6, which the
authors' reinforcement-learning rewards score.

The connector reads the four JSON Lines files of the [Hugging Face release][card] at commit
`fc88cb1305a96a208093e13b69d766b19c84c99b`. The source license is Apache-2.0.

[paper]: https://arxiv.org/abs/2602.07830
[card]: https://huggingface.co/datasets/HayleyZhou1113/VeriTime

## The source data

The release has two families of rows, the paper's two task categories, and one file per split. A
build of the pinned revision gives these row counts:

| Task | Family | Train | Test | Answer |
| --- | --- | ---: | ---: | --- |
| Anomaly detection | Scenario | 1,016 | 180 | `Yes` or `No` |
| Scenario attribution | Scenario | 855 | 176 | One of the options the question offers |
| Inferential calculation | Scenario | 188 | 105 | A count |
| CTU (computer type usage) | Knowledge | 131 | 144 | `Desktop` or `Laptop` |
| ECG (rhythm) | Knowledge | 584 | 200 | `normal sinus rhythm`, `atrial fibrillation`, `other cardiac rhythms` or `noise` |
| EMG (neuromuscular condition) | Knowledge | 300 | 150 | `Healthy`, `Myopathy` or `Neuropathy` |
| RCW (right whale call) | Knowledge | 127 | 200 | `Right Whale Present` or `No Right Whale` |
| Total | | 3,201 | 1,155 | |

Scenario questions are ChatTS questions on synthetic series. Knowledge questions are on real ECG,
EMG, household electricity and right whale audio, and keep the splits of their source datasets.

Each row has these fields:

| Field | Meaning |
| --- | --- |
| `task` | The task's name, as in the table above |
| `question` | The question, with one `<ts><ts/>` marker where the series goes |
| `output` | Scenario rows only: the ChatTS answer that the label was extracted from |
| `label` | The ground-truth answer |
| `cot` | `<THINK>` six steps `</THINK><ANSWER>The answer is …</ANSWER>`, the authors' fine-tuning target |
| `step1_label`, `step2_label`, `step4_label`, `step6_label` | The process labels: task intent, key patterns (separated by `;`), preliminary answer, final answer |
| `timeseries2` | Scenario: a list with one list of numbers. Knowledge: one string of numbers separated by `", "` |

## The TimeNet dataset

### Records

ChatTS asks several questions about one series. So rows that publish equal values of the same kind
share one Record, and each row keeps its own task. A build gives 3,095 Records from 4,356 rows:
1,259 scenario, 784 ECG, 450 EMG, 327 RCW and 275 CTU. 617 Records have more than one row, and 241
have rows in both splits, so a scenario test question can ask about a series that the train split
also asks about. Treat the scenario test split as new questions, not new series.

Each Record has one Source with one lazy float64 signal. The signal rereads its first row from the
file when it loads, and fails if the values differ from the ones read at conversion. The source
files must stay in place until the dataset is written.

| Kind | Signal | Spec | Unit | Axis |
| --- | --- | --- | --- | --- |
| Scenario | `series` | `chatts_series` | unknown | ordinal |
| CTU | `consumption` | `electricity_consumption` | unknown, z-scores | 120 s |
| ECG | `ecg` | `ecg` | unknown | 150 Hz |
| EMG | `emg` | `emg` | mV | ordinal |
| RCW | `audio` | `hydrophone_audio` | unknown | 250 Hz |

**The axes give the rate of the values as published, not the rate the question states.** The ECG
question says 300 Hz and the RCW question says 2 kHz with 4,000 points. The published series have 500
values. They are every second ECG sample and every eighth RCW sample of the authors' originals, so
150 Hz and 250 Hz. CTU states its 2-minute cadence. The scenario series are synthetic, and the EMG
question states no rate.

- The originals were compared in the authors' GitHub copy, whose `timeseries` column holds them for
  test rows only. The train rows keep no original, so their decimation is assumed to be the same.
- The median R-R interval of 196 normal sinus rhythm rows is 132.5 samples, which is 68 bpm at
  150 Hz. At 300 Hz it would be 136 bpm, which is implausible.
- No low-pass filter ran before the decimation, so RCW is aliased above 125 Hz. Its average
  spectrum is flat: 9 to 10% of the power in each tenth of the band, with a median lag-1
  autocorrelation of 0.02.

Each CTU series is z-normalised on its own (mean 0, standard deviation 0.9993), so its values are
dimensionless and not electricity consumption. The signal and spec keep the name of the quantity.

Each Record has metadata with the series `kind` and `rows`, which lists every row that publishes
the series by `part` and `row` (its position in the file, from 0).

### Tasks

Each row has one task, and its type follows the answer:

| Rows | Task | Target | `label_set` |
| --- | --- | --- | --- |
| Anomaly detection | `ClassificationTask` | `Yes` or `No` | `anomaly_detection` |
| Scenario attribution | `AnswerTask` | The label as published | None: each question offers its own options |
| Inferential calculation | `ScalarPredictionTask` | The label as a number: an `int` count, or one `float` | None |
| CTU, ECG, EMG, RCW | `ClassificationTask` | The class name | `ctu`, `ecg`, `emg` or `rcw`. Knowledge tasks list their labels in the order of the options in their questions (a), b), and so on). Anomaly detection is `Yes`, `No`, as the authors' `extract_label.py` takes it |

Every task also has these fields:

| Field | Value |
| --- | --- |
| `prompt` | The question verbatim, with its `<ts><ts/>` marker |
| `inputs` | The Record of the row's series |
| `rationale` | The published `cot`, verbatim |
| `annotations` | `step1_label`, `step2_label`, `step4_label`, `step6_label`, and on scenario rows `source_answer`, the ChatTS `output` |
| `metadata` | `source_task`, `part` and `row`: the row the task comes from |
| `split` | The row's file: `train` or `test`. There is no validation split |

The process labels and `source_answer` are supervision, not model input, so no task names them in
`input_annotations`. A label published as null has no annotation. `step1_label` and `step4_label`
are null in the same 137 scenario rows. One inferential calculation label is a decimal: scenario
train row 1125 asks for a count of days and an amplitude, and its label `10641.02` is the amplitude.

### Dataset annotations

| Annotation | Value |
| --- | --- |
| `source_repository` | `HayleyZhou1113/VeriTime` |
| `source_revision` | The pinned commit |

## What the connector checks

Conversion reads each file once. It raises `TimeNetConnectorError`, with the file and row, if a
row:

- is missing a field, or a field is not nonblank text (a process label can also be null where the
  release publishes null);
- names a task the connector does not know, or a task of the other family's file;
- has a question without exactly one `<ts><ts/>` marker;
- has a classification label outside the task's classes, or an inferential calculation label that
  is not a plain number;
- has a `timeseries2` that does not parse, is empty, or holds a value that is not finite.

The pinned revision fixes the content, so the connector keeps no row counts.

## Source notes

- The authors' scripts replace the marker with the series as text, send `cot` as the answer they
  fine-tune on, and score against `label`. The connector keeps the marker in the prompt and the
  series in the Record.
- The system prompt of the authors names `<THINK>` and `<ANSWER>`, yet some published `cot` values
  use `<think>`. The connector keeps every `cot` as published.
- The scenario attribution label is the extracted first sentence of the ChatTS answer. It can keep
  an option letter, such as `A) System maintenance`, and can differ from the option in
  capitalization.


## Connector layout

`connector.py` coordinates conversion; `source.py` validates rows and retains descriptors for
lazy rereading. `records.py` builds records, `tasks.py` builds tasks, and `vocabulary.py` holds
the published constants and signal specifications. Metadata uses `SourceRow`; numeric samples
use the shared signal validator before conversion to float64, rejecting lossy integer casts.
The typed JSONL reader validates both scans and rereads, rejecting extra fields, duplicate keys
and blank lines with physical line locations.
