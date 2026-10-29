# SensorCaps

SensorCaps is the set of IMU–caption pairs from the LLaSA paper
([Imran et al.](https://arxiv.org/abs/2406.14498), Section 2.2, "SensorCap: Sensor Data Captioning").
Each pair holds accelerometer and gyroscope readings from HHAR, MotionSense, PAMAP2, Shoaib, or UCI,
and a caption that GPT-4o-mini generated from those readings, the activity label, derived features,
and, for PAMAP2, the sensor location.

The connector imports the v2 files of the
[Hugging Face dataset](https://huggingface.co/datasets/BASH-Lab/SensorCaps/tree/4242e07feaacb0ee3aff86182a62914fa7abacef):
35,960 rows, the count the paper gives. The v1 files, a shorter captioning run that the paper does not
describe, are not imported. Each row becomes one TimeF record and one captioning task.

## Records and captioning tasks

Each task repeats how GPT-4o-mini generated its caption: the row's system message is the prompt,
and the user message's contents are the task's inputs.

| Source row | TimeF access |
| --- | --- |
| System message: the generation instruction | `task.prompt`, verbatim |
| `Summary:` (the activity label) | `summary` input annotation |
| `Gyroscope:` and `Accelerometer:` readings | Signals in `task.inputs[0]` |
| Derived features: moving average, FFT, minimum, maximum, median, variance | 12 input annotations, such as `gyroscope_fft` |
| `Location:` (PAMAP2 only) | `location` input annotation |
| Assistant message: the feature summary and narration | `task.targets[0]`, verbatim |
| Source partition and row | `record.metadata`: `partition` and `row` |

The input annotations are the record's annotations, in the user message's order. Their keys follow
the labels the instruction names. Feature text keeps the original NumPy printing, including complex
numbers and ellipses. The connector does not parse or recompute it, and the narrations refer to it.
Captions are kept without splitting or truncation.

The tasks have no split. Table 1 of the paper assigns whole source datasets to training or testing,
and marks PAMAP2 as both, so the release gives no split per row.

Every row of a file carries the same instruction, which the connector checks:

> Please consider yourself to be an expert on gyroscope and accelerometer sensor information given as
> a metadata of IMU datasets.You are given the IMU sensor readings of a human activity. The user also
> provides a brief summary of the event followed by 'Summary:'. They also give you gyroscopic and
> accelerometer sensor data followed by 'Gyroscope:' and 'Accelerometer:' respectively. They are
> written in a Python list of lists format and contain x, y, and z axis data respectively. Please pay
> attention to the values and the signs. Additional information computed for each axis is provided
> afterwards, with the windows for moving averages being 12. You should provide a comprehensive
> details of at least a couple of characteristic IMU features for that event would be within 20-25
> words, followed by 'Features:'. The IMU features should be concise and descriptive. Separate
> multiple features with commas. Derive the list of features based on the given sensor data. Then,
> narrate the temporal event with details that are context-aware based on the sensor data, followed
> by 'Narration:', in a step-by-step fashion, analyzing it within 500 words or less. Please analyze
> even the small movements thoroughly in a logical and contextual manner, utilizing deductive thought
> process while being aware of the knowledge regarding the meaning of the sensor data. Use
> descriptive terms that aren't vague.

PAMAP2's instruction adds, before "You should provide": "Finally, the sensor location is provided
after 'Location:'."

## Signals

Each record has one source named `IMU` with six signals of 60 float64 values:
`accelerometer_x`, `accelerometer_y`, `accelerometer_z`, `gyroscope_x`, `gyroscope_y`, and
`gyroscope_z`. The names are the source's labels, for every partition. TimeF orders signals by
name, so the accelerometer comes first, although the user message gives the gyroscope first.

```python
import numpy as np

signals = task.inputs[0].sources[0].signals
values = np.column_stack([signal.to_numpy() for signal in signals])
context = {a.key: read_value(a.value) for a in task.input_annotations}
```

Signals use a regular 10 Hz axis, the rate the paper and the dataset card state. Values are stored
as float64, which keeps every six-decimal value the generator read and the captions quote; float32
would change some PAMAP2 values at the sixth decimal. Units are unknown: neither the paper nor the
dataset card states them.

## Data limitations

These notes come from checks against the original recordings and the authors' code, not from the
paper. The connector changes no data because of them.

**PAMAP2 channels.** The published PAMAP2 arrays are shifted by one raw column relative to
PAMAP2's XYZ groups. Checked against all 9,672 rows and
[PAMAP2's column layout](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring),
the six arrays labelled accelerometer and gyroscope hold, in order: the ±16 g accelerometer's y and
z, the ±6 g accelerometer's x, the gyroscope's y and z, and the magnetometer's x. The signal names,
the instruction, the features, and the captions all follow the source's labels, so captions may make
incorrect claims about particular axes or sensor types.

**PAMAP2 missing data.** Original PAMAP2 recordings mark missing measurements as NaN. Of the 9,672
PAMAP2 rows, 3,231 correspond to windows with missing values in the included channels. The
published values are consistent with zero filling before resampling, which spreads the inserted
zeros across each window. Finite values therefore do not imply complete measurements, and individual
zeros cannot be treated as missing readings. The connector exposes no missingness mask.

**Timing.** The authors'
[generation script](https://github.com/BASHLab/LLaSA/blob/bf4d00df0395a61d01945455fb53e41b77933a40/llasa_v2_data_generation.py)
selects the 60 readings at `round(linspace(0, 119, 60))` from a 120-reading window at 20 Hz, so one
step spans three source readings instead of two. The axis follows the paper's 10 Hz.

**Derived features.** The instruction calls them "computed for each axis", but each minimum,
maximum, median, and variance in the release is a list of 120 values, not one per axis. The
authors' script computes them across the three axes of each reading of the 120-reading window, not
from the 60 published readings, so they cannot be recomputed from the signals.

**LLaSA training tensors.** These captioning inputs differ from LLaSA's sensor-encoder training
tensors. The
[training preparation notebook](https://github.com/BASHLab/LLaSA/blob/bf4d00df0395a61d01945455fb53e41b77933a40/OpenSQA-Parsing-per-Dataset.ipynb)
loads separate sensor arrays and divides acceleration by 9.8.
