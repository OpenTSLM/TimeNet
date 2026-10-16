# SensorCaps

SensorCaps contains 35,960 sensor-caption pairs drawn from HHAR, MotionSense, PAMAP2, Shoaib, and
UCI activity recordings. Each example provides six sensor channels, activity context, derived
features, and a GPT-4o-mini feature summary and narration.

The connector imports the [Hugging Face dataset](https://huggingface.co/datasets/BASH-Lab/SensorCaps/tree/4242e07feaacb0ee3aff86182a62914fa7abacef)
as one TimeF record and one captioning task per example. All tasks belong to the training split;
source partitions identify the contributing datasets, not train/test folds.

## Records and captioning tasks

Each record contains one source named `IMU` with six signals. Record annotations identify the source
dataset, partition, file, and row, along with the activity summary and sensor location when available.

| Content | TimeF access |
| --- | --- |
| Sensor readings | Signals in `task.inputs[0]` |
| Captioning instruction | Beginning of `task.prompt` |
| Activity label | `activity_summary` in `task.input_annotations` |
| Sensor location, when available | `sensor_location` in `task.input_annotations` |
| Moving averages, FFT, minimum, maximum, median, variance | Multiline feature text after the instruction in `task.prompt` |
| Feature summary and narration | Complete response in `task.targets[0]` |

Include the task's input annotations when using the captioning task: the activity label is an
intended input. Feature text is retained verbatim, including complex numbers and NumPy ellipses.
These features were computed before subsampling and cannot be regenerated from the 60 readings
alone. Caption responses are retained without splitting or truncation.

## Signals

Each signal contains 60 float32 values. HHAR, MotionSense, Shoaib, and UCI records contain
`accelerometer_x`, `accelerometer_y`, `accelerometer_z`, `gyroscope_x`, `gyroscope_y`, and
`gyroscope_z`, in that order.

PAMAP2 records contain the following physical channels, in order:

| Signal name | Spec type |
| --- | --- |
| `accelerometer_16g_y` | `accelerometer` |
| `accelerometer_16g_z` | `accelerometer` |
| `accelerometer_6g_x` | `accelerometer` |
| `gyroscope_y` | `gyroscope` |
| `gyroscope_z` | `gyroscope` |
| `magnetometer_x` | `magnetometer` |

`16g` and `6g` identify the accelerometers' full-scale ranges, not the stored value units. PAMAP2
records contain this mixed set of channels rather than complete accelerometer and gyroscope XYZ
triplets. Physical identities follow [PAMAP2's column definitions](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
and matching against the original recordings.

To obtain a time-major `(60, 6)` array and the task context:

```python
import numpy as np

signals = task.inputs[0].sources[0].signals
values = np.column_stack([signal.to_numpy() for signal in signals])
context = {a.key: a.value for a in task.input_annotations}
```

Values retain the HF dataset's channel order and undergo only float32 conversion with normal binary
rounding. Units are unspecified because the dataset does not establish consistent physical units
across sources.

## Timing

Signals use an irregular time axis with reconstructed relative offsets:
`round(linspace(0, 119, 60)) / 20` seconds, spanning 0 to 5.95 seconds. This models selection of 60
readings from a 120-reading window at 20 Hz and is applied to all five source partitions, including
PAMAP2. The offsets are reconstructed rather than measured acquisition timestamps or a perfectly
regular 10 Hz cadence.

Dataset annotations record the assumed source rate, window length, reconstruction formula, and
the HF dataset's declared 10 Hz rate. The selection logic is described in the
[general generator](https://github.com/BASHLab/LLaSA/blob/bf4d00df0395a61d01945455fb53e41b77933a40/llasa_v2_data_generation.py)
and [PAMAP2 generator](https://github.com/BASHLab/LLaSA/blob/bf4d00df0395a61d01945455fb53e41b77933a40/llasa_v2_data_generation_pamap2.py).

## Data limitations

PAMAP2's captioning instructions, feature headings, and captions describe the channels as
accelerometer XYZ and gyroscope XYZ. Those descriptions do not match the physical channels listed
above. Captions may therefore contain incorrect claims about particular axes or sensor types.

Original PAMAP2 recordings contain missing measurements marked as NaN. Of the 9,672 PAMAP2 examples,
3,231 correspond to windows with missing values in the included source channels. These gaps are
not exposed in the HF arrays, whose values are consistent with zero filling before resampling.
Resampling spreads the effect of inserted zeros across each window, so finite values do
not imply complete measurements and individual zeros cannot be treated as missing readings.
Neither a missingness mask nor a per-record missingness flag is available through this connector.

The connector retains all HF examples, sensor values, feature text, and caption targets for benchmark
comparability, subject to float32 rounding. It performs no zero filling, resampling, interpolation,
or substitution of original recordings. Using original signals or a different missing-data policy
would change the inputs behind the features and captions and constitute a different benchmark input.

These captioning inputs also differ from LLaSA's sensor-encoder training tensors. The
[training preparation notebook](https://github.com/BASHLab/LLaSA/blob/bf4d00df0395a61d01945455fb53e41b77933a40/OpenSQA-Parsing-per-Dataset.ipynb)
loads separate sensor arrays and divides acceleration by 9.8. Reproducing those tensors requires
the separate source data and training preprocessing.
