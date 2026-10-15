# SensorCaps v2

This connector imports all 35,960 v2 examples from the pinned
[Hugging Face release](https://huggingface.co/datasets/BASH-Lab/SensorCaps/tree/4242e07feaacb0ee3aff86182a62914fa7abacef).
Each example is one six-channel IMU record and one captioning task. All tasks belong to the training
split; source partitions identify the five contributing datasets, not train/test folds.

## Published input and output

The mapping follows the upstream
[general generator](https://github.com/BASHLab/LLaSA/blob/bf4d00df0395a61d01945455fb53e41b77933a40/llasa_v2_data_generation.py)
and
[PAMAP2 generator](https://github.com/BASHLab/LLaSA/blob/bf4d00df0395a61d01945455fb53e41b77933a40/llasa_v2_data_generation_pamap2.py).

| Upstream content | TimeF access |
| --- | --- |
| System instruction | Beginning of `task.prompt` |
| Gyroscope and accelerometer xyz arrays | Six named signals in `task.inputs[0]` |
| `Summary` activity label | `activity_summary` in `task.input_annotations` |
| Optional `Location` | `sensor_location` in `task.input_annotations` |
| Moving averages, FFT, minimum, maximum, median, variance | Original multiline text after the system instruction in `task.prompt` |
| Assistant `Features` and `Narration` | Complete, unchanged response in `task.targets[0]` |

Consumers must include the task's input annotations as context when reproducing the captioning task:
the source activity label is an intended input. The original textual feature arrays are preserved,
including complex numbers and NumPy ellipses. They were computed before subsampling and should not
be regenerated from the 60 exported readings. The connector does not split or truncate the response;
v2 generation used `max_tokens=1500`.

Each signal has 60 float32 values. To obtain a time-major `(60, 6)` array in the upstream channel
order, given a captioning `task`:

```python
import numpy as np

signals = {signal.name: signal for signal in task.inputs[0].signals}
names = [
    f"{sensor}_{axis}"
    for sensor in ("accelerometer", "gyroscope")
    for axis in ("x", "y", "z")
]
values = np.column_stack([signals[name].to_numpy() for name in names])
context = {a.key: a.value for a in task.input_annotations}
```

Values are converted to float32 with normal binary rounding. No additional normalization or unit
conversion is applied; the release does not establish consistent physical units across sources.

## Timing and LLaSA training inputs

All five partitions, including PAMAP2, use the same timing interpretation. The paper and HF card
describe SensorCaps readings collectively as 10 Hz and state no PAMAP2 exception. The generators
select 60 readings across an input window; the released feature arrays support 120-sample windows
for PAMAP2 as well as the other sources.

The connector reconstructs relative offsets using `round(linspace(0, 119, 60)) / 20` seconds,
spanning 0 to 5.95 seconds. This applies the 20 Hz / 120-sample interpretation of the other sources
to PAMAP2 too. The source rate, window length, reconstruction, and HF card's declared 10 Hz rate
are dataset annotations. The offsets are reconstructed rather than measured acquisition timestamps
or a perfectly regular 10 Hz cadence.

The upstream
[training preparation notebook](https://github.com/BASHLab/LLaSA/blob/bf4d00df0395a61d01945455fb53e41b77933a40/OpenSQA-Parsing-per-Dataset.ipynb)
separately loads the original sensor arrays, divides acceleration by 9.8, and saves six-channel
`.npy` inputs for LLaSA. Those original arrays are not contained in the HF SensorCaps chat release.
This connector exposes the published SensorCaps captioning examples; reproducing those LLaSA
training tensors requires the separate original data and preprocessing.
