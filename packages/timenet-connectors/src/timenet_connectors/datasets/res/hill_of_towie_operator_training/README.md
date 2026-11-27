# Hill of Towie operator-response training

`res/hill-of-towie-operator-training@1.0.0` contains the selected data for the
audited operator-response run whose update-1,500 checkpoint was archived in
hackathon commit `f6e6450aa3adb90be1233a9258fbcd9edca1b710`.

| Split | Rows | Role |
| --- | ---: | --- |
| Train | 9,709 | Selected supervised fine-tuning examples |
| Validation | 172 | Checkpoint selection |
| Test | 371 | Held-out evaluation |

The run selected examples from a base continuation export and an audited
operator-response supplement, using seed 42. The per-category and label caps
were 240, 3, and 8 for train, validation, and test, with replay limits of 512,
24, and 0, respectively. The reported 46,689 training examples count exposures
over the run, including repeated examples. The selected training split contains
9,709 rows.

The underlying ten-minute SCADA data is attributed to A. Clerc and E. Lingkan,
released by RES on behalf of TRIG under CC BY 4.0.
[Zenodo source, version 2.1.0](https://doi.org/10.5281/zenodo.22662930)

## Build and load

```sh
uv run timenet-build build res/hill-of-towie-operator-training@1.0.0
```

The connector checks the SHA-256 of each split archive. If you already have
the three pinned files, place `train.jsonl.gz`, `validation.jsonl.gz`, and
`test.jsonl.gz` under
`${TIMENET_CACHE}/downloads/res/hill-of-towie-operator-training/` before
building.

```python
from timenet import TimeNet

with TimeNet().open("res/hill-of-towie-operator-training@1.0.0") as reader:
    first = next(reader.iter_tasks(split="train", batch_size=1))
    record = first.inputs[0]
    by_name = {signal.name: signal for signal in record.signals}
    series = [by_name[name] for name in record.metadata["channel_order"]]
```

Each task points to a six-hour or seven-day input window. Signal names and
record metadata preserve the training media names and order. The first timestamp marks the end of the
first ten-minute interval. A null marks a reading absent from the model input.
Ordered label sets hold classification choices. Answer tasks keep any choices
in `metadata["options"]`. The exact supervision is in `task.targets`, and
`metadata["label"]` holds the scoring label. Use the signals, prompt, and
options as model inputs.

The targets describe observed SCADA patterns and outcomes. This release
includes sampled held-out cases for model comparison. Their event frequencies
do not estimate field alert rates.
