# TRACE Weather Events

`trace/weather-events` holds the weather data of TRACE ([arXiv:2506.09114][paper], NeurIPS 2025).
The data is NOAA GHCN-hourly station measurements before NOAA Storm Events, with the authors' event
labels, generated descriptions, forecasting windows and splits.

The connector reads the [Google Drive archive][drive] that the authors' code loads
([TRACE-Multimodal-TSEncoder at `1b65246`][code]). Drive has no fixed revision, so the connector
pins the 487 MiB archive by its sha256.

[paper]: https://arxiv.org/abs/2506.09114
[drive]: https://drive.google.com/file/d/1hX4D91QbXa0UQlgf6Jnf-1ii96gfp1aY/view
[code]: https://github.com/Graph-and-Geometric-Learning/TRACE-Multimodal-TSEncoder/tree/1b6524630cb3079754e17f7c88c88d820645096a

## The source data

A window is one station's measurements over a time span: seven channels, one timestamp per value,
and one of three modes.

| `mode` | Cadence | Span | Typical length |
| --- | --- | --- | --- |
| `7day_hourly` | 1 hour | 7 days | 168 |
| `28day_4hourly` | 4 hours | 28 days | 168 |
| `180day_daily` | 1 day | 180 days | 180 |

An event is one report in the NOAA Storm Events Database: an id, one of nine event types, a county
and a narrative. For each event, the authors take three windows at a nearby station, one per mode,
that end before the event begins. These three windows are an anchor, with window ids `3k`, `3k+1`
and `3k+2`. A window without a following event has the label `-100`, which no task uses.

Paper Table 8 counts the windows:

| File | Train | Val | Test | Content |
| --- | --- | --- | --- | --- |
| `pretrain/*.json` | 45,339 | 12,953 | 6,484 | Event and non-event windows, with `event_type` |
| `forecasting/*.json` | 6,690 | 1,914 | 957 | Non-event windows |
| `retrieval/*.parquet` | 42,180 | | 2,006 | Generated descriptions and event reports |

The connector does not read the archive's `.npy` copies of the windows, or the `timeseries` column
of the `retrieval` files.

When several events follow the same days at one station, each event gets its own windows. These
windows hold the same measurements but have their own id, label, county, description and split.

## Records

A Record is one published window, as the authors count it: the paper's 74,337 windows give 74,337
Records. Windows with the same station, mode, timestamps and values are copies. The archive has
63,366 distinct measurements, so 10,971 Records copy another. 5,007 copies are in another split
than their original, and 2,027 have another event type. 13,823 Records have gaps. Each Record has one
Source, `station`, with seven Signals on one time axis:

| Channel | Unit | Spec |
| --- | --- | --- |
| `temperature` | °C | float32 |
| `precipitation` | mm | float32 |
| `relative_humidity` | % | float32 |
| `visibility` | km | float32 |
| `wind_u` | m/s, eastward | float32 |
| `wind_v` | m/s, northward | float32 |
| `sky_code` | GHCN-h sky cover code, 0–10 | `enum` codebook |

The units come from the field table of the Hugging Face card. The `sky_code` values follow
[NCEI's hourly coverage code table][ncei-sky]. Codes 00–08 count oktas (eighths of the sky that
cloud covers), 09 means that the sky is obscured, and 10 means partial obscuration.

[ncei-sky]: https://www.aos.wisc.edu/oldsco/stations/NCEI-hrly-doc_for_clds-BNA.txt

The Record metadata holds the `station` id, its `coordinates`, the `sampling`, and `window_start`
and `window_end`, the first and last timestamps as published. It also holds the window's
`window_id`, `anchor`, `source_file`, `county` and published `event_type`. For a copy, `copy_of` is
the lowest id of the other windows with the same measurements. It is `null` for a window without
copies, and for the copy with the lowest id. Metadata is never model input. A Record has no
annotations.

## Time

The timestamps are the station's local clock time, with no time zone. So a Record has no absolute
start time. Its time axis counts local-clock time from its first timestamp. To get UTC, find the
station's time zone from its `coordinates`. A window without gaps has a regular time axis at its
mode's cadence. A window with gaps keeps the time offset of each value.

## Tasks

Each window or row that qualifies is a task of its own, under the authors' split. The task uses the
window's Record. Its metadata holds its window id, anchor and source file.

| Task | One per | Model input | Target | Split | Count |
| --- | --- | --- | --- | --- | --- |
| `ClassificationTask` | `pretrain` event window with at least 100 values | The series | The event type | `pretrain` file | 14,446 |
| `TSCorrespondenceTask` | Row of the `retrieval` files | The description, as the prompt | The window's Record | `retrieval` file | 44,186 |
| `ForecastingTask` | `forecasting` window with at least 100 values | The series before its last 24 values | Those 24 values | `forecasting` file | 9,474 |

Event type: the nine labels follow `EVENT_MAP` in the authors' `src/common.py`. The rationale is the
event's report, as the authors' `generate_er` writes it. The report describes what happened after
the window, so it is supervision only. The task metadata adds the `event_id`.

Retrieval: this is the paper's Stage 2. The prompt is the description, as the authors'
`generate_dsp` composes it. The text must find the window's Record. The task's
`channel_description` annotations hold the seven channel sentences.

Test retrieval tasks name the Records of the test windows as their candidates, as the authors
rank them. A copy of the target has its own Record, so finding the copy counts as a miss, as in the
paper.

Train retrieval tasks have a known issue. They name no candidates, and TimeF reads that as the
whole dataset, test Records included. The authors train with in-batch negatives, so they rank
against no fixed pool. Do not draw negatives from the pool of a train task, because it includes
test series. TimeF composition also refuses these tasks. An explicit train pool would store about
1.8 billion references. The fix needs a TimeF state for "no fixed pool", which TimeF does not have
yet.

A description starts with its window's dates and county. The Record keeps these only in metadata,
so a model that sees the Record cannot match them by string.

Forecasting: the target is the last 24 values of all seven channels. The input is the window before
them, as the authors' `ForecastingDataset` splits it. The horizon is 24 values, the default of the
authors' `forecast_finetune.py`.

Every event-type task shares one label set, `event_type`. It lists the nine event types in label
order, and each target holds exactly one of them.

## Checks

The connector reads each file twice. The first pass checks every window and retrieval row, and
stops with the file and the key or id at the first bad value. The second pass runs when the dataset
is written. It stops if a file no longer holds what the first pass saw.

## The authors' splits

The authors give each event its own windows and split. So the same measurements, and windows that
overlap in time, appear in more than one split. A model can score well on these by memorizing its
training data. The dataset keeps the authors' splits, so that results compare with the paper.

To split without these overlaps, split the Records by date. The cut dates below are an example and
do not come from the paper. Give each task the split of its Record: its input for event type and
forecasting, its target for retrieval.

```python
from datetime import datetime

from timenet.types import Split

FIRST_CUT, SECOND_CUT = datetime(2022, 6, 10), datetime(2023, 3, 18)  # example cuts


def time_split(record) -> Split | None:
    """Return a Record's split, or None when its window crosses a cut."""
    start = datetime.fromisoformat(record.metadata["window_start"])
    end = datetime.fromisoformat(record.metadata["window_end"])
    if end < FIRST_CUT:
        return Split.TRAIN
    if start >= FIRST_CUT and end < SECOND_CUT:
        return Split.VALIDATION
    if start >= SECOND_CUT:
        return Split.TEST
    return None
```

## Limitations

- The writer rereads each window once per channel, so a write parses the window files about seven
  times. On one laptop, the conversion took 21 s and the write took 110 s. The written dataset is
  166 MiB.
- The authors give precipitation in mm, and do not say how 4-hourly and daily values are made from
  hourly ones. A 4-hourly or daily value is not the total of the hourly values of the same anchor.
  None of about 9,800 compared values with rain matched a total. Most do not match a mean either.
  So the interval that a value covers is not known.
- Many precipitation series repeat one nonzero value for 24 values or more, such as 1.12 mm for
  days. 6,426 of the 21,123 distinct hourly series do this. This looks like filled gaps, but the
  authors do not say. The dataset keeps the values as published.
- Some values are not physically possible, such as relative humidity above 100%. The dataset keeps
  them as published.
- The generated descriptions have errors, such as `wind_v` called the "vertical" component.
- Some copies carry different event types, so no classifier can get every copy right. A copy and
  its original can also be in different splits.
- A retrieval row joins its window by id only. The connector does not compare the row's
  `timeseries` copy with the window.
- No release states a license. The NOAA sources list CC0-1.0 terms, but the descriptions are text
  that the authors generated with ChatGPT. The card declares `NONE`, and we asked the authors if the
  release can be used under CC BY 4.0.
