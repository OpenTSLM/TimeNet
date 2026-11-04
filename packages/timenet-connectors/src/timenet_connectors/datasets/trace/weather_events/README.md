# TRACE Weather Events

`trace/weather-events` holds the weather data of TRACE ([arXiv:2506.09114][paper], NeurIPS 2025):
NOAA GHCN-hourly station measurements before NOAA Storm Events, with the authors' event labels,
generated descriptions, forecasting windows, and splits.

The connector reads the [Google Drive archive][drive] that the authors' code loads
([TRACE-Multimodal-TSEncoder at `1b65246`][code]). Drive has no immutable revision, so the 487 MiB
archive is pinned by sha256.

[paper]: https://arxiv.org/abs/2506.09114
[drive]: https://drive.google.com/file/d/1hX4D91QbXa0UQlgf6Jnf-1ii96gfp1aY/view
[code]: https://github.com/Graph-and-Geometric-Learning/TRACE-Multimodal-TSEncoder/tree/1b6524630cb3079754e17f7c88c88d820645096a

## The source data

A **window** is one station's measurements over a time span: seven channels, one timestamp per
value, and one of three modes.

| `mode` | Cadence | Span | Typical length |
| --- | --- | --- | --- |
| `7day_hourly` | 1 hour | 7 days | 168 |
| `28day_4hourly` | 4 hours | 28 days | 168 |
| `180day_daily` | 1 day | 180 days | 180 |

An **event** is one report in the NOAA Storm Events Database: an id, one of nine event types, a
county, and a narrative. For each event, the authors build an anchor at a nearby station: three
windows, one per mode, that end before the event begins. The window ids of an anchor are `3k`,
`3k+1`, and `3k+2`. Windows without a following event have the label `-100`.

The archive holds 74,337 windows, as paper Table 8 counts them. The connector does not check these
counts. The sha256 of the archive fixes its content instead.

| File | Train | Val | Test | Content |
| --- | --- | --- | --- | --- |
| `pretrain/*.json` | 45,339 | 12,953 | 6,484 | Event and non-event windows, with `event_type` |
| `forecasting/*.json` | 6,690 | 1,914 | 957 | Non-event windows |
| `retrieval/*.parquet` | 42,180 | | 2,006 | Generated descriptions and event reports |

The connector does not read the archive's `.npy` copies of the windows, or the `timeseries` column
of the `retrieval` files. The archive has no event state or begin and end times.

**Windows repeat, once per event.** When several events follow the same days at one station, each
event gets its own anchor, window ids, label and split, but the windows hold the same
measurements. Two windows are the same when their station, mode, timestamps and every value are
equal. Copies differ only in what belongs to their event: its id, type, narrative, county,
description, and split. Some measurements were followed by events of different types.

## The TimeNet dataset

### Records

A Record is one distinct measurement window, so the dataset holds 63,366 Records for 74,337
published windows. Each Record has the seven channels as Signals of one Source named `station`, on
one time axis:

| Channel | Unit | Spec |
| --- | --- | --- |
| `temperature` | °C | float32 |
| `precipitation` | mm | float32 |
| `relative_humidity` | % | float32 |
| `visibility` | km | float32 |
| `wind_u` | m/s, eastward | float32 |
| `wind_v` | m/s, northward | float32 |
| `sky_code` | GHCN-h sky cover code, 0–10 | `enum` codebook |

The units come from the field table of the Hugging Face card. The `sky_code` codes follow
[NCEI's hourly coverage code table][ncei-sky]: codes 00–08 count oktas, 09 means that the sky is
obscured, and 10 means partial obscuration. Codes 09 and 10 are not amounts, so the channel is an
`enum` with one label per code.

[ncei-sky]: https://www.aos.wisc.edu/oldsco/stations/NCEI-hrly-doc_for_clds-BNA.txt

The Record metadata is one plain JSON object. Its `station` is the station id. A station is not
a subject, so the Records have no `subject_ids` annotation. Beside it are the station's
`coordinates`, the `sampling`, and `windows`: every published window with these measurements, in
id order. Each has its `window_id`, `anchor`, `source_file`, `county`, and its published
`event_type`, or `null` for a `forecasting` window. The county belongs to the window, not the station:
an event window names the county of its event. The connector splits the source's `location` into a `name` and a `state`, as
published. Metadata is never model input, and neither are the authors' window ids.

The Record annotations are `window_start` and `window_end`, the first and last timestamps as
published.

### Time

The timestamps have no time zone. They are the station's local clock time, with daylight saving
time. A Record has no absolute start time: its time axis counts local-clock time from its first
timestamp. To place a timestamp in UTC, find the station's time zone from the `coordinates`
metadata, for example with `timezonefinder`, and pin the tzdata version. Across a spring-forward
or fall-back night, a time offset differs from the elapsed time by one hour until you convert it.

A window without gaps has a regular time axis at its mode's cadence. A window with gaps keeps the
time offset of each value.

### Tasks

Every published sample is a task of its own, on the Record that holds its measurements, under the
authors' split. Its metadata holds its window id, anchor, and source file.

| Task | One per | Model input | Target | Split | Count |
| --- | --- | --- | --- | --- | --- |
| `ClassificationTask` | `pretrain` event window with at least 100 values | The series | The event type | `pretrain` file | 14,446 |
| `TSCorrespondenceTask` | Row of the `retrieval` files | The description, as the prompt | The window's Record | `retrieval` file | 44,186 |
| `ForecastingTask` | `forecasting` window with at least 100 values | The series before its last 24 values | Those 24 values | `forecasting` file | 9,474 |

A window without an event-type or forecasting task keeps its Record. Many such windows still
have a retrieval task.

**Event type.** The nine labels follow `EVENT_MAP` in the authors' `src/common.py`. The rationale
is the event's report as the authors' `generate_er` writes it: "The weather event is {type}.
{narrative}". The report describes what happened after the window, so it is supervision only. The
task metadata adds the event's `event_id`. The authors' classifier sees only the series, so the
task names no annotations.

**Retrieval.** This is the paper's Stage 2, which aligns each series with its description. The
prompt is the description as the authors' `generate_dsp` composes it. The target is the window's
Record, and the task has no input Record: the text must find the series. The task's
`channel_description` annotations hold the seven channel sentences, each with its channel in
`metadata["channel"]`. A row with an event adds its `event_id` to the task metadata.

Test tasks name the Records of the test windows as their candidates, the pool that the authors'
evaluation ranks. Copies of one measurement share a Record, so a copy counts as a right answer,
where the authors count it as a miss. Train tasks name no candidates.

**Evaluate retrieval with the candidates' series only.** A Record's `window_start` and
`window_end` annotations, and its windows' `county` metadata, repeat the dates and county that a
description opens with. A model shown them can match a description by string. The authors' encoder
sees only the series.

**Forecasting.** The target is the last 24 values of all seven channels, and the input is the
window before them, as the authors' `ForecastingDataset` splits each window. The horizon counts
values, not time. It is 24, the default of `forecast_finetune.py` that the authors' README runs;
the paper reports both H = 7 and H = 24.

### Dataset annotations

| Annotation | Value |
| --- | --- |
| `event_type_vocabulary` | The nine event types, in label order |
| `no_event_label` | `-100`, the source label for a non-event window, which no task uses |

## What the connector checks

The first pass checks every row and keeps only small facts and digests. The second pass runs when
the dataset is written, and it reads the files again.

The first pass stops with the file and the row or key if:

- a window lacks a field, or has a blank text, a value that is not a finite number, a sky cover
  code outside 0 to 10, a timestamp that is not a real date in `YYYY-MM-DDTHH:MM:SS` form, or
  timestamps that do not increase on the cadence of its `mode`;
- a `mode` does not match the position of the window id in its anchor triplet;
- a `pretrain` window has no event label, a `forecasting` window has one, or a label is not one of
  the numbers of `EVENT_MAP`;
- a window id is not a plain decimal number, appears twice, or belongs to a copy of a measurement
  that stands at other coordinates;
- a retrieval row has a blank description or report text, a bad event type, or an id that is
  repeated or names no window;
- a classified window has no retrieval row, a retrieval row has a report for a window that is not
  classified, or a report names another event type than its window.

The second pass stops if a window file no longer holds the series that the first pass saw. The
check covers the length and a SHA-256 digest of the station, mode, timestamps and every value.
It stops if a retrieval file no longer holds the same rows, in the same order.

## The authors' splits

The authors assign windows to splits one at a time, and each event has its own windows. So the
same measurements, and windows that overlap in time, appear in more than one split, for all three
tasks. A model can score well on these by memorizing its training data. The dataset keeps the
authors' splits, so that results compare with the paper.

For an evaluation without these overlaps, split the Records by date: train windows end before a
first cut, test windows start after a second, and a window that crosses a cut is left out. Give
each task the split of its Record: its input for event type and forecasting, its target for
retrieval.

```python
from datetime import datetime

FIRST_CUT, SECOND_CUT = datetime(2022, 6, 10), datetime(2023, 3, 18)


def time_split(record):
    """Return a Record's split, or None when its window crosses a cut."""
    dates = {a.key: a.value.read() for a in record.annotations}
    start = datetime.fromisoformat(dates["window_start"])
    end = datetime.fromisoformat(dates["window_end"])
    if end < FIRST_CUT:
        return "train"
    if start >= FIRST_CUT and end < SECOND_CUT:
        return "validation"
    if start >= SECOND_CUT:
        return "test"
    return None
```

## Not verified on the real archive

The tests use small fixtures that follow the layout of the archive. This version has not been run
on the 487 MiB archive. So these facts are not re-checked here: the counts of Records and tasks in
`dataset.yaml` and in the task table above, the time the conversion takes, and the memory it uses.
A test run of the first pass on a few fixture windows measured about 1.2 KB for each window in the
index and about 7.4 KB for each Record. For 74,337 windows and 63,366 Records, that is roughly
0.55 GB. This estimate is not measured on the real archive. The writer rereads a window once for
each of its seven channels, so a write parses each window file about seven times.

## Data limitations

- In 4-hourly and daily windows, each value is the mean of the hourly values. So precipitation
  there is a mean hourly amount, although its unit reads as mm.
- Some values are not physically possible, such as relative humidity above 100% and temperatures
  above 60 °C. The dataset keeps them as published.
- The timestamps are local clock time, and the source gives no time zone.
- The generated descriptions have errors, such as value ranges that miss the series and `wind_v`
  called the "vertical" component. A model that aligns with the descriptions learns these too.
- Some measurements carry more than one event type, so no classifier can get every copy right.
- The license of the TRACE release is not known. The NOAA sources list CC0-1.0 terms, but the
  descriptions are text that the authors generated with ChatGPT, and no release states a license.
  The card declares `NONE`, and we asked the authors if the release can be used under CC BY 4.0.
- The Drive archive has no fixed revision. Its sha256 detects a change, but cannot recover the
  pinned version.
