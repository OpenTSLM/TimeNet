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
county, and a narrative. The authors make one sample per event. For each event, they build an
anchor at a nearby station: three windows, one per mode, that end before the event begins. The
window ids of an anchor are `3k`, `3k+1`, and `3k+2`. Windows without a following event have the
label `-100`.

The archive holds 74,337 windows, as paper Table 8 counts them:

| File | Train | Val | Test | Content |
| --- | --- | --- | --- | --- |
| `pretrain/*.json` | 45,339 | 12,953 | 6,484 | Event and non-event windows, with `event_type` |
| `forecasting/*.json` | 6,690 | 1,914 | 957 | Non-event windows |
| `retrieval/*.parquet` | 42,180 | | 2,006 | Generated descriptions and event reports of 44,186 windows |

The archive also holds `.npy` copies of the windows, which the connector doesn't read. The
`pretrain/*_data/ids.txt` files name the window of each array. The `retrieval` files hold another
copy of each series in a `timeseries` column, which the connector skips. The Hugging Face release
adds event fields that the archive lacks, such as each event's state and its begin and end times,
and the dataset doesn't hold them.

### Windows repeat, once per event

When several events follow the same days at one station, each event gets its own anchor, window
ids, label, and split, but the windows hold the same measurements. Of the 74,337 windows, 63,366
are distinct: two windows are the same when their station, mode, timestamps, and every value of
every channel are equal. Almost all repeats are event windows. The 14,446 classified windows hold
only 3,696 distinct measurements, and one hourly week at station `USW00013739` precedes 52
Thunderstorm Wind events. Copies differ only in what belongs to their event: its id, type,
narrative, county, the generated description, and the split.

424 distinct measurement windows were followed by events of different types. For example,
windows 8178 and 5166 hold the same week at station `USW00094949`. Window 8178 is in `train`,
labelled Thunderstorm Wind, in Fillmore County. Window 5166 is in `val`, labelled Hail, in Furnas
County.

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

The units come from the field table of the Hugging Face card, and they agree with GHCNh. The
`sky_code` codes follow [NCEI's hourly coverage code table][ncei-sky]: codes 00–08 count oktas, 09
means that the sky is obscured, and 10 means partial obscuration. Codes 09 and 10 are not amounts,
so the channel is an `enum` with one label per code.

[ncei-sky]: https://www.aos.wisc.edu/oldsco/stations/NCEI-hrly-doc_for_clds-BNA.txt

The Record metadata holds the station id, its coordinates, the mode, and every published window
with these measurements: its window id, anchor, source file, `location`, and, for a `pretrain`
window, its published `event_type`. An event window's `location` is the county of its event, so
copies of one measurement can name different counties. A `pretrain` file gives `location` as a
string, such as `MADERA,CALIFORNIA`. A `forecasting` file gives a list, which the connector joins
as `County, State`. Metadata is provenance and is never model input.

The Record annotations are `window_start` and `window_end`. An annotation is model input only for a
task that names it.

The Record ids are digests of the measurements, such as `trace-3f9a…`. The authors' window ids are
sorted by event type: their event windows form exactly nine runs, one per class. So no id in the
dataset contains them, and they stay in metadata.

### Time

The timestamps have no time zone. They are the station's local clock time, with daylight saving
time: GHCNh values match only after adding the station's UTC offset on that date, and the daily
temperature peak falls near 15:00 on the series' own clock at every longitude. The authors' code
never reads `DATE`, and their models see values by position.

A Record has no absolute start time. Its time axis counts local-clock time from its first
timestamp, and `window_start` and `window_end` keep the first and last timestamps as published.
Each published timestamp is `window_start` plus the value's time offset. To place it in UTC, find
the station's time zone from the `latitude` and `longitude` metadata, for example with
`timezonefinder`, and pin the tzdata version.

Daylight saving time shows in the timestamps. Of the 437 distinct hourly windows that run past a
spring-forward change, 416 lack only the 02:00 hour that the clock skipped: the dataset stores it
as a gap, but no reading is missing, and converting to UTC removes it. The 15 at stations in
Arizona and Hawaii, which keep standard time, have that hour, and 6 have a longer gap. Of the 140
that run past a fall-back change, 136 hold one 01:00 value for two real hours, which no conversion
can split. 2 have a gap there, and 2 are in Arizona. Until a window is converted, a time offset
across either night differs from the elapsed time by one hour.

A window without gaps has a regular time axis at its mode's cadence. 13,823 published windows have
gaps, and keep the time offset of each value.

### Tasks

Every published sample is a task of its own, on the Record that holds its measurements, under the
authors' split. Its metadata holds its window id, anchor, and source file.

| Task | One per | Model input | Target | Split | Count |
| --- | --- | --- | --- | --- | --- |
| `ClassificationTask` | `pretrain` event window with at least 100 values | The series | The event type | `pretrain` file | 14,446 |
| `TSCorrespondenceTask` | Row of the `retrieval` files | The description, as the prompt | The window's Record | `retrieval` file | 44,186 |
| `ForecastingTask` | `forecasting` window with at least 100 values | The series before its last 24 values | Those 24 values | `forecasting` file | 9,474 |

Windows without a task keep their Records: 50,211 `pretrain` windows without an event, and 119
event windows and 87 `forecasting` windows with fewer than 100 values.

**Event type.** The nine labels follow `EVENT_MAP` in the authors' `src/common.py`: Lightning,
Debris Flow, Flash Flood, Heavy Rain, Tornado, Funnel Cloud, Hail, Flood, and Thunderstorm Wind.
The rationale is the event's report as the authors' `generate_er` writes it: "The weather event
is {type}. {narrative}". The report describes what happened after the window, so it is supervision
only. The task metadata adds the event's `event_id`. Splits: 10,110 / 2,884 / 1,452, equal to the
authors' `classification/*_data` arrays. The authors' classifier sees only the series, so the task
names no annotations. The paper reports no class balance. The authors' splits hold:

| Event type | Train | Validation | Test |
| --- | --- | --- | --- |
| Lightning | 136 | 39 | 20 |
| Debris Flow | 134 | 38 | 20 |
| Flash Flood | 1,433 | 407 | 206 |
| Heavy Rain | 527 | 150 | 76 |
| Tornado | 434 | 124 | 63 |
| Funnel Cloud | 140 | 40 | 21 |
| Hail | 1,102 | 315 | 158 |
| Flood | 1,433 | 409 | 207 |
| Thunderstorm Wind | 4,771 | 1,362 | 681 |

**Retrieval.** This is the paper's Stage 2, which aligns each series with its description
(`context_align.py`). The prompt is the description as the authors' `generate_dsp` composes it:
`Weather time series location: {location} Time range: {DATE} The weather is {labels}.`, then the
seven channel sentences joined by ` \n `. The target is the window's Record, and the task has no
input Record: the text must find the series. ChatGPT wrote the descriptions from each window's
series, location, and dates (paper Appendix B.4), and every description opens by restating the
location and dates.

The task's annotations hold the seven channel sentences, `key="channel_description"`, each with
its channel in `metadata["channel"]`: the texts that the authors' `generate_channel_description`
splits out and aligns with each channel. They sit on the task, not on the Record, because copies of
one measurement each have their own description. A row with an event adds the `event_id` to the
task metadata, which joins it to the event-type task whose rationale is the row's event text. A
row without one has no event text: the authors' code uses "No severe weather event." in its place.

Test tasks name the Records of the 2,006 test windows as their candidates, the pool the authors'
evaluation ranks in both directions. Copies of one measurement share a Record, so the pool holds
1,886 Records, and a copy counts as the right answer, where the authors' evaluation counts it as a
miss. Train tasks name no candidates: the authors draw training negatives from each batch. Splits:
42,180 / 2,006.

The authors also rank the descriptions for each series. TimeNet has no task type whose candidates
are texts, so that direction isn't a task: rank the test tasks' prompts for each target instead.

**Forecasting.** The target is the last 24 values of all seven channels, and the input is the
window before them, as the authors' `ForecastingDataset` splits each window into `ts[:, :-H]` and
`ts[:, -H:]`. The horizon counts values, not time, so across a gap it spans more time. Splits:
6,627 / 1,900 / 947.

The horizon follows the authors' code as their README runs it. `forecast_finetune.py` declares
`parser.add_argument("--forecast_horizon", type=int, default=24)` and then sets
`args.forecast_horizon = args_cmd.forecast_horizon`, which replaces `forecast_horizon: 7` from
`configs/finetune.yaml`. The README's two forecasting commands don't pass the argument. The paper
reports both horizons: Table 8 lists the weather task as "Forecasting (H=7)", Table 5 reports
H = 7 and H = 24, and Table 3 evaluates the "next 24 steps".

### Dataset annotations

| Annotation | Value |
| --- | --- |
| `event_type_vocabulary` | The nine event types, in label order |
| `no_event_label` | `-100`, the source label for a non-event window, which no task uses |

## The authors' splits

The authors assign windows to splits one at a time, and each event has its own windows (see
[Windows repeat, once per event](#windows-repeat-once-per-event)). So the same measurements, and
windows that overlap in time, appear in more than one split:

- **Event type.** Of the 3,696 Records with event-type tasks, 1,589 have tasks in more than one
  split. 1,289 of the 1,452 test inputs, and 2,487 of the 2,884 validation inputs, equal an input
  of an earlier split, mostly with the same label.
- **Retrieval.** The `retrieval` split doesn't follow the `pretrain` split. 596 of the 2,006 test
  tasks target a Record that a train task also targets, though no test description equals a train
  one.
- **Forecasting.** No test input equals a train input, but windows overlap in time: in 211 of the
  947 test tasks, every target value appears at the same timestamp in a training window of the
  same station and mode, and in 173 more, some do.
- **Across files.** The three windows of an anchor end on the same date, and they are often in
  different splits. 8,046 anchors have windows of one station that overlap in time in both
  `pretrain` and `forecasting`, although the paper calls the forecasting set disjoint.

A model can score well on these by memorizing its training data. The dataset keeps the authors'
splits, so that results compare with the paper.

### A split by time

For an evaluation without these overlaps, split the Records by date. Train windows end before a
first cut, test windows start after a second, and a window that crosses a cut is left out. Copies
of a measurement share their dates, so no window repeats or overlaps across splits, and every
station can appear in each split.

```python
from datetime import datetime

# Near the 70% and 85% quantiles of the windows' last timestamps.
FIRST_CUT, SECOND_CUT = datetime(2022, 6, 10), datetime(2023, 3, 18)


def time_split(record):
    """Return a Record's split, or None when its window crosses a cut."""
    dates = {a.key: a.value for a in record.annotations}
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

Give each task the split of its Record: its input for event type and forecasting, its target for
retrieval. With these cuts:

| | Train | Validation | Test | Left out |
| --- | --- | --- | --- | --- |
| Records | 44,298 | 7,070 | 6,845 | 5,153 |
| Event type | 9,673 | 1,786 | 1,806 | 1,181 |
| Retrieval | 30,456 | 5,068 | 5,023 | 3,639 |
| Forecasting | 6,606 | 1,077 | 1,008 | 783 |

All nine event types appear in the test split. A retrieval task's candidates stay the authors'
test windows, so rank a time-split test task against the time split's test Records instead. The
cuts compare local clock times, so they fall a few hours apart across time zones.

## Data limitations

- In 4-hourly and daily windows, each value is the mean of the hourly values, not their sum. So
  precipitation there is a mean hourly amount, although its unit reads as mm.
- Some values are not physically possible, and the dataset keeps them as published. 13,748
  relative humidity values in 2,079 windows are above 100%, up to 81,265%. 58 temperatures in 47
  windows are above 60 °C, up to 438 °C. One reading in Chesapeake, Virginia, is −61 °C, and the
  highest hourly precipitation is 463.8 mm. Visibility exceeds the 16.09 km reporting limit in
  29,108 values in 729 windows, up to 225.31 km.
- The timestamps are local clock time with daylight saving time, as published. Across a
  spring-forward or fall-back night, a time offset differs from the elapsed time by one hour (see
  [Time](#time)). Converting to UTC needs each station's time zone, which the source doesn't give.
- The descriptions have errors. About 6% of their 72,526 "from X to Y" range claims miss the
  series, some date sentences contradict their window, and 666 call `wind_v` the "vertical"
  component. The `labels` keywords follow no stated rule. A model that aligns with the descriptions
  learns these too.
- 424 distinct measurement windows carry more than one event type, so no classifier can get every
  copy right.
- The license of the TRACE release is not known. The NOAA sources list CC0-1.0 terms, but the
  descriptions are text that the authors generated with ChatGPT, and no release states a license.
  The card declares `NONE`, and we asked the authors if the release can be used under CC BY 4.0.
- The Drive archive has no fixed revision. Its sha256 detects a change, but cannot recover the
  pinned version.
