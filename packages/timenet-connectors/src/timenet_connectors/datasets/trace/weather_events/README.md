# TRACE Weather Events

TRACE pairs multivariate weather station series with NOAA severe weather event reports and
LLM-generated captions. It accompanies [TRACE: Grounding Time Series in Context for Multimodal
Embedding and Retrieval](https://arxiv.org/abs/2506.09114) (NeurIPS 2025). The paper's code lives
in [TRACE-Multimodal-TSEncoder][trace-code].

The authors publish two releases of the same data:

| Release | Pin | Content |
| --- | --- | --- |
| [Hugging Face](https://huggingface.co/datasets/catherpker/TRACE-TimeseriesRAG-Dataset/tree/3e02bda9740792d4eeba63ff398ecbb660c086f3) | revision `3e02bda` | 44,565 captioned station windows and 4,855 event reports, as raw JSONL |
| [Google Drive](https://drive.google.com/file/d/1hX4D91QbXa0UQlgf6Jnf-1ii96gfp1aY/view) | sha256 `a5d24d75…be58` | The preprocessed `dataset/` folder that the code loads, with the authors' splits |

[The TimeNet dataset](#the-timenet-dataset) describes what the connector builds, and
[Source data](#source-data) the files it reads. Read [The authors' splits](#the-authors-splits)
before comparing results with the paper or with other models trained on TRACE. The card declares
a `NONE` license, since neither release states a license of its own (see
[Data limitations](#data-limitations)).

[trace-code]: https://github.com/Graph-and-Geometric-Learning/TRACE-Multimodal-TSEncoder/tree/1b6524630cb3079754e17f7c88c88d820645096a

## The TimeNet dataset

The connector reads the Drive archive, the release the authors' code loads. It downloads the
archive once, checks its sha256, and extracts it. The build fails if a split file's row count or
event count differs from the pinned archive.

### Records

The dataset has one Record per window, 74,337 in all: every window of the paper.

- **Signals.** A window's seven channels are Signals of one station Source. `start_time` is the
  window's first timestamp, read as UTC (see [Windows](#windows)). A window without gaps has a
  regular axis at its mode's cadence (60,514 windows). A window with gaps keeps its explicit
  offsets (13,823 windows).
- **Annotations.** Each Record carries its window's `location`, `window_start`, and
  `window_end`. They are the context the authors' caption generator received with the series
  (paper Appendix B.4), and the caption tasks take them as input annotations. `location` is the
  published string, such as `MADERA,CALIFORNIA`; a forecasting file's `["County", "State"]` list
  is joined as `County, State`, because an annotation key has one value type.
- **Metadata.** Each Record's metadata holds its original `id` (`source_id`), its `anchor`
  (`id // 3`), its source file, and its station, coordinates, and mode. No task reads them, so
  they are not annotations.

| Channel | Unit | Spec |
| --- | --- | --- |
| `temperature` | °C | float32 |
| `precipitation` | mm | float32 |
| `relative_humidity` | % | float32 |
| `visibility` | km | float32 |
| `wind_u` | m/s, eastward | float32 |
| `wind_v` | m/s, northward | float32 |
| `sky_code` | GHCN-h sky cover code, 0–10 | `enum` codebook |

The units come from the field table in the Hugging Face card. The Drive archive and the code
state none. The ranges fit them: visibility tops out at 16.09 km, the 10-mile reporting cap.

The card calls `sky_code` "0–8 oktas", but the data also holds code 9, in 23,964 values. These are
GHCN-h sky cover codes, as listed in [NCEI's hourly coverage code table][ncei-sky]. Codes 00–08
count oktas, 09 means the sky is obscured, and 10 means partial obscuration. Codes 09 and 10 are
not amounts, so the connector stores the channel as an `enum` with one label per code.

[ncei-sky]: https://www.aos.wisc.edu/oldsco/stations/NCEI-hrly-doc_for_clds-BNA.txt

### Event-type classification

One `ClassificationTask` per event window that the authors classify: the 14,446 `pretrain`
windows with an event and at least 100 values. The task keeps the authors' split: 10,110 train,
2,884 validation, and 1,452 test.

- **Input.** The Record only, with no prompt and no input annotations. The authors' classifier
  sees only the series: its head reads the time-series encoder output before any text is fused.
- **Target.** The NOAA event type. Each task's `target_schema` is `trace-noaa-event-types`, the id
  of the dataset's `event_type_vocabulary` annotation, which lists the nine types in the
  authors' label order:

  | Label | Event type |
  | --- | --- |
  | 0 | Lightning |
  | 1 | Debris Flow |
  | 2 | Flash Flood |
  | 3 | Heavy Rain |
  | 4 | Tornado |
  | 5 | Funnel Cloud |
  | 6 | Hail |
  | 7 | Flood |
  | 8 | Thunderstorm Wind |

- **Rationale.** The window's NOAA event report, as the authors' `generate_er` writes it (see
  [Generated text](#generated-text)).
- **Metadata.** The label index and the report's `event_id`. 4,848 events cover the 14,446 tasks,
  most of them with all three windows. The `event_id` ties together an event's windows, which
  the authors' split often separates, so use it to group or deduplicate.

The rationale is supervision only, never an input. It names the target, and its narrative
describes what happened after the window ended, such as damage, rainfall totals, and
evacuations. It is an event report written by people, not reasoning from the observed series.

Non-event windows (label −100 in the source) have no classification task. The authors' code
masks them out of the classification loss and metrics, although paper Appendix B.5 lists `None`
as a category.

### Captioning

One `AnswerTask` per row of `retrieval/*.parquet`: the 44,186 captioned windows with at least 100
values, event and non-event alike. The task keeps the authors' retrieval split: 42,180 train and
2,006 test.

- **Input.** The Record, TimeBraid's captioning prompt (arXiv:2609.29792, Figure 17), and the
  Record's `location`, `window_start`, and `window_end` annotations. Every caption names the
  window's county and state, and most name its dates, which the series alone does not carry.
- **Target.** The window's description, as the authors' `generate_dsp` writes it (see
  [Generated text](#generated-text)). It opens with the same location and time range that the
  input annotations give, so caption metrics partly reward copying them.

TimeBraid trains on "TRACE Captioning" but publishes neither its prompt nor its target text for
it. The prompt here is the one TimeBraid's paper shows for captioning, not one confirmed for
TRACE. The target is the only full-window caption the TRACE code defines.

TimeBraid reports 43,998 samples without describing its filter. Keeping windows with at least
120 values reproduces that count exactly (42,002 train and 1,996 test here), while 119 and 121
miss. To approximate TimeBraid's rows, keep the caption tasks whose window has at least 120
values, through the signals' `n_values`. This is a hypothesis, not a documented recipe.

A Record can sit in different splits for its two tasks, such as validation for classification
and test for captioning. See [The authors' splits](#the-authors-splits).

### Windows by task

Every window falls into one of four groups:

| Windows | Count | Classification | Caption |
| --- | --- | --- | --- |
| Event windows of `pretrain` with at least 100 values | 14,446 | yes | yes |
| Other captioned windows with at least 100 values | 29,740 | no | yes |
| Non-event windows without a caption | 29,772 | no | no |
| Captioned windows under 100 values, 119 of them event windows | 379 | no | no |

`forecasting` windows carry no event type, because the authors use them only for forecasting
fine-tuning. The authors captioned about half the non-event windows (see
[How the releases relate](#how-the-releases-relate)).

The connector keeps the 30,151 Records without a task so that the dataset holds every window of
the paper, with the authors' split membership in each Record's `source_file`. A later task, such
as forecasting on the `forecasting` split, can use them without changing the Records. Source
fields that no task uses yet: each event's state, county or zone, and begin and end times (which
would give the time to the event), the per-channel descriptions, and the paper's forecast horizon
(H = 7 for weather).

### Dataset annotations

| Annotation | Value |
| --- | --- |
| `event_type_vocabulary` | The nine event types, in label order |
| `no_event_label` | `-100`, the source's label for a non-event window, which no task uses |

## Source data

### Windows

Each window comes from one NOAA GHCN-h station, with the window's `id`, the station's
`station_id`, coordinates, and `location`, the window's `mode`, a `DATE` timestamp per value, and
the seven channels. `pretrain` windows also carry an `event_type`.

Every anchor, meaning a station and an end date, yields three windows with consecutive ids
`3k, 3k+1, 3k+2`:

| `mode` | Cadence | Span | Typical length |
| --- | --- | --- | --- |
| `7day_hourly` | 1 hour | 7 days | 168 |
| `28day_4hourly` | 4 hours | 28 days | 168 |
| `180day_daily` | 1 day | 180 days | 180 |

The timestamps have no timezone. Read as UTC, every timestamp falls on its mode's grid. Read as a
local zone that observes daylight saving time, some days appear to last 23 or 25 hours. 13,823
windows have gaps, meaning missing grid points, and are shorter than the typical length.

### Events

Each event window links to one NOAA Storm Events report, with an `event_id`, an event type, and a
narrative. Event-linked windows end 0–4 days **before** the event begins. They are pre-event
history, which is why the paper calls its classification task "multi-class event prediction". An
event narrative describes what happened after its window, so it must never be an input to a task
on that window.

### Generated text

ChatGPT wrote a description of each captioned window, from the window's location, its dates, and
its series (paper Appendix B.3–B.4). The description has a sentence for each of the seven
channels, a `DATE` sentence, a `location` sentence, and a `labels` keyword list such as
`Cold, Rainy, Cloudy, Windy`, drawn from ten keywords the generation prompt names. The prompt
does not mention the event, and the paper reports no human validation.

The code turns descriptions and event reports into text with two templates in
`src/data/load_data.py`, which the connector reproduces:

| Text | Built by | Template |
| --- | --- | --- |
| Sample description | `generate_dsp` | `Weather time series location: {location} Time range: {DATE} The weather is {labels}.`, then the seven channel sentences joined by ` \n ` |
| Event report | `generate_er` | `The weather event is {event_type}. {narrative}` |

### Drive archive layout

The archive unpacks to a `dataset/` folder of 487 MiB zipped and 1.75 GB unpacked. Each folder is
one task's split:

| Folder | Train | Val | Test | Content |
| --- | --- | --- | --- | --- |
| `pretrain/*.json` | 45,339 | 12,953 | 6,484 | Windows keyed by `id`, with `event_type` 0–8, or −100 for no event |
| `forecasting/*.json` | 6,690 | 1,914 | 957 | Non-event windows keyed by `id`, split by whole anchor |
| `retrieval/*.parquet` | 42,180 | — | 2,006 | Captioned windows with at least 100 values, plus descriptions and event reports |
| `pretrain/*_data/*.npy` | 44,936 | 12,857 | 6,421 | The `pretrain` windows as arrays, after dropping short ones |
| `classification/*_data/*.npy` | 10,110 | 2,884 | 1,452 | The event windows of `pretrain`, with at least 100 values |

The connector reads the JSON files for windows, ids, and splits, and the parquet files for
captions, event reports, and the retrieval split. The `.npy` arrays carry no ids and follow the
JSON files in order, so it does not read them. The `pretrain` and `forecasting` counts match
Table 8 of the paper exactly. No released script builds these splits, and their sampling, seeds,
and ratios are not published.

### How the releases relate

The ids run from 0 to 75,023, the authors' original numbering kept after rows were removed. The
Drive `pretrain` and `forecasting` files hold exactly the paper's 74,337 windows; 184 events and
45 distant-station anchors were dropped without explanation. The Hugging Face release holds only
the 44,565 captioned windows: every event window and a random half of the non-event windows,
which paper Appendix B.3 describes as sampling "for event-label balance". Its channel values and
timestamps match Drive for every shared window.

## The authors' splits

The code uses three independent splittings of the same windows:

| Splitting | Unit | Used for |
| --- | --- | --- |
| `pretrain`, and `classification` within it | single window, at random | Stage 1 pretraining and event classification |
| `forecasting` | whole anchor | Forecasting fine-tuning |
| `retrieval` | single window, at random | Stage 2 alignment and retrieval evaluation |

The `pretrain` split assigns windows individually, so an anchor's three windows often land in
different splits: 13,983 of 24,687 pretraining anchors span more than one. The three windows of
an anchor end on the same date and overlap in time. The three windows of event 1065296 (Debris
Flow, Madera, California) fall in train, val, and test respectively. An evaluation on the
authors' test split therefore sees series that overlap training series, and event reports that
also label training windows.

The `retrieval` split ignores the `pretrain` split:

| `retrieval` test windows | Split they belong to elsewhere |
| --- | --- |
| 1,232 | `pretrain` train |
| 341 | `pretrain` val |
| 197 | `pretrain` test |
| 236 | `forecasting` |

Stage 2 starts from the Stage 1 encoder, so most retrieval test windows are windows that encoder
was trained on. `retrieval` also has no validation split: Stage 2 evaluates on test after every
epoch, so test is the only held-out signal it sees, including for choosing a checkpoint. Its
test set is the paper's "2,000 bidirectional Text–Timeseries query pairs" (Table 1).

## Data limitations

- **Generated captions.** In a 646-row sample, 7.7% of the "between X and Y" or "from X to Y"
  range claims missed the series' actual minimum or maximum by more than 1 unit and more than
  10% of the range. Some `wind_v` sentences repeat the `wind_u` range, or call `wind_v` the
  "vertical" component when it is northward. These sentences are the caption tasks' targets, so
  a model trained on them also learns their errors.
- **Label keywords.** Besides the ten keywords the prompt asks for, the `labels` field contains
  other words (`Warm`, `Dry`, `Cool`) and stray braces (`{Clear`, `Windy}`).
- **Short windows.** 379 captioned windows have fewer than 100 values, and 15 have a single
  value. They are Records without a task.
- **Implausible values.** The archive keeps upstream values that fail basic range checks, and the
  connector stores them as published. 13,748 relative humidity values in 2,079 windows exceed
  100%, up to 81,265%. 58 temperatures in 47 windows exceed 60 °C, up to 438 °C. 29,103
  visibility values in 724 windows exceed the 16.09 km cap, up to 225 km.
- **Aggregation.** The archive does not say how 4-hourly and daily values were aggregated from
  hourly readings, for example whether precipitation is a sum or a mean.
- **Duplicates.** 10,903 captioned windows repeat another window's station and channel values
  exactly, and 10,900 repeat its station and window dates. Some of those copies carry different
  captions.
- **Split leakage.** The authors' splits separate windows that overlap in time and share an
  event. See [The authors' splits](#the-authors-splits).
- **Availability.** The Drive archive has no immutable revision. Its sha256 detects a change but
  cannot recover the pinned version.
- **License.** The original NOAA sources, GHCN-hourly weather observations (see "Use
  Constraints") and the Storm Events Database (see "Access & Use"), both list CC0-1.0 terms, but
  the captions are the TRACE authors' ChatGPT-generated text, and we could not find an explicit
  license for the TRACE release itself: neither release nor the code repository states one, and
  CC BY-NC-ND 4.0 on arXiv covers the paper, not the data. The card therefore declares
  `NONE`. We have contacted the authors to clarify whether the release can be used under
  CC BY 4.0.

The counts in this README come from full passes over both releases at their pins. The one
exception is the caption range check, which comes from the first 646 Hugging Face rows.
