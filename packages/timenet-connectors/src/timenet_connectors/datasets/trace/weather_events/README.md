# TRACE Weather Events

TRACE pairs weather series from NOAA stations with NOAA reports of severe weather events. It also
gives many series a caption that ChatGPT wrote. The data comes with the paper [TRACE: Grounding
Time Series in Context for Multimodal Embedding and Retrieval](https://arxiv.org/abs/2506.09114)
(NeurIPS 2025). The code for the paper is in [TRACE-Multimodal-TSEncoder][trace-code].

The authors publish the same data in two releases:

| Release | Pin | Content | Used by the connector |
| --- | --- | --- | --- |
| [Google Drive](https://drive.google.com/file/d/1hX4D91QbXa0UQlgf6Jnf-1ii96gfp1aY/view) | sha256 `a5d24d75…be58` | The preprocessed `dataset/` folder that the code loads, with the authors' splits | Yes, as the only source |
| [Hugging Face](https://huggingface.co/datasets/catherpker/TRACE-TimeseriesRAG-Dataset/tree/3e02bda9740792d4eeba63ff398ecbb660c086f3) | revision `3e02bda` | 44,565 captioned station windows and 4,855 event reports, as raw JSONL | No |

The connector downloads and reads only the Google Drive archive, because the authors' code loads
that release. It does not download the Hugging Face release. This README uses the Hugging Face
release only as a reference: the channel units come from its card (see [Records](#records)), and
some counts come from a comparison of the two releases (see
[How the releases relate](#how-the-releases-relate)). Read
[The authors' splits](#the-authors-splits) before you compare results with the paper or with
other models trained on TRACE. The card declares a `NONE` license, because neither release states
a license (see [Data limitations](#data-limitations)).

[trace-code]: https://github.com/Graph-and-Geometric-Learning/TRACE-Multimodal-TSEncoder/tree/1b6524630cb3079754e17f7c88c88d820645096a

## The TimeNet dataset

The connector downloads the archive one time, makes sure that its sha256 is correct, and extracts
it. The build stops if the row count or the event count of a split file is different from the
pinned archive.

A window is one series of seven weather channels from one station. The dataset has one Record for
each window of the paper, 74,337 in all.

### Records

Each Record holds the seven channels of its window as Signals of one station Source. Its
`start_time` is the first timestamp of the window, read as UTC (see [Windows](#windows)). A window
without gaps has a regular time axis at the cadence of its mode (60,514 windows). A window with
gaps keeps the time offset of each value (13,823 windows).

| Channel | Unit | Spec |
| --- | --- | --- |
| `temperature` | °C | float32 |
| `precipitation` | mm | float32 |
| `relative_humidity` | % | float32 |
| `visibility` | km | float32 |
| `wind_u` | m/s, eastward | float32 |
| `wind_v` | m/s, northward | float32 |
| `sky_code` | GHCN-h sky cover code, 0–10 | `enum` codebook |

The units come from the field table of the Hugging Face card. The Drive archive and the code give
no units. The values agree with these units. For example, the highest visibility is 16.09 km,
which is the reporting limit of 10 miles.

The card calls `sky_code` "0–8 oktas", but the data also holds code 9, in 23,964 values. The codes
are GHCN-h sky cover codes, from [NCEI's hourly coverage code table][ncei-sky]. Codes 00–08 count
oktas, which are eighths of the sky that cloud covers. Code 09 means that the sky is obscured, and
code 10 means partial obscuration. Codes 09 and 10 are not amounts. For this reason, the connector
stores the channel as an `enum` with one label for each code.

[ncei-sky]: https://www.aos.wisc.edu/oldsco/stations/NCEI-hrly-doc_for_clds-BNA.txt

Each Record has three annotations: `location`, `window_start`, and `window_end`. The authors'
caption generator received this context with the series (paper Appendix B.4). The caption tasks
use these annotations as inputs. A `pretrain` file gives `location` as a string, such as
`MADERA,CALIFORNIA`. A `forecasting` file gives a list, such as `["County", "State"]`. An
annotation key has one value type, so the connector joins the list as `County, State`.

The metadata of each Record holds these values:

- The original window `id`, as `source_id`.
- The `anchor`, which is `id // 3`.
- The source file, which gives the split of the window in the archive.
- The station, its coordinates, and the `mode` of the window.
- For a `pretrain` window, the published `event_type`. A non-event window has `-100`. A window
  keeps its `event_type` also when it is too short for a classification task.

No task reads the metadata, so these values are not annotations.

### Event-type classification

The dataset has one `ClassificationTask` for each event window that the authors classify. These
are the 14,446 `pretrain` windows that have an event and at least 100 values. The tasks keep the
authors' split: 10,110 train, 2,884 validation, and 1,452 test.

The input is the Record only, with no prompt and no input annotations. The authors' classifier
also sees only the series. Its head reads the output of the time-series encoder before the model
adds text.

The target is the NOAA event type. The `target_schema` of each task is `trace-noaa-event-types`.
This is the id of the `event_type_vocabulary` annotation of the dataset. The annotation lists the
nine types in the label order of the authors:

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

The rationale is the NOAA event report of the window, in the form that the authors'
`generate_er` writes (see [Generated text](#generated-text)). Use the rationale only for
supervision. Do not use it as an input. It names the target, and it describes what happened after
the window ended, such as damage, rainfall totals, and evacuations. People wrote these reports.
They are not reasoning from the observed series.

The task metadata holds the label index and the `event_id` of the report. 4,848 events cover the
14,446 tasks, and most events have all three of their windows. The authors' split often puts the
windows of one event in different splits. To group these windows or to remove duplicates, use the
`event_id`.

Non-event windows have the label −100 in the source, and they have no classification task. The
authors' code removes them from the classification loss and metrics, but paper Appendix B.5 lists
`None` as a category.

### Captioning

The dataset has one `AnswerTask` for each row of `retrieval/*.parquet`. These are the 44,186
captioned windows with at least 100 values, with and without an event. The tasks keep the
authors' retrieval split: 42,180 train and 2,006 test.

The input is the Record, a prompt, and the `location`, `window_start`, and `window_end`
annotations of the Record. The prompt is the captioning prompt of TimeBraid (arXiv:2609.29792,
Figure 17). Every caption names the county and the state of the window, and most captions name
its dates. The series alone does not give this information.

The target is the description of the window, in the form that the authors' `generate_dsp` writes
(see [Generated text](#generated-text)). The description starts with the location and the time
range that the input annotations give. For this reason, caption metrics partly reward a model that
copies them.

TimeBraid trains on "TRACE Captioning", but it does not publish its prompt or its target text for
that task. The prompt here is the one that the TimeBraid paper shows for captioning. Nobody
confirmed it for TRACE. The target is the only caption of a full window that the TRACE code
defines.

TimeBraid reports 43,998 samples and does not describe its filter. If you keep the windows with at
least 120 values, you get exactly that count: 42,002 train and 1,996 test. A limit of 119 or 121
values gives a different count. To get approximately the rows of TimeBraid, keep the caption tasks
whose window has at least 120 values, from the `n_values` of the signals. This is a hypothesis,
not a documented recipe.

A Record can be in different splits for its two tasks, for example validation for classification
and test for captioning. See [The authors' splits](#the-authors-splits).

### Windows by task

Each window is in one of four groups:

| Windows | Count | Classification | Caption |
| --- | --- | --- | --- |
| Event windows of `pretrain` with at least 100 values | 14,446 | yes | yes |
| Other captioned windows with at least 100 values | 29,740 | no | yes |
| Non-event windows without a caption | 29,772 | no | no |
| Captioned windows under 100 values, 119 of them event windows | 379 | no | no |

`forecasting` windows have no event type, because the authors use them only to fine-tune for
forecasting. The authors captioned about half of the non-event windows (see
[How the releases relate](#how-the-releases-relate)).

The 30,151 Records without a task stay in the dataset, so that the dataset holds every window of
the paper. The `source_file` of each Record gives its split in the archive. A later task can use
these Records without a change to them, for example a forecasting task on the `forecasting` split.

The dataset does not hold these values yet:

- The state, county or zone, and begin and end times of each event. The begin time gives the time
  from the window to the event.
- The sentence for each channel as a separate field. Each caption target includes these
  sentences.
- The forecast horizon of the paper, which is H = 7 for weather.

### Dataset annotations

| Annotation | Value |
| --- | --- |
| `event_type_vocabulary` | The nine event types, in label order |
| `no_event_label` | `-100`, the source label for a non-event window, which no task uses |

## Source data

### Windows

Each window comes from one NOAA GHCN-h station. A window has its `id`, the `station_id`, the
coordinates, and the `location` of its station, its `mode`, one `DATE` timestamp for each value,
and the seven channels. `pretrain` windows also have an `event_type`.

An anchor is one station and one end date. Each anchor gives three windows, with the consecutive
ids `3k`, `3k+1`, and `3k+2`:

| `mode` | Cadence | Span | Typical length |
| --- | --- | --- | --- |
| `7day_hourly` | 1 hour | 7 days | 168 |
| `28day_4hourly` | 4 hours | 28 days | 168 |
| `180day_daily` | 1 day | 180 days | 180 |

The timestamps have no time zone. If you read them as UTC, every timestamp is on the grid of its
mode. If you read them in a local time zone with daylight saving time, some days seem to have 23
or 25 hours. 13,823 windows have gaps, which are grid points with no value. These windows are
shorter than the typical length.

### Events

Each event window links to one report in the NOAA Storm Events Database. The report has an
`event_id`, an event type, and a narrative. An event window ends 0–4 days before the event
starts. The window is the history before the event. For this reason, the paper calls its
classification task "multi-class event prediction". The narrative describes what happened after
the window. Do not use it as an input to a task on that window.

### Generated text

ChatGPT wrote a description of each captioned window from the location, the dates, and the series
of the window (paper Appendix B.3–B.4). The description has one sentence for each of the seven
channels, a `DATE` sentence, a `location` sentence, and a `labels` keyword list such as
`Cold, Rainy, Cloudy, Windy`. The generation prompt names ten keywords for this list. The prompt
does not mention the event. The paper reports no human review of the text.

The code in `src/data/load_data.py` makes text from the descriptions and the event reports with
two templates. The connector uses the same templates:

| Text | Built by | Template |
| --- | --- | --- |
| Sample description | `generate_dsp` | `Weather time series location: {location} Time range: {DATE} The weather is {labels}.`, then the seven channel sentences joined by ` \n ` |
| Event report | `generate_er` | `The weather event is {event_type}. {narrative}` |

### Drive archive layout

The archive is 487 MiB, and it extracts to a `dataset/` folder of 1.75 GB. Each folder holds the
splits of one task:

| Folder | Train | Val | Test | Content |
| --- | --- | --- | --- | --- |
| `pretrain/*.json` | 45,339 | 12,953 | 6,484 | Windows keyed by `id`, with `event_type` 0–8, or −100 for no event |
| `forecasting/*.json` | 6,690 | 1,914 | 957 | Non-event windows keyed by `id`, split by whole anchor |
| `retrieval/*.parquet` | 42,180 | — | 2,006 | Captioned windows with at least 100 values, with descriptions and event reports |
| `pretrain/*_data/*.npy` | 44,936 | 12,857 | 6,421 | The `pretrain` windows as arrays, without the short windows |
| `classification/*_data/*.npy` | 10,110 | 2,884 | 1,452 | The event windows of `pretrain` with at least 100 values |

The connector reads the JSON files for the windows, the ids, and the splits. It reads the parquet
files for the captions, the event reports, and the retrieval split. The `.npy` arrays have no ids
and hold the same windows as the JSON files, in the same order. For this reason, the connector
does not read them. The `pretrain` and `forecasting` counts are the same as in Table 8 of the
paper. No released script makes these splits. The authors do not publish the sampling, the
seeds, or the ratios.

### How the releases relate

The ids go from 0 to 75,023. They are the original numbers of the authors, which the authors kept
after they removed rows. The `pretrain` and `forecasting` files on Drive hold exactly the 74,337
windows of the paper. The authors removed 184 events and 45 anchors at distant stations, and they
do not give a reason. The Hugging Face release holds only the 44,565 captioned windows. These are
every event window and a random half of the non-event windows. Paper Appendix B.3 calls this
sampling "for event-label balance". The channel values and the timestamps are the same in both
releases for every window that the two releases share.

## The authors' splits

The code splits the same windows in three different ways:

| Splitting | Unit | Used for |
| --- | --- | --- |
| `pretrain`, and `classification` within it | single window, at random | Stage 1 pretraining and event classification |
| `forecasting` | whole anchor | Forecasting fine-tuning |
| `retrieval` | single window, at random | Stage 2 alignment and retrieval evaluation |

The `pretrain` split assigns each window separately. For this reason, the three windows of an
anchor are often in different splits: 13,983 of 24,687 pretraining anchors are in more than one
split. The three windows of an anchor end on the same date, so they overlap in time. For example,
the three windows of event 1065296 (Debris Flow, Madera, California) are in train, val, and test.
An evaluation on the authors' test split therefore sees series that overlap training series. It
also sees event reports that label training windows.

The `retrieval` split does not follow the `pretrain` split:

| `retrieval` test windows | Split they belong to elsewhere |
| --- | --- |
| 1,232 | `pretrain` train |
| 341 | `pretrain` val |
| 197 | `pretrain` test |
| 236 | `forecasting` |

Stage 2 starts from the Stage 1 encoder, and Stage 1 trained that encoder on most of the retrieval
test windows. `retrieval` also has no validation split. Stage 2 evaluates on test after every
epoch, so test is the only held-out data that it sees, also when it selects a checkpoint. This
test set is the "2,000 bidirectional Text–Timeseries query pairs" of the paper (Table 1).

## Data limitations

- The captions have errors. In a sample of 646 rows, 7.7% of the range claims, such as "between X
  and Y" or "from X to Y", missed the actual minimum or maximum of the series. Each miss was more
  than 1 unit and more than 10% of the range. Some `wind_v` sentences repeat the `wind_u` range.
  Others call `wind_v` the "vertical" component, but it is the northward component. These
  sentences are the targets of the caption tasks, so a model that trains on them also learns
  their errors.
- The `labels` field has words that are not among the ten keywords of the prompt, such as
  `Warm`, `Dry`, and `Cool`. It also has stray braces, such as `{Clear` and `Windy}`.
- 379 captioned windows have fewer than 100 values, and 15 windows have a single value. They are
  Records without a task.
- Some values are not physically possible. The archive keeps these upstream values, and the
  connector stores them as published. 13,748 relative humidity values in 2,079 windows are above
  100%, up to 81,265%. 58 temperatures in 47 windows are above 60 °C, up to 438 °C. 29,103
  visibility values in 724 windows are above the 16.09 km limit, up to 225 km.
- The archive does not say how the authors made the 4-hourly and daily values from the hourly
  readings. For example, it does not say if precipitation is a sum or a mean.
- 10,903 captioned windows repeat the station and the channel values of another window exactly,
  and 10,900 repeat its station and its window dates. Some of these copies have different
  captions.
- The authors' splits separate windows that overlap in time and share an event. See
  [The authors' splits](#the-authors-splits).
- The Drive archive has no fixed revision. Its sha256 detects a change, but it cannot recover the
  pinned version.
- The license of the TRACE release is not known. The original NOAA sources both list CC0-1.0
  terms: the GHCN-hourly weather observations (see "Use Constraints") and the Storm Events
  Database (see "Access & Use"). But the captions are text that the TRACE authors generated with
  ChatGPT. The two releases and the code repository state no license. The CC BY-NC-ND 4.0
  license on arXiv covers the paper, not the data. For this reason, the card declares `NONE`. We
  asked the authors if people can use the release under CC BY 4.0.

The counts in this README come from full passes over both releases at their pins. The caption
range check is the one exception. It comes from the first 646 Hugging Face rows.
