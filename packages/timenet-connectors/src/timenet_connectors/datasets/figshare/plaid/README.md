# PLAID

`figshare/plaid` holds PLAID, the Plug-Load Appliance Identification Dataset
([Medico et al., Sci Data 2020][paper]): current and voltage of household appliances, sampled at
30 kHz, for identifying appliances from their electrical signature.

The connector reads version 2 of the [figshare article "PLAID 2018"][article], the release the
paper describes, each file pinned by its SHA-256. The data license is CC BY 4.0. The authors' code
is [jingkungao/PLAID][code] (GPL-3.0): a parsing notebook and the appliance-identification notebook
of their GlobalSIP 2015 paper. The connector reimplements what it needs and copies no code.

[paper]: https://doi.org/10.1038/s41597-020-0389-7
[article]: https://doi.org/10.6084/m9.figshare.10084619.v2
[code]: https://github.com/jingkungao/PLAID

## The source data

The article holds:

| File | Content |
| --- | --- |
| `submetered.zip` | `submetered_new/<n>.csv`, n = 1 to 1876: one appliance switching on, measured alone |
| `aggregated.zip` | `aggregated/<n>.csv`, n = 0 to 575: two to four appliances switching on and off |
| `metadata_submetered.json` | One entry per submetered file, keyed by `n` (Box 1 of the paper) |
| `metadata_aggregated.json` | One entry per aggregated file, keyed by `n` (Box 2 of the paper) |
| `plaid_hdf5.zip` | The same measurements again, as HDF5; not read |
| `collecting_data.m`, `.vi` | The MATLAB and LabVIEW capture scripts; not read |

- **Signals.** Each CSV row is one sample: current in A, then voltage in V, with no header and no
  time column. A row's time is its index over 30 kHz, the `sampling_frequency` of every entry. The
  values have 0 to 8 decimals (the paper says three); the connector keeps them as float64.
- **Submetered entries.** The appliance's `type` (16 types), `load` (`R`, `I` or `NL`), operating
  mode (`status`, such as `off-on` or `highcool`), and the facts the measurer filled in: brand,
  model, year, rated current, voltage and power, notes. Also the campaign (`collection_time`: July
  2013, December 2014 or August 2017), its note, the `location` and the stated `length`.
- **Households.** Locations `house1` to `house9` recur in the 2013 and 2014 campaigns as different
  households. The authors' GlobalSIP notebook groups by `collection_time` and `location` together,
  which gives the paper's 65 locations.
- **Aggregated entries.** Two to four appliances, each with `on` and `off`: the sample indices at
  which it switched, written as a NumPy array (`[37043]`). The soldering iron switches on in two
  steps, and both are labelled. All aggregated files come from the CMU lab, August 2017.

## The TimeNet dataset

### Records

One Record per listed CSV file: 1,876 submetered, then 575 aggregated, each in file order. Each
Record has one Source, `NI-9215` (the acquisition card), with two lazy float64 signals, `current`
in ampere and `voltage` in volt, on a 30 kHz axis from 0. Values are as written: no calibration,
filtering or resampling.

`Record.metadata` is the file's entry, as written, minus blank fields; `records.record_metadata`
reads it back as `PlaidRecordMetadata`:

| Field | Value |
| --- | --- |
| `subject_id` | The household, as the authors' notebook keys it: `July, 2013_house1` |
| `subset`, `file` | `submetered` or `aggregated`, and the file's path in its zip |
| `collection_time`, `location`, `header_notes`, `length`, `status` | As written |
| `appliances` | The file's appliances in entry order: type, load type, and the filled-in facts |

Metadata is never model input. The appliance type is there as provenance; it is the appliance
task's target.

An aggregated Record carries one annotation per labelled event: key `appliance_on` or
`appliance_off`, value the appliance type as written, span a `TimePoint`. An event at sample `i`
sits at `RegularAxis.time_offset_us(i)`, sample `i`'s time floored to whole microseconds, and
`index_at_or_after` gives `i` back exactly. The release doesn't say whether its indices count from
0 or 1; they are read as counting from 0, so the alternative moves an event by one sample, 33 µs.

### Dataset annotations

| Key | Value |
| --- | --- |
| `source_url` | The article's version 2 DOI |
| `source_files` | Each pinned file, as `<name> sha256:<digest>` |
| `appliance_type_vocabulary` | The 16 submetered types, sorted; the appliance task's `target_schema` |

### Tasks

The paper names two uses and defines no benchmark. Each becomes one task per Record:

| Task | Records | TimeNet task | Target |
| --- | --- | --- | --- |
| Appliance type | 1,876 submetered | `ClassificationTask` | The type, as written |
| Switching events | 575 aggregated | `TemporalLocalizationTask` | The Record's event annotations |

- **Appliance type.** The paper: the submetered data enables "appliance classification", and
  "the generalization of the labelling methods across houses can be tested". The authors'
  GlobalSIP notebook classifies the type, on the 2013 files. It builds its features from each
  file's last 10,000 samples, in steady state. The task has no scope: the paper notes the start-up
  transient carries information about the appliance, so the whole file is the input.
- **Switching events.** The paper: on the aggregated data, "two important steps in NILM are event
  detection and load identification. This dataset provides the means to learn and implement both
  tasks". One task asks both: its `target_annotations` are the Record's events, each a point keyed
  on or off whose value is the appliance. The mode is `sparse`, since the labels need not mark
  every event (see [Known issues upstream](#known-issues-upstream)).

No task has a prompt; neither use is asked in words.

**Split.** None. The authors evaluate by leaving one household out: their notebook trains on 54
households and tests on the 55th, for each household in turn. A Record's `subject_id` is that
household, so the same folds can be rebuilt. All aggregated files share one location, so a fold by
household doesn't apply to them.

### What the build leaves out

- **`aggregated/0.csv`**, 10 s long, which no metadata entry describes. The paper's 576
  aggregated files count it.
- **The HDF5 copy and the capture scripts.**

## How the connector builds it

```text
download   metadata_*.json ── download_files, sha256 ─┐
           submetered.zip, aggregated.zip ────────────┼─► cache_dir/
             ensure_archive, sha256                   │
                                                      ▼
parse      metadata_submetered.json ── read_submetered ──► SubmeteredEntry
           metadata_aggregated.json ── read_aggregated ──► AggregatedEntry
                                                             └─ SwitchedAppliance
                                                                  └─ events():
                                                                     SwitchEvent

pair       entry + <n>.csv ── read_measurements ──► Measurement
                                                     subset, path,
                                                     n_values, entry

build      build_record(Measurement) ──► Record (lazy current, voltage)
           annotate_events(Record, AggregatedEntry) ──► event annotations

populate   dataset annotations, then the Records, then the task stream:
           appliance_tasks ────────► ClassificationTask
           switching_events_tasks ─► TemporalLocalizationTask

write      ValueLoader(measurement, column)
             └─ read_values(path): both columns, once per file
```

| Module | Holds |
| --- | --- |
| `connector.py` | The download, the provenance annotations, and `convert` |
| `release.py` | What is pinned: the files and their SHA-256, directories, rate, counts, known defects |
| `entries.py` | The metadata files: `SubmeteredEntry`, `AggregatedEntry`, `Appliance`, `SwitchEvent` |
| `measurements.py` | One CSV file: `Measurement`, its row count, its values |
| `keys.py` | Every annotation key the connector writes |
| `records.py` | A file's Record, its metadata, its events, and the lazy `ValueLoader` |
| `tasks.py` | `appliance_task` and `switching_events_task`, which build the TimeNet tasks |
| `loader.py` | `PlaidDatasetLoader`, which runs the parse and streams the tasks of `appliance_tasks` and `switching_events_tasks` |

Every build checks the pins in `release.py`: 1,876 submetered and 575 aggregated files, 2,714
events, 65 households, the one unlisted file, and the 90 entries with a wrong stated length. Every
other file's rows must last its stated `length`, to the hundredth of a second it is written to.

## Known issues upstream

- **Counts.** The paper reports 17 appliance types, 576 aggregated files and 1,314 aggregated
  appliance instances (Table 4: 1,305). The release has 16 submetered types, 575 aggregated
  entries plus the unlisted `0.csv`, and 1,303 aggregated appliance entries. The 17th type is the
  fridge defroster: the aggregated metadata names it a type, `Fridge defroster`, while the
  submetered metadata writes it as a `Fridge` with status `off-defroster`.
- **Type names differ between the files.** `Soldering Iron` in the submetered metadata is
  `Iron solder` in the aggregated one. The connector keeps both as written.
- **Calibration.** The 2013 files (1 to 1026) were not calibrated each time the set-up moved; the
  paper leaves normalization to the user, and the campaign's note says so in `header_notes`. The
  values stay as written. 317 submetered files exceed 20 A at some sample, up to 104.6 A; the paper
  states its 20 A filter for steady state. Every file's steady-state RMS voltage is within the
  paper's 110 to 130 V.
- **Misstated lengths.** 90 aggregated entries, all between 325 and 473, state 10 s for a 20 s
  file or 20 s for a 10 s one (one 20 s and 30 s). Their labelled events still match a step in their
  own file's current as often as everyone else's (94% against 93%), and far less often in the file
  numbered one before or after (37%), so only the stated length is wrong. `PlaidRecordMetadata.length` keeps it as written.
- **Status strings and labels disagree.** An aggregated entry's `status` lists its events, such as
  `on-on-off-off`. In 363 files it agrees with the labels. In 86, the labels have one more `on`, the
  soldering iron's second step, which the paper says is labelled. In 126, mostly with a fridge or a
  fridge defroster, `status` has one more `on` than the labels, so an event may be unlabelled; the
  task's mode is `sparse` for this reason.
- **One unlabelled switch-off.** File 364's air conditioner has an `on` index and no `off`.
