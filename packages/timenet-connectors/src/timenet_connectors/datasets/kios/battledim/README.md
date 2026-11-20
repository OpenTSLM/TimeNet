# BattLeDIM: the release, and the decisions of this connector

BattLeDIM 2020 (Battle of the Leakage Detection and Isolation Methods) released two years of
SCADA data from L-Town, a water distribution network, together with the 33 leaks the data
contains.
- The data is **simulated**. The release's own configuration names `L-TOWN_Real.inp` as the
  network that generated it.
- The connector pins every file it downloads by sha256, so a changed file fails the build instead
  of silently building something else.

This document covers:
- what the release holds;
- how the connector maps it onto TimeF;
- the decisions the release leaves open;
- what a user should know before training on it.

*(measured)* marks a number taken from the files, not from the dataset page.

## Source, licence and citation

**Source:** Zenodo record [4017659](https://zenodo.org/records/4017659), doi:10.5281/zenodo.4017659, published 2020-09-07. It has one version.

**Licence:** CC BY 4.0, as stated on the Zenodo record. The organisers' GitHub repository
[KIOS-Research/BattLeDIM](https://github.com/KIOS-Research/BattLeDIM) states the same. A user must credit the authors.

**Citation:**
> Vrachimis, S. G., Eliades, D. G., Taormina, R., Kapelan, Z., Ostfeld, A., Liu, S., Kyriakou, M.,
> Pavlou, P., Qiu, M., and Polycarpou, M. M. (2022). Battle of the Leakage Detection and Isolation
> Methods. *Journal of Water Resources Planning and Management*, 148(12).
> doi:10.1061/(ASCE)WR.1943-5452.0001601

**Files the connector downloads** (13 of the 18 on the record):

| File | What it holds |
| --- | --- |
| `2018_SCADA_*.csv`, `2019_SCADA_*.csv` | Pressures, Flows, Levels and Demands, one CSV per year and type |
| `2018_Leakages.csv`, `2019_Leakages.csv` | The outflow of every leak active that year, in m³/h |
| `dataset_configuration.yaml` | The configuration that generated the data: time range, the leak list, the sensor ids |
| `2018_Fixed_Leakages_Report.txt` | The repair times of the 2018 leaks that were fixed |
| `L-TOWN.inp` | The EPANET model given to participants (nominal parameters) |

**Files it skips:**
- The two `.xlsx` files, which repeat the CSVs (README.txt of the release).
- `L-TOWN_Real.inp` (177 MB), the simulation model. Nothing in the tasks needs it.

## The record

The two years join into **one record** on one 5-minute grid:
- 2018-01-01 00:00 → 2019-12-31 23:55, 210 240 steps *(measured)*.
- The 2019 files continue the 2018 files with no gap.
- **Time zone:** the release writes naive timestamps and states no zone. The connector reads them as UTC.

Each sensor type is one source. The signal names are the EPANET ids of `L-TOWN.inp`.

| Source | Signals | Unit (README.txt of the release) | Range *(measured)* |
| --- | --- | --- | --- |
| Pressure sensors | 33: n1, n4, n31, n54, n105, n114, n163, n188, n215, n229, n288, n296, n332, n342, n410, n415, n429, n458, n469, n495, n506, n516, n519, n549, n613, n636, n644, n679, n722, n726, n740, n752, n769 | m (pressure head) | 25.7–57.0 |
| Flow sensors | PUMP_1 (pump into tank T1), p227, p235 (the two reservoir inlets) | m³/h | 0–205 |
| Tank level sensor | T1 | m | 1.95–3.90 |
| Automated meter readings | 82 nodes, the `amrs` list of the configuration | L/h | 0–37 796 |

**Reading and checking:**
- **Values:** float64, as parsed from the decimal-comma text. Nothing is filled, trimmed or resampled.
- **Grid:** a step that is not 5 minutes fails the build.
- **Values:** a missing or non-finite value fails the build.
- **Sensors:** each year's columns must equal the sensor list of the configuration.
- **Network:** every sensor must exist in `L-TOWN.inp`, as a node or, for flow, as a link.

## The leaks

The configuration lists **33 leaks** *(measured)*:
- 15 abrupt and 18 incipient;
- hole diameters of 8.8–22.9 mm;
- peak outflows of 5.1–35.4 m³/h.

| Year | Leaks starting | Active during the year | Repaired that year | Open at the end of 2019 |
| --- | --- | --- | --- | --- |
| 2018 | 14 | 14 | 10 | 4 carry over into 2019 |
| 2019 | 19 | 23 | 7 | 16 in total |

An **abrupt** leak appears at full size at its start. An **incipient** leak grows from its start until its `peakTime`.

Each leak becomes one **dataset annotation** with the key `leak`. Its value is the pipe's EPANET id, and its metadata holds:
- `leak_type`: `abrupt` or `incipient`;
- `start`, `end`, `peak`: the configured times, as text;
- `start_us`, `end_exclusive_us`, `peak_us`: the same times in microseconds from the record start. The interval runs from the configured start to the step after the configured end;
- `hole_diameter_mm`;
- `peak_outflow_m3h`: the largest outflow in the leak tables;
- `repaired`: whether the leak was repaired before the end of 2019.

**The leaks are answers, so they are not on the record.** Every task reads the record, and anything that serves the record to a model also serves the record's annotations. TimeNet's `TimeFTorchDataset`, for example, returns them in each item. The record therefore holds the sensor signals and nothing else. Each window's answers are in its task targets.

**Consistency checks.** The build fails if any of these disagree:
- the configuration, the two leak tables and the repair report, which must list every leak that ends in 2018, exactly once;
- a leak's outflow and its configured start and end (for p800 the leak tables keep one step of outflow after its end, an exception the connector names);
- a leak pipe and the pipes of `L-TOWN.inp`.

It also fails on:
- an abrupt leak whose peak is not its start;
- a sensor listed twice in the configuration;
- a configuration without its `times` mapping or `leakages` list;
- a CSV cell that is not a timestamp or a number.

The release passes all of these checks.

## Windows and tasks

**Windows:** a window is 14 days, starting every 7 days.
- The first week is context; the second week is the **target week**.
- The recording holds **103 windows** *(measured)*.
- Every leak start and every repair falls in exactly one target week: 33 starts, 17 repairs. The build checks this and fails otherwise.
- 33 windows have a new leak, 16 have a repair, and 59 have neither *(measured)*.
- **Embargo:** 5 windows get no tasks, because their context week is a held-out target week (see The split). They hold no new leak, but two of them describe a repair (p514, p142), so those two repairs appear in no task.

**Tasks:** each of the other 98 windows gives four tasks, all scoped to the window. That is 392 tasks.

| Task | Answer | Source of the answer |
| --- | --- | --- |
| `ClassificationTask` | the types of the leaks that start in the target week: a multi-label set {`abrupt`, `incipient`}, empty when none starts | configuration |
| `TemporalLocalizationTask` (exhaustive) | a time point at each leak start in the target week | configuration |
| `ScalarPredictionTask` `mean_leak_outflow` | the mean total leak outflow over the target week, m³/h | leak tables |
| `AnswerTask` | a short description of every leak that starts or is repaired in the target week | configuration, leak tables, `L-TOWN.inp` |

**How the text is made.** Fixed templates fill in the answer; no model writes it.

*New leak:*
> A sudden leak started on pipe {pipe} on {start}, with a hole of {diameter} mm and an outflow of up
> to {max outflow in the window} m3/h in this window. The nearest pressure sensor is {sensor}, {n}
> links away.

**Incipient leaks.** The sentence opens "A gradually growing leak started on pipe {pipe} on {start}". It continues with:
- "and peaked on {peakTime}" if the peak lies inside the window;
- "and was still growing at the end of this window" if it does not. A peak after the window is not stated, because the scoped input cannot show it.

**Outflow figures.**
- An outflow under 0.1 m³/h keeps the leak tables' second decimal, e.g. "0.02 m3/h".
- When the leak tables hold no outflow at all in the window (an incipient leak's first hours), the text says "no outflow in the leak tables yet". This applies to 3 windows *(measured)*.

*Repair:*
> The leak on pipe {pipe} was repaired on {end}; its outflow before the repair was {outflow} m3/h.

*Neither:*
> No leak started or was repaired in the last 7 days.

*Always last:*
> At the end of the window {k} leaks are active, with a total outflow of {total} m3/h.

A leak counts as active when the leak list's interval holds the window's last step. The total is the leak tables' outflow at that step.

**Nearest sensor.** "Nearest" counts links in the `L-TOWN.inp` graph (pipes, pumps and valves) from the nearer end of the leaking pipe. Ties go to the sensor listed first in the configuration.

**Example input and output.** The window that starts on 2018-01-29 (target week 2018-02-05) holds all 119 signals over 14 days. Its text target is:

> The leak on pipe p232 was repaired on 2018-02-10 09:20; its outflow before the repair was 26.0
> m3/h. At the end of the window 2 leaks are active, with a total outflow of 9.8 m3/h.

**Task metadata:** `window_index`, `window_start`, `target_week_start`, `calendar_block`, `new_leaks` and `repaired_leaks`.

## The split

The release's own partition was 2018 for training and 2019 for evaluation. **This connector does not use it.**
- **Why:** 74 of the 82 meter series repeat 2018 exactly in 2019 *(measured)*. As a result, the 2019 inflow minus the 2018 inflow at the same calendar time equals the 2019 leak outflow minus the 2018 leak outflow: slope 1.00, daily correlation 0.989 *(measured)*.
- **The risk:** a year split would test on a demand background the model saw in training.

**What it does instead:**
- Windows are grouped by the **calendar block** of their target week: the ISO week, in blocks of four.
- A block holds the same weeks of both years.
- `StratifiedSplitter` (8/1/1 folds, stratified on "a new leak starts", seed 42) assigns whole blocks.

**Embargo.** Windows overlap by one week: a window's context week is the target week of the window before it.
- A window gets no tasks when that week is a validation or test target week of a different partition.
- Otherwise a training window would show the model the signals that a test task asks about.
- This affects 5 training windows *(measured)*.

**Result** *(measured)*:

| Partition | Windows | With a new leak | Tasks | Calendar blocks |
| --- | --- | --- | --- | --- |
| train | 74 | 27 | 296 | 0, 1, 2, 3, 5, 7, 8, 9, 10, 11 |
| validation | 8 | 3 | 32 | 4 |
| test | 16 | 3 | 64 | 6, 12 |

**Remaining overlap:** a held-out window's context week can be a training target week. That shows held-out tasks training data, not the other way round. Target weeks never repeat across partitions.

**One leaked day per block boundary:** the 2019 demand repeats 2018 by calendar date, but 2019 weekdays fall one day later. A 2019 target week (Monday to Sunday) therefore shares one date with the 2018 week before it. Where a block boundary separates partitions, one day of a target week's demand background also appears in the neighbouring partition. Week-aligned windows cannot avoid this.

## Release details a user should know

- **Simulated data.** The SCADA values come from a hydraulic simulation of `L-TOWN_Real.inp`. They are not field measurements.
- **Leaks almost never stop.** At least one leak is active in 97.8 % of 2018 and in 100 % of 2019, with up to 16 at once *(measured)*. There is almost no leak-free baseline.
- **Incipient onsets are not visible at once.** For incipient leaks, the leak tables show the first non-zero outflow 3 hours to 2.5 days after the configured start, because the early ramp rounds to zero *(measured)*. The localization target is the configured start.
- **One end time differs.** For p800, the last non-zero outflow is 5 minutes after the configured end *(measured)*. The connector keeps the configured end.

## Limitations and intended use

**Limitations:**
- 33 leak events on one network.
- Small validation and test partitions (8 and 16 windows).
- Text targets are templates, not expert explanations.
- Some window statements (the active leaks, the outflow) come from labels that a model can only infer indirectly from the sensors.

**Intended use:**
- Training and evaluating time-series language models that detect, time, size and describe leaks from SCADA data.
- Studying explanations that can be checked against a known ground truth.

*Contributed by Team RAMadan during the TimeNet hackathon, Round 1 (November 2026).*
