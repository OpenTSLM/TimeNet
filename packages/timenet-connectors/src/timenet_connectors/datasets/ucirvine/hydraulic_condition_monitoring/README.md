# Hydraulic condition monitoring: the release, and the decisions of this connector

UCI dataset 447 records a hydraulic test rig that repeats one 60 s load cycle while four of its
components are held at documented degradation levels. Seventeen sensors record every cycle. The
release is one archive, published once, in April 2018. The connector pins it by sha256, so a changed
archive fails the build instead of silently building something else.

This document states what the release holds, how the connector maps it onto TimeF, and the
decisions the release leaves open. *(measured)* marks a number taken from the release itself, and
not from the dataset page.

## The source of truth

The archive holds one tab-separated matrix per sensor, with one row per cycle and one column per
sample, plus `profile.txt`, `description.txt` and `documentation.txt`.

| Fact | Source of truth | Where the connector keeps it |
| --- | --- | --- |
| sensor names, physical quantities, units, rates | `description.txt`, transcribed into `_SENSORS` | the `TimeSeriesSpec` and `RegularAxis` of each signal |
| which sensors are virtual | `description.txt` marks `CE` and `CP` "(virtual)" | a `derived` annotation on those two signals |
| the four component levels of a cycle | `profile.txt` | one `<component>_level` annotation per record, with its unit |
| the meaning of each level | `description.txt` | the `<component>_vocabulary` dataset annotations and the task targets |
| whether the rig had settled | the stable flag in `profile.txt` | a `stable` annotation per record |

The connector replaces no stated fact. A row of `profile.txt` with a level that `description.txt`
does not document fails the build, and so does a sensor file whose shape is not the number of
cycles by the sensor's rate times 60 s. The connector never fills, trims or resamples.

## Records

**One record is one cycle.** A cycle is the unit that `profile.txt` labels, so it is the smallest
thing the release says anything about. A record per run would lose the ability to answer for one
cycle, and a record per sensor would split a multivariate measurement whose signals only mean
something together. A build gives 2205 records *(measured)*.

**One source holds all seventeen signals.** The release says nothing about the sensors as
separate devices beyond their names, so there is nothing to put on seventeen sources. The physical
quantity lives on each signal's spec: `pressure`, `motor_power`, `volume_flow`, `temperature`,
`vibration`, `efficiency_factor`, `cooling_efficiency` and `cooling_power`. Each signal keeps its
own rate, so a cycle holds 6000 samples for a pressure sensor and 60 for a temperature sensor.

| Signals | Quantity | Unit | Rate |
| --- | --- | --- | --- |
| `PS1` to `PS6` | pressure | bar | 100 Hz |
| `EPS1` | motor power | W | 100 Hz |
| `FS1`, `FS2` | volume flow | l/min | 10 Hz |
| `TS1` to `TS4` | temperature | degC | 1 Hz |
| `VS1` | vibration | mm/s | 1 Hz |
| `SE` | efficiency factor | % | 1 Hz |
| `CE`, `CP` | cooling efficiency, cooling power (virtual) | %, kW | 1 Hz |

**Values load lazily, one sensor file at a time.** The seven 100 Hz files are 47 MB to 91 MB of
text each *(measured)*. The connector parses a file on its first use and serves one row per
record, so a build reads each file once and never holds all of them as separate arrays.

**The levels, the stable flag and the run number are record annotations, not task inputs.** The
levels are the answers. The flag and the run describe the measurement rather than the machine. A
model should not read any of them. The numeric level keeps its unit (`percent` for cooler and
valve, `bar` for the accumulator, none for the pump's ordinal codes), so a consumer who wants a
regression target has it beside the record.

**The run number is derived.** The release does not state runs. The connector numbers each
maximal stretch of consecutive cycles that share all four levels, starting at 0. That gives 194
runs *(measured)*: 154 of 10 cycles, 36 of a single cycle, one of 19, and the three warm-up runs
described below. The upstream cycle index is in `Record.metadata["cycle_index"]`.

## Tasks

**Each cycle gets four `ClassificationTask`s, one per component.** The release varies four things
at once and `documentation.txt` reports that the literature treats them as four targets of
different difficulty, with the accumulator the hardest. Four tasks let a consumer train or
evaluate one component at a time, and the union of four predictions is the joint answer. A build
gives 8820 tasks *(measured)*.

**The target is the level's documented name, not its number.** A task answers "severe lag", not
"80". `documentation.txt` says the levels describe degradation processes and are continuous
values rather than categories. The release still holds only three or four settings per component,
chosen by the experimenters, so a regression would predict values the rig never ran. The numeric
level stays on the record annotation for whoever wants it.

**Each task cites a vocabulary.** The dataset carries one `<component>_vocabulary` annotation
listing the labels in `description.txt` order, and each task names it as `target_schema`. A
consumer reads the full label set from the dataset without scanning the tasks.

**An unstable cycle keeps its tasks.** The flag says the rig may not have reached static
conditions. Its levels are still the levels the experimenters set, so the label is true. What the
flag changes is where the cycle lands in the split, and only for the warm-up runs.

## The split

The release ships no split. Where a cycle lands is this connector's decision, and the split is
stored on each task.

**Cycles that share all four levels never straddle a partition.** The release holds 144 level
combinations *(measured)*. Consecutive cycles of one combination are near copies, and the same
combination can recur hours apart. The splitter groups by combination, so a held-out combination is
unseen in training and a test score measures generalisation to a new combination of conditions,
not recall of a neighbouring cycle.

**The split stratifies on the accumulator level.** It is the hardest target in the literature and
the one with the least even level counts in the release. The ten-fold split assigns groups so that
every partition holds all four accumulator levels.

**The three warm-up runs go to train.** After each change of the cooler setting the rig takes a
long run to reach its new thermal state. The release holds three such runs, of 210, 200 and 200
cycles, flagged unstable throughout *(measured)*. Any run of 50 cycles or more counts as a warm-up.
A warm-up run would fill a whole validation or test partition with one combination, so it is placed
in train before the splitter runs. Its labels are still true, so it trains rather than being
dropped.

**Short unstable runs follow their combination.** Only the warm-up runs are forced. The other 146
unstable cycles sit in short runs *(measured)* and land wherever the splitter puts their
combination. A consumer who wants a test set of settled cycles only filters on the `stable`
annotation.

The resulting partitions, with seed 42 *(measured)*:

| Partition | Cycles | Combinations | Unstable cycles |
| --- | --- | --- | --- |
| train | 1888 | 115 | 729 |
| validation | 154 | 14 | 14 |
| test | 163 | 15 | 13 |

Train is large because it holds the three warm-up runs. Validation and test are small, about 150
cycles each, because the ten-fold split gives one fold to each and the warm-up runs are out of the
pool.

## What the release states that a consumer should know

**The stable flag covers whole runs.** No run mixes stable and unstable cycles *(measured)*. Of the
2205 cycles, 756 are flagged unstable, 610 of them in the three warm-up runs.

**Single-cycle runs exist.** 36 runs hold one cycle *(measured)*: stretches where a component
setting changes on every cycle. Each such cycle still belongs to a combination of 10 or more
cycles, so the group split handles it like any other.

**The dataset page's class counts match the release.** The per-level instance counts in
`documentation.txt` agree with `profile.txt` *(measured)*: cooler 732 / 732 / 741, valve 1125 /
360 / 360 / 360, pump 1221 / 492 / 492, accumulator 599 / 399 / 399 / 808.

**The signals are raw.** The release says it holds raw sensor data without feature extraction,
and the connector stores them as float32 with no other change. The papers listed in
`documentation.txt` reach their reported accuracies with extracted features, which a consumer
computes at load time.
