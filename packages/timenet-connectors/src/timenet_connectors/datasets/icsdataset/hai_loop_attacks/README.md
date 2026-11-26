# HAI loop attacks

This connector turns the HAI security dataset into 24,132 classification tasks: 4,022 windows of
300 seconds, each seen through the six control loops of a power plant testbed. Each task shows one
window of one loop and asks whether that loop was under attack. Each of the 1,454 attack tasks also
has a localization task for the seconds in which the attack is flagged.

## Source

The series come from the [`icsdataset/hai`](../hai/README.md) parent, which holds every HAI data
file as published, with all columns, HAI's own attack flags and the units of the HAI manual. The
card declares it under `parents`, and the connector imports from it in `compose()`, as
`physionet/ecg-qa-cot` does with `physionet/ptb-xl`. A build needs the parent release in the local
registry. If it is missing, `timenet-build` fetches it from the remote registry or, by default,
builds it first.

This connector downloads only the examples table, `examples.csv.gz`, from a Hugging Face dataset
repo. `sources.py` pins it by commit and SHA-256, in one place: `EXAMPLES_REPO`,
`EXAMPLES_REVISION` and `EXAMPLES_SHA256`. The table derives from HAI, so it is licensed CC BY-SA
4.0, like HAI, and is not part of this MIT package.

## Records and signals

The examples use 17 of HAI's 28 data files. For each of them the connector imports a copy of its
parent record that holds only the series of the six loops: 21 to 23 signals per file, 379 in all.
The signals keep their column name in that version, their `float64` values and the spec of the
parent, which gives the quantity and the unit from the HAI manual. HAI's attack flags stay in the
parent: no record of this dataset holds them.

| Loop | Target | Measurement | Command |
| --- | --- | --- | --- |
| Boiler pressure (P1-PC) | `P1_B2016` (bar) | `P1_PIT01` (bar) | `P1_PCV01D`, `P1_PCV02D` (%) |
| Boiler level (P1-LC) | `P1_B3004` (mm) | `P1_LIT01` (mm) | `P1_LCV01D` (%) |
| Boiler flow (P1-FC) | `P1_B3005` (l/h) | `P1_FT03` (mmH2O) | `P1_FCV03D` (%) |
| Boiler temperature (P1-TC) | `P1_B4022`, `P1_B4002` (°C) | `P1_TIT01` (°C) | `P1_FCV01D`, `P1_FCV02D` (%) |
| Turbine speed (P2-SC) | `P2_AutoSD` (RPM) | `P2_SIT01` (RPM) | `P2_SCO` (no unit) |
| Water level (P3-LC) | `P3_LH01`, `P3_LL01` (%) | `P3_LIT01` (%) | `P3_LCP01D`, `P3_LCV01D` (no unit) |

These are the 22.04 names and the units the manual gives for them. "No unit" means the manual
prints `-`. In the flow loop the target is a flow in l/h, but the measurement `P1_FT03` is a
differential pressure in mmH2O, so the two cannot be compared as numbers. `P1_FT03Z` is the same
flow in l/h and is in the parent. The loop keeps `P1_FT03`: HAI's attack table names it in 17 flow
attacks and `P1_FT03Z` in none, and no rule of the table compares two series as numbers.

Older and newer versions name some columns differently. Each signal's metadata gives its 22.04 name
(`tag`), its loop, its role, and how its column was matched to the tag (`tag_matched_from`):

| 22.04 tag | 20.07 | 21.03 | 23.05 |
| --- | --- | --- | --- |
| `P1_B2016` | same | same | `x1001_15_ASSIGN_OUT` (`manual`) |
| `P1_B3004` | same | same | `x1002_07_SETPOINT_OUT` (`data`) |
| `P1_B3005` | same | same | `x1002_08_SETPOINT_OUT` (`data`) |
| `P1_B4002` | same | same | `x1003_18_SETPOINT_OUT` (`data`) |
| `P1_B4022` | same | same | `x1003_24_SUM_OUT` (`data`) |
| `P2_AutoSD` | left out | left out | same |
| `P2_SCO` | none | `P2_CO_rpm` (`data`) | same |
| `P3_LH01`, `P3_LL01` | `P3_LH`, `P3_LL` (`name`) | `P3_LH`, `P3_LL` (`name`) | same |
| `P3_LIT01` | `P3_LT01` (`manual`) | same | same |

- **`same_name`:** the column has the 22.04 name.
- **`manual`:** the manual names the column as that point. It lists `P3_LT01` in the row of
  `P3_LIT01`, and DCS block 1001.15 as a duplicate of `P1_B2016`.
- **`data`:** a guess from the data. In 23.05 `hai-train3`, `P1_LIT01` and `P1_FT03` track
  `x1002_07` and `x1002_08` as closely as they track `P1_B3004` and `P1_B3005` in 22.04 (r = 0.999
  and 0.997, on 10-minute means). `x1003_18` and `x1003_24` sit in the 22.04 ranges of `P1_B4002`
  and `P1_B4022` and step like them, but `P1_TIT01` does not track them (r = 0.07 and 0.11, against
  0.61 and 0.59 in 22.04). 23.05 drops the eight `P1_B` points of 22.04 and adds seven columns of
  DCS blocks 1001 to 1003, which line up with the `P1_B2`, `P1_B3` and `P1_B4` groups. `P2_CO_rpm`
  moves in all 7 attacks that name `P2_SCO`. The manual gives no unit for the 23.05 DCS columns, so
  the parent leaves their unit unset.
- **`name`:** a guess from a similar name. The manual does not list `P3_LH` and `P3_LL`.

`P2_AutoSD` is left out of 20.07 and 21.03: the manual confirms `P2_SD01` as its 20.07 name, but it
is constant in each file, and 21.03 `P2_ASD` is 0 in every row. 20.07 has no speed-controller
output. So a loop has one to five series.

## The examples table

`examples.csv.gz` names every example: which file, which window, which loop and which of its series,
the label, the split and the flagged runs. It holds no series value.

The table comes from a separate pipeline, Night Shift
([vid-t/night-shift](https://github.com/vid-t/night-shift)). It reads the attack tables of the HAI
manual, checks them against the flags in the data files, maps every version's column names to the
22.04 names, draws the windows with fixed seeds and writes the rationales by rule.
`uv run hai-build --table v2`, then `uv run hai-export --table v2`, writes this table again, byte
for byte. The export then audits the table against the raw files and fails if a check fails: the
17 file hashes, one-second steps, every label and run against HAI's flags, every run sitting
exactly on its flag inside its window, and every normal window at least 600 s from any flagged
second. It also checks every attack against the pipeline's attack table: one flag run that starts
in the stated minute and lasts the stated seconds. That table must equal the manual's attack tables
plus the fixes coded in the pipeline, each with its reason. Most fixes are durations, which the
manual prints differently from the flags: for 37 of the 38 attacks of 20.07, and by one second for
every attack of 22.04 and 23.05. Without `--table v2`, the pipeline writes the first table, on
which Night Shift was trained and scored.

### Windows

- **Attack windows:** six per attack. The attack flag starts at a random second between 30 and
  270, one draw from each of six equal bands, so a model cannot guess the start second.
- **Normal windows:** at least 600 seconds from any flagged second. Half come from the attack files,
  half from the normal file of the version.
- A window whose only attacked loops are outside the six (P2-TC, P1-CC, P1-HC) is dropped.
- A window that is an exact copy of a window of another file (same start, same rows) is dropped.
  None is, today: see the limits.

### The label rule

A window is seen once through each loop. The loop comes from the manual's scenario code, not from
the data points it lists. The example is `attack` when an attack that the manual assigns to the
loop is flagged in at least one second of the window, and `normal` otherwise, even if another loop
of the window is under attack.

The flag is the flag of the loop's own process (`attack_P1`, `attack_P2` or `attack_P3`) in 20.07
and 21.03, inside the run of the attack's `attack` flag, and HAI's attack flag in 22.04 and 23.05,
which have no flag per process. The process flags pulse during some combined attacks. For example,
`attack_P2` is on for 29 of the 255 seconds of 21.03 A308. So an example can have several flagged
runs.

## Tasks

| Task | Count | Scope | Target |
| --- | --- | --- | --- |
| `ClassificationTask` | 24,132 | the window, over the loop's series | `attack` or `normal` |
| `TemporalLocalizationTask` | 1,454 | the same | one interval per flagged run: 1,770 in all |

The localization task has one interval for each run of flagged seconds of the loop in the window.
98 examples have more than one run. A run that outlasts the window stops at the window end, and the
metadata says `cut_at_window_end: true`. The task derives from its classification task
(`from_tasks`).

### Rationales

Each classification task, except the 102 below, carries a rationale of one to five sentences,
written by rules, that ends with `Answer: attack` or `Answer: normal`. It is a rationale, not a
separate `AnswerTask`, because it explains the label rather than answering a different question. The
texts were read against the charts for 24 examples only.

The rationale reads each series with two rules: in an attack example, the 60 seconds before the
first flagged second against the seconds after it; in a normal example, the first half of the window
against the second half.

- **Level shift:** `rises` or `falls` when the mean moves by more than three times the noise of the
  60 seconds before, or by 2% of the series' normal-file range.
- **Swing change:** `starts swinging` or `stops swinging` when the one-second steps grow or shrink
  more than threefold and the swinging side goes up and down. This rule counts only in loops that
  rarely swing on their own: in quiet windows, under 10% show such a change at a random second.
  That holds for every loop except boiler temperature (P1-TC), whose texts never use the swing
  words.

The second in "Change starts near second N", which ends every attack rationale, is the first
flagged second. It comes from HAI's flag, not from the series.

102 attack examples show neither a level shift nor a swing change in any series
(`no_visible_change`). They have no rationale: a text could only name a second that the series do
not show.

### Metadata

The classification task's metadata names the example and its series: `example_id`, `window_id`,
`version`, `file`, `loop`, `tags`, `source_tags`, `roles` and `tag_matched_from`. It holds nothing
that depends on the label.

The localization task's metadata holds the attack: `attack_ids`, `attack_start_s` and
`attack_end_s` (the first and last flagged second, counted from the window start), `attack_runs`,
`cut_at_window_end`, `window_kind`, `anchor`, `scored`, `loop_uncertain` (the manual names one loop
in the scenario and another in the points) and `no_visible_change`.

Three flags of the evaluation protocol also apply to normal examples, and each correlates with the
label, so no task carries them. The dataset lists them instead, as annotations:

| Annotation | Lists | Use |
| --- | --- | --- |
| `training_sample` | the 3,850 example ids of the training sample | Train on these: the full train split is 96% normal |
| `scored_windows` | the 181 window ids, one per attack, picked at random | Score per attack on these windows only |
| `reacting_examples` | the 323 example ids of normal boiler-loop examples whose series start or stop swinging when another loop of the same process is attacked | Count on their own line, neither for nor against the model |

The lists depend on the label: every reacting example is normal, and every scored window is an
attack window. Use them to select and score examples, and never show them to a model.

The training sample holds every train attack example with a visible change (770), plus two normal
examples from attack windows and two from normal windows for each of them. It never holds a
`no_visible_change` or reacting example; the connector raises if the table says otherwise.
Validation and test are never sampled. A reacting example keeps the label `normal`. The boiler loops
are physically linked, so an attack on one can move its neighbours; whether a marked loop was
pushed by the attack or moved by chance at that second is not settled.

## Splits

By whole file, so no attack crosses splits. Normal-file windows go to train.

| Split | 20.07 | 21.03 | 22.04 | 23.05 | attack | normal |
| --- | --- | --- | --- | --- | --- | --- |
| train | test1, train1 | test1, test2, test3, train1 | test1, test2, train1 | hai-test2, hai-train3 | 819 | 17,817 |
| validation | | test4 | test3 | | 144 | 948 |
| test | test2 | test5 | test4 | hai-test1 | 491 | 3,913 |

Validation holds only 15 attacks. 20.07 and 23.05 have no validation file.

## Limits

- The label says which loop the manual says was attacked. It does not check that the attack is
  visible in that loop's series. `no_visible_change` marks the clearest cases, by the two rules
  above.
- 22.04 and 23.05 have no flag per process, so their runs come from HAI's attack flag. In a combined
  attack on two processes, a run there can be wider than the attack on that loop.
- The windows are drawn around the start of HAI's attack flag. In 20.07 and 21.03 the loop's own
  flag can start later, so its first flagged second can lie after second 270. Four 20.07 examples
  whose process flag starts only after their window are `normal`.
- `hai-21.03/train1` and `hai-21.03/test3` hold the same 12 hours (2020-07-13 00:00:00 to 12:00:00,
  43,201 identical rows). 432 train1 and 336 test3 examples lie in these hours. No two of their
  windows start on the same second, so none is dropped as a copy, but many overlap. Both files are
  in the train split, so nothing leaks today. Keep it in mind if the splits ever change.
- The rules are coarse. Slow ramps may read as `stays flat` early in the window. Half of all quiet
  examples show a level shift at a random second, so a level shift alone proves little.
- Reacting examples are judged on the whole attack, so a loop can be marked for a swing that begins
  well after the attack starts.
- 22.04 A301, A304, A305 and A306 name one loop in the scenario and another in the points. They keep
  the scenario's loop and are marked `loop_uncertain`.
- Some points that do not fit their scenario are kept as printed, for example 22.04 A303 and 20.07
  A211 to A213 (`P3_SP02`, not a 20.07 column). The loop label is not affected.
- 20.07 attack durations come from the attack flags: the manual's 20.07 durations disagree with the
  flags for 37 of 38 attacks, while the start minute and the flagged processes agree for all 38.
- 23.05: no target series moves during any setpoint attack, not even `x1001_15_ASSIGN_OUT`, which
  the manual ties to `P1_B2016`. The measurement and the command must carry the evidence. Four of
  the five 23.05 targets are matched from data (`tag_matched_from: data`), not from the manual.
- The water-level targets (`P3_LH01`, `P3_LL01` and their older names) are constant in every file,
  also during the attacks that name them.
- An attack window can overlap a second attack on another loop. That loop is labelled `attack` too,
  with its own runs.
- Each attack is scored on one window with a random start second. Another draw would move the
  counts by an attack or two.
- The ratio of the training sample (two plus two normal examples per attack example) is a choice,
  not a tuned value.
