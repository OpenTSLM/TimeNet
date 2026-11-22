# HAI loop attacks

This connector turns the HAI security dataset into 24 132 classification tasks. Each task shows
300 seconds of one control loop of a power plant testbed and asks whether that loop was under
attack. Each of the 1 458 attack tasks also has a localization task for the attack interval.

## Source

[HAI](https://github.com/icsdataset/hai) records a hardware-in-the-loop testbed of a boiler, a
turbine and a water treatment process, one row per second, with staged attacks. The connector
reads four versions (20.07, 21.03, 22.04 and 23.05) at commit
`2a814cebc9a66b06c9e5cd545e2d72e65d383737`. Every file is pinned by URL and SHA-256. 20.07 and
21.03 come from `raw.githubusercontent.com`. 22.04 and 23.05 sit in Git LFS, so they come from
`media.githubusercontent.com`. Their raw link returns a 133-byte pointer, not the data.

HAI is licensed under CC BY-SA 4.0. The card's `citation` holds the three citations that the HAI
README asks for.

The connector reads the 17 data files that hold examples. It does not need the two 23.05 label
files, because the examples table already holds the labels.

| version | separator | time column | line end | files |
| --- | --- | --- | --- | --- |
| 20.07 | `;` | `time` | LF | test1, test2, train1 |
| 21.03 | `,` | `time` | LF | test1 to test5, train1 |
| 22.04 | `,` | `timestamp` | CRLF | test1 to test4, train1 |
| 23.05 | `,` | `timestamp` | CRLF | hai-test1, hai-test2, hai-train3 |

The connector checks that every file steps one second per row and fails on a gap.

## Records and signals

Each data file is one record. Its metadata holds the version, the file, its first time stamp and
its length in seconds. HAI prints local wall-clock times without a time zone, so the record keeps
the time stamp as text.

The record's signals are the series that the six control loops use, in `float64` on a one-second
axis, under the column names of that version. The other columns of the file are left out: the
tasks do not use them, and they would make the build three to four times larger (HAI has 59 to 86
sensor columns per version, the loops use 21 to 23). HAI states no
unit per column, so every spec leaves the unit unknown. The spec type gives the role of the
series in its loop: `control_setpoint` (target), `process_measurement` or `control_command`.

| Loop | Target | Measurement | Command |
| --- | --- | --- | --- |
| Boiler pressure (P1-PC) | `P1_B2016` | `P1_PIT01` | `P1_PCV01D`, `P1_PCV02D` |
| Boiler level (P1-LC) | `P1_B3004` | `P1_LIT01` | `P1_LCV01D` |
| Boiler flow (P1-FC) | `P1_B3005` | `P1_FT03` | `P1_FCV03D` |
| Boiler temperature (P1-TC) | `P1_B4022`, `P1_B4002` | `P1_TIT01` | `P1_FCV01D`, `P1_FCV02D` |
| Turbine speed (P2-SC) | `P2_AutoSD` | `P2_SIT01` | `P2_SCO` |
| Water level (P3-LC) | `P3_LH01`, `P3_LL01` | `P3_LIT01` | `P3_LCP01D`, `P3_LCV01D` |

These are the 22.04 names. Older and newer versions name some columns differently, and some
loops lack a series in some versions, so a loop has one to five series. Each signal's metadata
gives its 22.04 name (`tag`), its loop and its role.

## The examples table

`examples.csv.gz` names every example: which file, which 300-second window, which loop and which
of its series, the label, the split and the attack interval. It holds no series value. The
connector builds the series from the raw files.

The table derives from HAI, so it is licensed CC BY-SA 4.0, like HAI. TimeNet is MIT, so the
table is not in this repository. It sits in its own repository,
[vid-t/hai-loop-attacks-data](https://github.com/vid-t/hai-loop-attacks-data), with HAI's
attribution and the CC BY-SA 4.0 licence. The connector downloads it from
`https://raw.githubusercontent.com/vid-t/hai-loop-attacks-data/<commit>/examples.csv.gz`,
pinned to commit `0e569b70a270908a86feba8bdc3b4d4f3bce0f4c` (`EXAMPLES_OWNER`,
`EXAMPLES_REPO` and `EXAMPLES_COMMIT` in `connector.py`), and checks its SHA-256
(`EXAMPLES_SHA256`).

To read the table from a local folder instead, for example a checkout of that repository, set
`HAI_LOOP_ATTACKS_DIR` to the folder. The connector then reads `examples.csv.gz` from there and
checks the same SHA-256. The variable reaches an isolated build, because the build environment
inherits the caller's environment.

```bash
HAI_LOOP_ATTACKS_DIR=/path/to/hai-loop-attacks-data \
    uv run timenet-build build icsdataset/hai-loop-attacks --local-registry /tmp/hai-registry
```

The table comes from a separate pipeline that reads the attack tables of the HAI manual, checks
them against the attack flags in the data files, maps every version's column names to the 22.04
names, draws the windows with fixed seeds and writes the rationales by rule. That pipeline is not
part of this repository. It lives in Night Shift,
[vid-t/night-shift](https://github.com/vid-t/night-shift): `uv run hai-build`, then
`uv run hai-export`, writes the table again, byte for byte.

### Windows

- **Attack windows:** six per attack. The attack starts at a random second between 30 and 270,
  one draw from each of six equal bands, so a model cannot guess the start second.
- **Normal windows:** at least 600 seconds from any attack. Half come from the attack files, half
  from the normal file of the version.
- A window whose only attacked loops are outside the six (P2-TC, P1-CC, P1-HC) is dropped.

### The label rule

A window is seen once through each loop. The example is `attack` when an attack that the HAI
manual assigns to that loop is flagged in at least one second of the window. Otherwise it is
`normal`, even if another loop of the same window is under attack. The loop comes from the
manual's scenario code, not from the data points it lists.

## Tasks

| Task | Count | Scope | Target |
| --- | --- | --- | --- |
| `ClassificationTask` | 24 132 | the window, over the loop's series | `attack` or `normal` |
| `TemporalLocalizationTask` | 1 458 | the same | the attack interval |

The localization task runs from the first flagged second of the loop's first attack in the window
to its last flagged second, inclusive. When the attack outlasts the window, the interval stops at
the window end and the metadata says `cut_at_window_end: true`. It derives from its
classification task (`from_tasks`).

Each classification task carries a rationale: one or two sentences, written by rules from the
series, that say how each series moves and end with `Answer: attack` or `Answer: normal`. It is a
rationale, not a separate `AnswerTask`, because it explains the label rather than answering a
different question. The texts were read against the charts for a few dozen examples only.

The rationale reads each series with two rules, at the second the attack starts (or a seeded
second in a normal window):

- **Level shift:** `rises` or `falls` when the mean moves by more than three times the noise of
  the 60 seconds before, or by 2% of the series' normal-file range.
- **Swing change:** `starts swinging` or `stops swinging` when the one-second steps grow or
  shrink more than threefold and the swinging side goes up and down. This rule counts only in
  loops that rarely swing on their own: in quiet windows, under 10% show such a change at a random
  second. That holds for every loop except boiler temperature (P1-TC), whose texts never use the
  swing words.

The task metadata holds the example id, version, file, loop, the 22.04 and source names of its
series, the role of each series, and the columns below.

| Column | Meaning | Use |
| --- | --- | --- |
| `scored` | True for one of the six windows of each attack, picked at random | Score per attack on these windows only |
| `no_visible_change` | True for an attack example whose series show neither a level shift nor a swing change (98 examples) | Leave out of training; count on its own line when scoring |
| `reacting` | True for a normal example in an attack window whose series start or stop swinging when another loop of the same process is attacked (323 examples, all boiler loops) | Leave out of training; count on its own line, neither for nor against the model |
| `train_use` | True for the training sample (3 880 examples) | Train on these rows: the full train split is 96% normal |
| `attack_ids` | The HAI attack ids of the loop in this window | |
| `attack_start_s` | The second of the window where the attack starts, or null | |
| `loop_uncertain` | True when the manual names one loop in the scenario and another in the points | |

The training sample holds every train attack example with a visible change (776), plus two
normal examples from attack windows and two from normal windows for each of them. It never holds
a `no_visible_change` or `reacting` example; the connector raises if the table says otherwise.
Validation and test are never sampled.

`reacting` is a mark, not a label: the label of a reacting example stays `normal`. The boiler
loops are physically linked, so an attack on one can move its neighbours. Whether a marked loop
was pushed by the attack or moved by chance at that second is not settled.

## Splits

By whole file, so no attack crosses splits. Normal-file windows go to train.

| Split | 20.07 | 21.03 | 22.04 | 23.05 | attack | normal |
| --- | --- | --- | --- | --- | --- | --- |
| train | test1, train1 | test1, test2, test3, train1 | test1, test2, train1 | hai-test2, hai-train3 | 822 | 17 814 |
| validation | | test4 | test3 | | 144 | 948 |
| test | test2 | test5 | test4 | hai-test1 | 492 | 3 912 |

Validation holds only 15 attacks. 20.07 and 23.05 have no validation file.

## Limits

- The label says which loop the manual says was attacked. It does not check that the attack is
  visible in that loop's series. `no_visible_change` marks the clearest cases, by the two rules
  above.
- The rules are coarse. Slow ramps may read as `stays flat` early in the window. Half of all
  quiet examples show a level shift at a random second, so a level shift alone proves little.
- `reacting` looks at the whole attack, so a loop can be marked for a swing that begins well
  after the attack starts.
- 22.04 A301, A304, A305 and A306 name one loop in the scenario and another in the points. They
  keep the scenario's loop and are marked `loop_uncertain`.
- Some points that do not fit their scenario are kept as printed, for example 22.04 A303 and 20.07
  A211 to A213 (`P3_SP02`, not a 20.07 column). The loop label is not affected.
- 20.07 attack durations come from the attack flags: the manual's 20.07 durations disagree with
  the flags for 37 of 38 attacks, while the start minute and the flagged processes agree for all
  38.
- 23.05: no target series moves during any setpoint attack, so the measurement and the command
  must carry the evidence. `P1_B3004` and `P1_B3005` are matched from the data, not the manual.
  `P1_B4022` and `P1_B4002` are not matched, so the 23.05 temperature loop has no target.
- The water-level targets (`P3_LH01`, `P3_LL01` and their older names) are constant in every file,
  also during the attacks that name them.
- Turbine speed: 20.07 has only `P2_SIT01`, so that loop has one series there. 21.03 has two and
  no target, because `P2_ASD` is always 0.
- An attack window can overlap a second attack on another loop. That loop is labelled `attack`
  too, with its own start second.
- Each attack is scored on one window with a random start second. Another draw would move the
  counts by an attack or two.
- The ratio of the training sample (two plus two normal examples per attack example) is a choice,
  not a tuned value.
- Whether `P1_B4002` moves during the temperature setpoint attacks was not checked.
