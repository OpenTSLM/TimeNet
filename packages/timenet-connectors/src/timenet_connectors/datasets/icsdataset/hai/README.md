# HAI

This connector holds the HAI security dataset as published: every data file of versions 20.07,
21.03, 22.04 and 23.05, with all columns, HAI's own attack labels and the units of the HAI manual.
It adds no task. Child datasets, such as `icsdataset/hai-loop-attacks`, add tasks on top of it.

## Source

[HAI](https://github.com/icsdataset/hai) records a hardware-in-the-loop testbed of a power plant
once per second: a boiler (P1), a turbine (P2) and a water treatment process (P3), coupled through
a power grid simulator (P4). Its authors staged attacks on the control loops. The connector reads
the repository at commit `2a814cebc9a66b06c9e5cd545e2d72e65d383737`. Every file is pinned by URL
and SHA-256 in `files.py`. 20.07 and 21.03 come from `raw.githubusercontent.com`. 22.04 and 23.05
sit in Git LFS, so they come from `media.githubusercontent.com`; their pinned SHA-256 is the `oid`
of the LFS pointer.

HAI is licensed under CC BY-SA 4.0. The card's `citation` holds the three citations that the HAI
README asks for.

| version | separator | time column | line end | data files | HAI's labels | rows | points per file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20.07 | `;` | `time` | LF | test1, test2, train1, train2 | `attack`, `attack_P1` to `attack_P3` | 995,400 | 59 |
| 21.03 | `,` | `time` | LF | test1 to test5, train1 to train3 | `attack`, `attack_P1` to `attack_P3` | 1,323,608 | 79 |
| 22.04 | `,` | `timestamp` | CRLF | test1 to test4, train1 to train6 | `Attack` | 1,365,602 | 86 |
| 23.05 | `,` | `timestamp` | CRLF | hai-test1, hai-test2, hai-train1 to hai-train4 | `label-test1.csv`, `label-test2.csv` | 1,180,800 | 86 |

The connector leaves out what is not a data file of these four versions: HAIEnd 23.05 (a
separate dataset of the boiler's DCS), the 22.04 `summary` folder and the 23.05 summary texts.

## Records and signals

Each of the 28 data files is one record on a one-second axis. Its metadata holds the version, the
file, its path, its first time and its length in seconds. HAI prints local wall-clock times without
a time zone, so the record keeps the first time as text. The connector checks that every file steps
one second per row and fails on a gap.

A record has two sources:

- **`HAI data points`:** one `float64` signal for each data point of the file, under its column
  name, in file order. That is 2,244 signals over the 28 files.
- **`HAI attack flags`:** HAI's own labels, as `int8` signals: `attack` (any process) and
  `attack_P1` to `attack_P3` (one per process) in 20.07 and 21.03, `Attack` in 22.04, and `label`
  in the two 23.05 test files, read from their label files. HAI publishes no labels for the 23.05
  training files, so their records have no flag. That is 60 signals.

`label-test2.csv` prints its times to the minute, such as `2022-08-17 0:00`. The connector checks
that every label row has the time of its data row, or, in that file, its minute, and fails
otherwise.

## Units

The spec of each data point comes from the "HAI data points" table of
`hai_dataset_technical_details.pdf` (pages 12 to 15, at the pinned commit), copied into
`points.py`. The spec names the quantity in the manual's words, for example
`Heat-exchanger outlet pressure` in `bar` for `P1_PIT01`, and carries the manual's unit. Two
points that the manual describes in the same words and units share a spec: `P1_FT01` and
`P1_FT03`, and `P2_VTR03` and `P2_VTR04`. The manual's wording is kept, typos included
(`PCV2`, and `P2_VTR04` described as the limit of `P2_VIBTR03`).

| Manual | Unit |
| --- | --- |
| bar, mm, l/h, mmH2O, ℃, %, RPM, ㎛, rad/s, mHz, Hz, MW | the same unit in `ureg` |
| Voltage (`P2_24Vdc`) | volt |
| Boolean | dimensionless |
| `-` | unset |

The unit stays unset where the manual gives none:

- **The manual prints `-`:** `P2_SCO`, `P3_FIT01`, `P3_LCP01D`, `P3_LCV01D`, `P3_PIT01`,
  `P4_ST_GOV`, `P4_ST_PT01` and `P4_ST_TT01`.
- **The manual does not list the column:** 20.07 `P2_VT01e`, `P3_LH` and `P3_LL`; 21.03 `P2_ASD`,
  `P2_CO_rpm`, `P2_HILout`, `P2_MSD`, `P2_SIT02`, `P3_LH` and `P3_LL`; 23.05 `P1_PP04D`. Each gets
  a spec of its own, named after the column.
- **23.05 DCS blocks:** 23.05 has seven columns named after blocks of the boiler's DCS, such as
  `x1002_07_SETPOINT_OUT`. The manual lists those blocks among the HAIEnd points, with `-` as the
  unit, and says that block 1001.15 duplicates `P1_B2016`. So `x1001_15_ASSIGN_OUT` gets the spec
  of `P1_B2016` (bar), and the other six get a spec of their own without a unit.

The manual names some points under an older name in brackets. Those columns get the spec of the
point: 20.07 `P2_Auto`, `P2_SD01`, `P2_Emgy`, `P2_On` and `P3_LT01`, and 20.07 and 21.03
`P2_VYT02`, `P2_VXT02`, `P2_VYT03` and `P2_VXT03`. The files spell `P2_AutoGo` as `P2_AutoGO`.
Each signal's metadata names its point of the manual (`manual_point`), or `null`.

## Size

The build downloads about 1.7 GB, holds up to about 3.8 GB in memory while it writes, and stores
about 350 MB. Values stay in `float64`, exact to the source text.

## Limits

- `hai-21.03/train1.csv.gz` and `hai-21.03/test3.csv.gz` hold the same 12 hours
  (2020-07-13 00:00:00 to 12:00:00, 43,201 rows with the same values in every column). Both files
  are records here, so these hours are stored twice.
- `hai-20.07/train2.csv.gz` holds 776 flagged seconds, although its name says training.
- Points that share a name across versions are taken to be the same point, as the manual's table
  does.
