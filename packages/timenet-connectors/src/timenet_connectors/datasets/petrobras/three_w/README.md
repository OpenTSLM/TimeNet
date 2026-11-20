# 3W: the decisions of this connector

3W v2.0.0 ships 2,228 Parquet files, one per instance, and a configuration file, `dataset.ini`. It
ships no task, no split, and no time zone. This document states what the connector builds from it
and why.

*(measured)* marks a number that this connector took from the release at the pinned commit.

## The source

The release keeps its Parquet files in plain git, without LFS. The connector pins each file to a raw
GitHub URL at commit `c86cc7c70063646910be7292de8a54a6fb2a8399` and checks its SHA-256 from
`files.csv`. To move to a later commit, run `generate_manifest.py` on a checkout of that commit, then
update `SOURCE_REVISION` in `connector.py` and the version in `dataset.yaml`.

| Provenance | File prefix | Instances *(measured)* |
| --- | --- | --- |
| real | `WELL-<well>_<start>` | 1,119 from 40 wells |
| simulated | `SIMULATED_` | 1,089 |
| hand-drawn | `DRAWN_` | 20 |

Every file is sampled at exactly 1 Hz *(measured)*. The connector checks this and raises
`TimeNetConnectorError` otherwise.

## Records and signals

One instance becomes one record. The 27 variables become signals in seven sources, one for each
measuring point. They share six specs by quantity.

| Source | Variables | Spec |
| --- | --- | --- |
| Downhole gauge (PDG) | `P-PDG`, `T-PDG` | pressure, temperature |
| Subsea Xmas tree | `P-TPT`, `T-TPT`, `PT-P`, `P-ANULAR` | pressure, temperature |
| Production line and choke (PCK) | `P-MON-CKP`, `P-JUS-CKP`, `T-MON-CKP`, `T-JUS-CKP`, `ABER-CKP`, `P-MON-SDV-P` | pressure, temperature, valve opening |
| Gas lift line | `P-MON-CKGL`, `P-JUS-CKGL`, `ABER-CKGL`, `QGL` | pressure, valve opening, flow rate |
| Service line | `P-JUS-BS`, `QBS` | pressure, flow rate |
| Valve states | nine `ESTADO-*` variables | valve state (0, 0.5, 1) |
| Well status | `state` | well status code (0-8) |

A real instance records a median of 11 of the 27 variables *(measured)*. The connector reads which
variables an instance records from the Parquet column statistics, so it reads no value to decide.

**This connector fills in nothing.**

| What the file holds for a variable | What the record gets |
| --- | --- |
| no value in the whole instance | no signal, because the well has no such sensor. `PT-P`, `P-JUS-BS`, `QBS` and `P-MON-SDV-P` are empty in every instance *(measured)* |
| values with gaps | a signal that keeps the gaps as nulls, with the reason `not_recorded` |
| one value for the whole instance | a signal with that value and a `constant_value` annotation, because a frozen reading usually means a stuck sensor |

The release states no meaning for the `state` codes, so the connector keeps the codes. It states no
time zone for the timestamps either, so a record keeps its first timestamp as text in
`metadata["start"]` and has no absolute start time.

## Labels and tasks

Each second carries a `class` label: `0` for normal operation, `k` for the steady state of event
`k`, and `k + 100` for its transient phase. About 5% of the seconds carry no label *(measured)*. Each
run of equal labels becomes an `event_phase` interval annotation: `normal`, `transient` or `steady`.
A label that does not belong to the instance's event raises `TimeNetConnectorError`.

| The question | Type | Count *(measured)* |
| --- | --- | --- |
| which event is this instance | `ClassificationTask` over ten event types | 2,228 |
| where is the transient phase of the event | `TemporalLocalizationTask`, `EXHAUSTIVE` | 1,142 |

Events 3 (severe slugging) and 4 (flow instability) have no transient phase, so their instances get
no localization task.

## The split

The release states no split. A split by instance would put windows of one well in train and test,
so the connector splits real instances by well: `StratifiedSplitter` groups them by well and
stratifies them by event type, with 3/1/1 folds. Simulated and hand-drawn instances come from no
well, so they always go to train.

Real instances per split *(measured)*:

| Event | Train | Validation | Test |
| --- | --- | --- | --- |
| 0 Normal Operation | 429 | 58 | 107 |
| 1 Abrupt Increase of BSW | 4 | 0 | 0 |
| 2 Spurious Closure of DHSV | 17 | 2 | 3 |
| 3 Severe Slugging | 1 | 0 | 31 |
| 4 Flow Instability | 201 | 83 | 59 |
| 5 Rapid Productivity Loss | 5 | 6 | 0 |
| 6 Quick Restriction in PCK | 6 | 0 | 0 |
| 7 Scaling in PCK | 9 | 19 | 8 |
| 8 Hydrate in Production Line | 8 | 4 | 2 |
| 9 Hydrate in Service Line | 34 | 12 | 11 |

Events 1, 3, 5 and 6 have real instances from two or three wells only, so one split can hold all
of them. A user who evaluates those events should train on the simulated instances and test on the
real ones.
