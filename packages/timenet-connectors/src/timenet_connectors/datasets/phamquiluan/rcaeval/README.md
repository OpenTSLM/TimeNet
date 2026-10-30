# RCAEval

`phamquiluan/rcaeval` holds RCAEval ([WWW 2025 companion][paper], [arXiv:2412.17015][arxiv]), a
benchmark for root cause analysis (RCA) of microservice systems. Its 735 failure cases each inject
one fault into one service of Online Boutique, Sock Shop, or Train Ticket, and record the system's
telemetry before and after: metrics in every case, and logs, traces and derived series in RE2 and
RE3.

The connector reads the nine zips of the [Zenodo record 14590730][zenodo]
(DOI `10.5281/zenodo.14590730`), the files the authors' `main.py` downloads and scores. Each zip
is pinned by its SHA-256. The record's license is CC-BY-4.0. The authors' README states that
their datasets are MIT; the card keeps the record's own, stricter, license. The authors' code is
[phamquiluan/RCAEval at `fb9abb0`][code].

[paper]: https://doi.org/10.1145/3701716.3715290
[arxiv]: https://arxiv.org/abs/2412.17015
[zenodo]: https://zenodo.org/records/14590730
[code]: https://github.com/phamquiluan/RCAEval/tree/fb9abb09ff571070f92fc2467f5e8b4fc6e7d726

## The source data

A zip holds one suite on one system. A case is a directory
`<suite>/<service>_<fault>/<repetition>/`, so its path names its ground truth: the root cause
service and the injected fault.

| Suite | Cases | Faults | Metric columns | Logs | Traces | Derived series |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| RE1-OB | 125 | cpu, mem, disk, delay, loss | 49–59 | — | — | — |
| RE1-SS | 125 | cpu, mem, disk, delay, loss | 57–63 | — | — | — |
| RE1-TT | 125 | cpu, mem, disk, delay, loss | 198–238 | — | — | — |
| RE2-OB | 90 | the RE1 faults and socket | 69–77 | 15,053,223 lines | 34,461,235 spans | `logts`, `tracets_err`, `tracets_lat` |
| RE2-SS | 90 | the RE1 faults and socket | 74–82 | 7,566,220 lines | — | `logts` |
| RE2-TT | 90 | the RE1 faults and socket | 340–376 | 21,291,651 lines | 67,345,051 spans | `logts`, `tracets_err`, `tracets_lat` |
| RE3-OB | 30 | f1, f2, f3, f4, f5 | 68–101 | 2,095,206 lines | 4,755,090 spans | — |
| RE3-SS | 30 | f1, f2, f3, f4 | 80–107 | 2,298,238 lines | — | — |
| RE3-TT | 30 | f1, f2, f3, f4 | 294–322 | 1,627,863 lines | 4,152,676 spans | — |

The paper names the faults (section 3.2): CPU hog, memory leak, disk stress, socket stress, network
delay and packet loss, and five code-level bugs: incorrect parameter values (f1), missing
parameters (f2), a missing function call (f3), incorrect return values (f4), and a missing
exception handler (f5). RE3 is not a uniform grid: f5 occurs only in RE3-OB, and a service has
between 3 and 6 repetitions of a fault.

A case's files, and what the connector does with each:

| File | Content | Use |
| --- | --- | --- |
| `inject_time.txt` | The fault injection time, Unix seconds | The `inject_time` annotation |
| `data.csv` (RE1-OB), `simple_data.csv` (RE1-SS, RE1-TT), `simple_metrics.csv` (RE2, RE3) | `time` in Unix seconds, then one `<service>_<metric>` column per series, one row per second | The `metrics` Source |
| `logs.csv` | One row per log line: `timestamp` in Unix nanoseconds, `time` as `HH:MM`, `container_name`, `message`, and per suite `level`, `req_path`, `error`, `cluster_id`, `log_template`, or `pod_name`, `node_name` | The `logs` Source |
| `traces.csv` | One row per span: `traceID`, `spanID`, `parentSpanID`, `serviceName`, `methodName`, `operationName`, `startTime` in Unix microseconds, `startTimeMillis`, `duration`, `statusCode`, and `time` as `HH:MM` | The `traces` Source |
| `logts.csv`, `tracets_err.csv`, `tracets_lat.csv` (RE2 only) | The authors' series derived from the logs and traces, one row every 15 s | One Source each |
| `root_cause.txt` (8 RE3-SS cases) | The root cause log line | The `root_cause_log` task annotation |
| `data.csv` (RE1-SS, RE1-TT), `metrics.csv` (RE2, RE3) | The raw cAdvisor export, 437 to over 1,200 columns | Not read |
| `metrics.png`, `metrics_postprocess.log`, `cluster_info.json`, `pod-node-*.csv` | A plot, the authors' preprocessing log, the log template clusters, and pod placement | Not read |

The metrics files are the ones the paper counts (Table 2: 49 to 376 metrics per case) and the
authors' baselines rank. For RE1-SS and RE1-TT, `main.py`'s glob `**/data.csv` picks the raw
`data.csv`, whose cAdvisor names, such as `carts_container-cpu-system-seconds-total`, hold none of
the indicators the authors score, such as `carts_cpu`. The connector reads `simple_data.csv`, named
like every other suite's metrics.

The release states no units. The metrics' values suggest CPU in percent and memory in bytes, but
nothing says so, so every Signal's unit is unknown.

### What the files get wrong, and what the connector does about it

Every item is pinned in `release.py`, so a zip that differs fails the build.

- **Metric rows without a time.** Eight cases end with rows whose `time` is empty but whose
  metrics hold values. `main.py` forward-fills `time` into them, so they enter its window at the
  last timestamp, repeated. Nothing in the file says when they were measured, so the connector
  leaves them out (`release.PINNED_UNTIMED_ROWS`):

  | Case | Rows | Values left out |
  | --- | ---: | ---: |
  | RE1-OB/adservice_cpu/5 | 8 | 272 |
  | RE1-OB/checkoutservice_cpu/4 | 262 | 4,033 |
  | RE1-OB/productcatalogservice_cpu/4 | 14 | 476 |
  | RE1-SS/catalogue_disk/3 | 74 | 1,628 |
  | RE1-SS/user_mem/1 | 114 | 2,203 |
  | RE1-TT/ts-order-service_mem/3 | 113 | 9,209 |
  | RE2-OB/checkoutservice_cpu/2 | 256 | 5,716 |
  | RE2-OB/checkoutservice_mem/2 | 258 | 5,585 |

- **Two tables in one file.** 50 RE1-OB `data.csv` files repeat the `time` header: the four
  `<service>_error` columns after the second `time` (read as `time.1`, as pandas names it) are a
  separate table with their own clock. Where both clocks are set they agree, except in
  `productcatalogservice_cpu/3`, whose error table holds 63 rows timed 582 s after the first row of
  the main table. Each column takes the clock to its left, and the metrics Source's `clocks`
  metadata lists which. `main.py` drops `time.1` and so reads the error columns against `time`.
- **Another system's metrics.** `RE1-OB/productcatalogservice_cpu/3` also holds Sock Shop
  services (`carts`, `catalogue`, `orders`, `front-end`) in both tables, next to the Online
  Boutique ones. They are kept as published.
- **A directory that is not a case.** `RE2-OB/checkoutservice_cpu/multi-source-data/` is the
  authors' multi-source demo: a copy of `checkoutservice_cpu/1` with the same inject time, logs
  and traces, and no `simple_metrics.csv`. `main.py` skips it, and so does the connector.
- **A second fault directory.** RE3-TT holds `ts-route-service_f3_1/{1,2,3}` next to
  `ts-route-service_f3/{1,2,3}`: three more cases, with their own inject times. The connector reads
  the service and fault from the part before the suffix and keeps the path as published. `main.py`
  can't read these: it splits the directory name on `_` into two names.
- **Header-only traces.** 13 RE3-SS cases hold a `traces.csv` with no row. Sock Shop is not
  traced, so they have no `traces` Source.
- **Repeated span ids.** Two RE2-OB cases (`recommendationservice_loss/1` and `/2`) repeat 48 and
  99 span ids. The spans are kept as published.

The authors' README states that RE3 holds `logts.csv`, `tracets_err.csv` and `tracets_lat.csv`;
the RE3 zips don't.

### The Hugging Face copy

The authors also publish a Parquet copy, [`phamquiluan/RCAEval`][hf] on Hugging Face, which their
README recommends for downloading. `main.py` doesn't read it, and it differs from the Zenodo files
it was converted from:

- `re1ob_currencyservice_loss_1` has the inject time `16933142`; the Zenodo file has `1693314223`.
- `re1ob_productcatalogservice_cpu_3` has only the 63 times of its error table, with the main
  table's values at them; the Zenodo file's main table has 4,201 rows.
- `re2tt_ts-auth-service_cpu_1` has no logs; the Zenodo file has 279,630 lines.
- It holds no derived series and no raw metrics, keeps log times in seconds rather than
  nanoseconds, uses `time.1` for the RE1-OB rows without a `time`, and numbers the second RE3-TT
  fault directory's cases as repetitions 4 to 6.

Elsewhere its metric values equal the Zenodo files' to a relative 1e-12 on the suites compared
(RE1-OB and RE3-OB), the precision a CSV round trip prints.

[hf]: https://huggingface.co/datasets/phamquiluan/RCAEval

## The TimeNet dataset

### Records

Each case becomes one Record. Its origin, `start_time`, is the earliest time in any of its files,
so every offset is non-negative; the derived series and logs start up to 14 s before the
metrics. The Record has one Source per file:

| Source | Signals | Time axis |
| --- | --- | --- |
| `metrics` | One float64 Signal per column, named as published, such as `checkoutservice_cpu` | Regular, 1 s, from each column's clock |
| `logs` | One text Signal per column but `timestamp`, and `timestamp_sub_us_ns` | Irregular: each line's `timestamp`, rounded down to the microsecond |
| `traces` | One Signal per column but `startTime`: text, or float64 for `startTimeMillis`, `duration` and `statusCode`; and `source_row` | Irregular: each span's `startTime` |
| `logts`, `tracets_err`, `tracets_lat` | One float64 Signal per column, named as published | Regular, 15 s |

Values are as published:

- Missing values stay missing. An empty CSV field is a null; `main.py`'s forward-fill and
  `fillna(0)` are its own preprocessing and are not applied.
- A text field is null only when it is empty, so a log line `NA` stays `NA`.
- Integers become float64 only when every value is below 2^53, which float64 holds exactly; the
  build fails otherwise. The axis keeps a log line's time to the microsecond, and
  `timestamp_sub_us_ns` keeps the 0 to 999 ns past it, so the published nanoseconds survive.
- Spans are not in start-time order in `traces.csv`. They are stably sorted by `startTime`, the
  order a time axis needs, and `source_row` keeps each span's row in the file.

Each Source's metadata names its `file` and lists its `column_order`, since TimeF doesn't keep
Signal order on read.

`Record.metadata` is `records.RCAEvalRecordMetadata`: the `suite`, such as `RE2-OB`, and the shared
`subject_id`, the system, such as `Online Boutique`. The case's path names its root cause, so it is
task metadata, not Record metadata. `records.record_metadata(record)` reads the metadata back.

Two Record annotations are the inputs every method in `main.py` gets besides the telemetry:

| Annotation | Value |
| --- | --- |
| `inject_time` | The inject time in Unix seconds, as published, with a point span at it |
| `system` | The system, such as `Online Boutique` |

### Tasks

The paper scores RCA at two levels (section 4.2): coarse-grained, the root cause service, and
fine-grained, the root cause indicator, "the metric specifying resource usage" for a resource fault
(section 3.2). A method returns a ranking; `main.py` scores it with AC@k and Avg@k. Each level is a
`ClassificationTask` per case, whose target is the answer a ranking is scored against.

The loader gives each level by its own method: `service_tasks()` and `indicator_tasks()`.
`tasks()` keeps the case order, in which a case's indicator task follows its service task.

| Field | Service task | Indicator task |
| --- | --- | --- |
| `inputs` | The case's Record | The case's Record |
| `scope` | `main.py`'s window, below | The same |
| `input_annotations` | `inject_time`, `system` | `inject_time`, `system` |
| `targets` | The root cause service, such as `checkoutservice` | The indicator's metric Signal, such as `checkoutservice_cpu` |
| `annotations` | `fault`, such as `cpu`, with the paper's name as description; `root_cause_log` where published | The same |
| `metadata` | `case`, the case's path, such as `RE2-OB/checkoutservice_cpu/1` | `case`, and `authors_indicator`, the indicator as `main.py` names it |
| `split` | None | None |
| `prompt` | None | None |

All 735 cases have a service task. The 645 cases of RE1 and RE2 have an indicator task: `main.py`
scores no indicator for the code-level faults. The paper defines no split, and the benchmark asks
no question in words, so tasks have neither a split nor a prompt.

**Window.** `main.py --length 20`, the default the paper's scores use, gives a method the last 600
metric rows before the inject time and the first 600 from it. The metrics are gap-free at one row
per second, which the build checks, so the scope is `[inject - 600 s, inject + 600 s)`, cut at the
first and last metric row. The multi-source methods take 40 rows of the 15 s derived series on
each side, the same 600 s. The whole case stays in the Record.

**Indicator.** `main.py` scores a ranked `<service>_<metric>` against the case's service and a
metric that depends on the fault:

| Fault | Indicator metric |
| --- | --- |
| cpu, mem, socket | `cpu`, `mem`, `socket` |
| delay, loss | `latency`: `main.py` renames `<service>_latency-90` to `<service>_latency` and drops `_latency-50` |
| disk | `diskio` in RE2; `latency` in RE1, which has no disk metric |
| f1 to f5 | None |

The target is the Signal under its published name, such as `checkoutservice_latency-90`, and
`authors_indicator` is the name `main.py` compares, such as `checkoutservice_latency`. The paper
names a different indicator for loss: "the metric showing failed requests and/or error response
codes". The connector follows the code, which produced the paper's scores.

**Fault.** `fault` is a task annotation, not an input: it would reveal the indicator. The authors
report their scores per fault, so it is kept for grouping.

### Dataset annotations

| Annotation | Value |
| --- | --- |
| `source_url` | `https://zenodo.org/records/14590730` |
| `source_doi` | `10.5281/zenodo.14590730` |

### How the connector follows the authors

| Authors' setup | Connector choice |
| --- | --- |
| `main.py` downloads the Zenodo zips | The connector reads the same zips, pinned by digest |
| A method gets the case's telemetry, its inject time, and the dataset's name | The Record, and the `inject_time` and `system` input annotations |
| A method sees 600 s on each side of the inject time | The task's scope |
| The coarse-grained answer is the service in the case's path | The service task's target |
| The fine-grained answer is `<service>_<metric>`, by fault | The indicator task's target, the metric's Signal |
| Scores are reported per fault | The `fault` task annotation |
| `main.py` forward-fills and zero-fills missing values | Not applied: missing values stay null |

Three parts of `main.py` are not reproduced:

- It passes every method an SLI, a metric to watch, chosen as `<service>_latency` of the root
  cause service when that column exists. That passes the ground truth to the method, so the dataset has
  no SLI.
- It drops the `_latency-50` columns before a method runs. The Records keep them; a method that
  wants `main.py`'s columns drops them itself.
- It scores the coarse level by the part of a ranked name before the first `_`, without a `-db`
  suffix, so `orders-db_cpu` counts as `orders`, and drops repeated services from the ranking. A
  score that wants to match the paper's has to do the same.

TimeNet ships no metric: evaluation is out of its scope.
