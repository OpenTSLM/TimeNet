# Time-MMD (Energy and Environment)

[Time-MMD](https://github.com/AdityaLab/Time-MMD) pairs a government time series of each of nine
domains with text the authors collected for it and summarized with Llama3-70B. This connector
converts two domains at commit `00281e2d`:

| Domain | Series | Cadence | Rows | Span |
| --- | --- | --- | --- | --- |
| Energy | U.S. retail gasoline prices (EIA), national average plus eight regions | weekly | 1,622 | 1993-04-05 to 2024-04-29 |
| Environment | Daily AQI of the New York CBSA (EPA), with category, defining pollutant, defining site and site count | daily | 15,979 | 1980-01-01 to 2023-09-30 |

## Records

One record per domain. The numerical columns are its signals on a `RegularAxis` of one week or
one day. The forecast target of the paper keeps its upstream name, `OT`; the dataset annotation
`target_signal` names it. The record's `metadata` names its domain and its three source files.
Static record annotations hold the columns that are constant through a file: for Environment the
`cbsa` and `cbsa_code`.

The record's origin is the earliest date any of the domain's three files states, which is a text
row from 1979 for both domains, and its `time_span` reaches to the latest. The signals are offset
from that origin by however much text precedes them.

## Text

Each row of `<Domain>_report.csv` and `<Domain>_search.csv` becomes interval annotations spanning
the days the row covers. The fact cell is one annotation, `report_fact` or `search_fact`. The
prediction cell holds a long-term prediction, a `;`, then a short-term one, and each half is its
own annotation: `report_prediction_long`, `report_prediction_short`, and the `search_` pair.
Each occurrence carries the release's row index as `source_row`, so the halves of one cell share
it. Which cells yield no annotation is settled under modelling assumptions.

## Forecasting tasks

Each task is one window of `lookback + horizon` steps. Its `scope` is the context interval over
every signal of the record, its target the horizon interval scoped to `OT`, and its
`input_annotations` the text annotations that fall in its context window. The
`target_annotations` stay empty: the answer is the numerical horizon alone.

| Domain | Lookback | Horizons |
| --- | --- | --- |
| Energy | 36 weeks | 12, 24, 36, 48 |
| Environment | 96 days | 48, 96, 192, 336 |

Task metadata states `context_steps`, `horizon_steps`, and `first_context_step`. The tasks are
streamed, so a build never holds them all in memory.

## Modelling assumptions

The release leaves several things unstated. This connector settles each one as follows, and a
consumer who disagrees should know which choice to undo.

**Time**

- The release dates its rows by calendar day with no time zone. Every date is read as midnight
  UTC, so the same date lands on the same microsecond in every build.
- A row that states a start and an end date covers those days inclusive, which becomes the
  half-open interval from the midnight starting the first day to the midnight ending the last.
- The weekly cadence is taken from the row dates themselves: each row must start on the date it is
  labelled with and end six days later. A row that skips or repeats a period fails the build
  rather than being filled.
- Text that predates the numerical series is kept; it sits before the signals on the record
  timeline.

**Values**

- Every numerical cell is present in the release, so no signal is nullable.
- Prices are stored as float32 and the AQI and site count as int16, both wide enough for the
  observed ranges.
- The AQI category and defining pollutant are enums with the EPA's fixed codebooks, and a value
  outside them fails the build.
- The AQI and the site count are dimensionless.
- The gasoline price states no unit because the registry defines no currency; the spec's name says
  "US dollars per gallon".

**Text**

- The facts and predictions are the output of Llama3-70B over the authors' sources, not the
  sources themselves, and are stored as the release gives them without cleaning.
- A cell or half that states nothing gets no annotation: an empty one, or one the release filled
  with `NA`.
- A prediction cell is read as long-term, `;`, short-term. A cell with no `;` does not say which
  half it holds, so it is dropped rather than guessed from its wording; the release has two such
  cells.
- A fact that contains a semicolon is one annotation.
- Rows whose spans repeat, as in the Environment report, each stay a separate annotation.

**Tasks**

- The windows follow the protocol of MM-TSFlib, the paper's library: a chronological 70/10/20
  split, a window slid with stride one over each stretch, and validation and test stretches that
  start `lookback` steps early. This is the paper's evaluation protocol, not a partition the data
  itself carries.
- Only `OT` is a target, the one column the paper forecasts; the other columns are context.
- A task sees every text annotation that overlaps its context interval and ends at or before the
  forecast boundary, including one that started before the context did.
- A span that runs past the boundary tells about the horizon, so it is withheld in full rather
  than cut, even where it starts inside the context.
- A text annotation can serve as input to tasks in more than one split, because the stretches
  overlap by one lookback as the protocol dictates.
