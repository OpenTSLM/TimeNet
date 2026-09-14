# Time-MMD: what the release does not state, and the decisions of this connector

Time-MMD pairs a numerical series of each of nine domains with text over stated days. The
release states the days and the texts, and little else. Its files name no unit, define no
column, carry no license and date no publication. Each gap forces a decision, and this document
states the decisions this connector takes for the two domains it covers, `energy` (weekly) and
`environment` (daily).

*(measured)* marks a number that this connector took from the release itself, or from the
agency that publishes the series, and not from the paper or the README of the release.

## The source of truth

Two parts of the release, or the release and its source, sometimes state one fact differently.
This connector follows one rule:

**This connector replaces no stated fact. Where two sources state the same fact differently, one
source wins for a named reason, and the connector keeps the other beside it.**

| Fact | Source of truth | The other reading |
| --- | --- | --- |
| what `OT` holds | the agency's own series, matched value by value | the README figure of the release, which names a domain and not a series |
| the days a text covers | the `start_date` and `end_date` of its row | the date the text quotes, kept inside the text |
| the cadence of a series | the `date` column, checked row by row | `start_date` and `end_date`, checked to agree with it |
| the license | Appendix V of the paper, which states ODC-By v1.0 | the repository, which ships no LICENSE file, and the arXiv page, whose CC BY 4.0 covers the paper's text |
| the bytes of the release | one commit of the repository, with a digest for each file | the agencies, which revise what they publish |

## What becomes a signal, an annotation and a task

**A column that changes from row to row becomes a signal.** The energy file holds the weekly US
retail gasoline price, which the release names `OT`, and the eight regional prices EIA
publishes beside it. All nine become signals of one spec. The environment file holds the daily
AQI of one core-based statistical area, again named `OT`, and the four columns EPA states
beside it. Each becomes a signal of its own spec, typed as the column is: the index and the
site count as `int16`, the category and the defining pollutant as an `enum` over the codebook
EPA uses, the defining site as `str`. A category is not a number and a site id is not a
category, and the type says so.

**A column that holds one value on every row becomes an annotation with no span.** The two
CBSA columns of the environment file name the New York area on all 15 979 rows *(measured)*.
They describe the record and not a day.

**Each text becomes an annotation over the days its row states.** A row of a textual file
states a `fact` and a `preds` field. Each becomes one annotation, keyed by the file it came from
(`report_fact`, `report_prediction`, `search_fact`, `search_prediction`), over the half-open
interval from the midnight that begins `start_date` to the midnight that follows `end_date`. An
annotation names no series, because a text is about the domain and not about one column.

**Two facts about provenance become annotations with no span.** `target_signal` names the
signal that holds `OT`, and `source_columns` maps every signal to the header it was read from,
so the long EIA headers survive the short signal names.

**Two questions become tasks.** The forecasting question the release was built for, and the
caption question its reports answer. The section after next states them.

## One record for each domain, on one timeline

A record is one domain. It carries the whole series, and every task is stated with a scope on
it, so nothing cuts a window out of the raw recording.

**The record's zero is a midnight in UTC.** The release states dates and no time of day, and
names no zone. At a cadence of a day or a week, another zone moves every value by the same few
hours and reorders nothing. The zone is a convention of this connector and the card says so.

**The zero sits on the cadence of the values, at or before the earliest text.** The search text
of the energy domain begins on 1979-12-31, 692 whole weeks before the first price of
1993-04-05 *(measured)*. A TimeF span starts at zero and holds no moment before it, so the
record's zero moves back to that Monday, and the weekly axis of the prices starts at index 692.
The environment domain moves back one day, from 1980-01-01 to the search week that starts on
1979-12-31 *(measured)*.

**The session reaches to the later of the last value and the last text.** Both records declare
a `time_span` that ends on 2024-05-06, the day after the last search week *(measured)*. The
prices end on 2024-04-29 and the AQI on 2023-09-30, so text after those dates has a place in
the record and no value beneath it. The consequences are stated below.

**The two cadences meet in microseconds.** A weekly axis has a period of seven days and a daily
axis a period of one, both from the same kind of zero. A text span reads the same way on
either. That is how one schema holds a weekly and a daily domain.

## The tasks this connector builds

The release ships numbers and text. It ships no task. What those become is a decision, and this
section states the decisions this connector takes. A build gives 13 785 tasks over 2 records
*(measured)*.

| the question | type | count | scope |
| --- | --- | --- | --- |
| what are the next 12, 24, 36 or 48 weekly prices | `ForecastingTask` | 1180 | from the first price to the origin |
| what are the next 48, 96, 192 or 336 daily AQI values | `ForecastingTask` | 12 112 | from the first AQI day to the origin |
| what happened to the price over the days a report covers | `AnswerTask`, a caption | 354 | from the first price to the end of those days |
| what happened to the air over the days a report covers | `AnswerTask`, a caption | 139 | from the first AQI day to the end of those days |

### The forecasts

**The protocol is the one the paper evaluates.** Time-MMD splits a series 70/10/20 in time
order, as the Time-Series-Library convention its MM-TSFlib extends does, and slides a window
one value at a time through the test part. The paper states four horizons for each cadence:
12, 24, 36 and 48 values for a weekly series and 48, 96, 192 and 336 for a daily one (Appendix
K.4). This connector builds one task for each origin of the test part and each horizon that
still fits before the last value. The test part begins at value `n - n // 5`, which is what the
library's `int(n * 0.2)` gives. That is 313, 301, 289 and 277 tasks for the four weekly
horizons and 3148, 3100, 3004 and 2860 for the four daily ones *(measured)*.

**The target is the target series alone.** The release names `OT` as the default target, so a
task's `target_span` names that one series. The other series of the record are context. A
consumer who wants the regional prices forecast too widens the span.

**The context is everything the record holds from the first value to the origin.** A scope from
the first value of the target series to the origin covers every series and every text in that
stretch, and `step_range` resolves it to the steps of the series. A scope from the record's zero
would name steps the series does not have, so the text before the first value lies outside
every scope, and stays in the record. The paper fixes a lookback window of 36 weeks or 96 days,
and MM-TSFlib admits a text when its end date is earlier than the end date of the input window.
Both are decisions for whoever trains, and this connector does not cut them into the task. A
consumer crops the scope and selects the text by span.

**No forecast names its text.** `input_annotation_ids` stays empty, because a task that named
every annotation before its origin would carry thousands of ids, and one that named a chosen
few would fix the lookback this connector leaves open.

### The captions

**A report fact is a caption of the days it covers.** The task carries no prompt, so it is a
caption: given the record from the first value to the end of the report's days, state what the
report states.
Its answer is the `report_fact` annotation itself, by reference. That is the "explain what is
happening, and since when" question, asked with the history the reporter had.

**A search fact is context and not a caption.** The release's own statistics put the relevance
of search text at about 17 percent against 84 percent for reports (Table 2 of the paper), and
the search facts of the environment domain state a Caltech grant or a NASA launch as readily as
the air of New York. A caption whose target is not about the series is a wrong training
target. The search facts stay annotations, and a consumer who wants them as targets has them.

**A report past the last value gets no caption.** One environment report fact lies after the
last AQI day *(measured)*. Nothing can ask a record to describe values it does not hold. The
annotation stays.

### What becomes no task

**A prediction is context and never a target.** The `preds` fields are the outlook a language
model wrote from the source text, for the long and the short term. They are what the model
guessed and not what happened. As targets they would teach a model to guess like another
model. They stay annotations.

**The category is not a task.** It is a function of the AQI: EPA's breakpoints reproduce the
stated category on all 15 979 days *(measured)*. A task that asked for it would ask a model to
apply a lookup table.

**Provenance does not become a task.** `target_signal`, `source_columns` and the two CBSA
facts state where the record came from, and a question about them asks a model to recover a
fact the dataset states beside it.

### How the connector writes them

**The tasks stream, and the connector holds them nowhere as a list.** `set_task_stream` exists
for a dataset with far more tasks than records, and this one has thousands for each of its two.
Streamed tasks carry their own `record_ids`, and none appears in `Record.task_ids`.

**The stream reads no file.** It walks the series and the annotations the records already
carry. A second read of the release could let the tasks and the annotations disagree.

**Every id is derived, and two builds of one commit give one set of ids.** A record is named
after its domain, a series after its signal, an annotation after its file, its field and the
position of its row, a task after its origin and horizon or after the annotation it asks for.

**The connector invents no prompt.** The release states no question in words.

## Inconsistencies

### The energy reports are filed a week before the price they quote — **Handled**

**Problem.** An energy report row covers a Monday to a Friday, and its fact reads like "the
national average retail regular gasoline price increased to $3.258 per gallon on December 26,
2011". The row that holds it states the days 2011-12-19 to 2011-12-23. Of the 351 report facts
that quote a date, 349 quote the Monday one week after their row's `start_date` *(measured)*.
That Monday is the next value of the series. A report of week *w* thus states the price of
week *w + 1*, and could not have been written before it. The regular-grade price the reports
quote is also not the all-grades price `OT` holds: on 2011-12-26 the release holds 3.317 and the
report quotes 3.258 *(measured)*, so the quote leaks the direction and not the value.

**Decision.** The connector keeps every span as the release states it. To move a report by a
week is to re-date what the authors verified by hand, and this connector does not correct the
release.

**Consequence.** A consumer who hands a forecaster the reports inside its context window hands
it a text that quotes the first value of the horizon, in the last week of every window. The
end-date rule MM-TSFlib applies does not exclude it, because the report's end date precedes
the input's. A consumer that reads the energy reports as context must drop the last context
week's report or accept the leak. The energy captions ask the same question one step ahead:
their scope ends where the report's days end, and the price the report quotes is the value
after it, so a model that answers one must extrapolate as the reporter did.

### A text can state that nothing was found — **Handled**

**Problem.** The prompt that produced the texts told the model to write `NA` for a part it
could find nothing for (Appendix F of the paper). The release writes it in several forms: `NA`
alone, `NA (no relevant information found)`, `NA` followed by a note on its own lines, and an
empty field. The environment search file holds 80 such facts and the energy search file 13,
plus 2 empty ones; the environment reports hold 6 and 10 empty *(measured)*. A `preds` field
reads `NA;NA` on 613 energy and 370 environment search rows and on 28 environment report rows,
and is empty on 5, 3 and 10 more *(measured)*.

**Decision.** A field that is empty, or that starts with `NA` or `N/A` as a word, states that
nothing was found, and it becomes no annotation. A fact that starts with the same letters
inside a word, such as `NASA`, states something and stays. Two environment search facts state
their emptiness in prose, as "None of the search results contain objective facts", and stay,
because the connector matches the token the prompt defined and does not read the prose
*(measured: 2)*.

**Consequence.** A build carries 2299 search facts and 1696 search predictions for the energy
domain, and 2232 and 1939 for the environment domain, against 2314 and 2312 rows *(measured)*.
A week with no annotation is a week where the model found nothing, and not a week the release
lacks. A consumer who counts annotations by week finds gaps that mean that.

### A prediction comes as one field with two parts — **Handled**

**Problem.** The prompt asked for a short-term and a long-term prediction as two parts. The
release joins them with a semicolon into one `preds` field and states the order nowhere. In
every energy report, and in every row where both parts name their horizon, the long-term part
comes first *(measured: 354 of 354 energy reports)*. On 133 energy and 164 environment search
rows, and 11 environment report rows, one part reads `NA` and the other states something. Two
rows hold one part and no semicolon *(measured)*.

**Decision.** The connector keeps the field whole, as the release writes it. A field whose
every part states nothing becomes no annotation. A field with one stated part is kept whole,
`NA` included, because the position of the `NA` says which horizon the model had nothing for.

**Consequence.** A consumer who wants the two horizons apart splits on the semicolon and reads
the long-term part first. A text that itself holds a semicolon would split into more than two
parts, and none of the release's rows does *(measured)*.

### The text reaches outside the values — **Handled**

**Problem.** The energy search text begins in the week of 1979-12-31, and 692 of its 2314 weeks
end before the first price of 1993-04-05 *(measured)*. The environment search text runs to
2024-05-05, and 31 of its 2312 weeks begin after the last AQI day, with 2 more that straddle
the edges of the series; 5 of the 156 environment reports are dated after that day too
*(measured)*.

**Decision.** The connector keeps every text in the record. The record's zero moves back to
cover the earliest text, and its declared session reaches the latest, so every span lies
inside the record. To drop what the release pairs with the domain is a preprocessing decision,
and this connector does not make it.

**Consequence.** A span with no value beneath it is valid, because the record declares a
session and TimeF checks an unscoped span against that. A consumer that reads an annotation and
then the values under it must check the window of the series first. The caption tasks do, and
one report gets none for it. A consumer who wants only the text of the covered years selects by
span against the series' window.

### The release is a snapshot of sources that revise — **Handled**

**Problem.** EPA republishes its daily AQI files as late data arrives. The 2023 file EPA served
on 2026-09-12 agrees with the release on all five fields for 43 of the 273 days of 2023 the
release holds, and the release stops on 2023-09-30 where EPA's file runs to the end of the year
*(measured)*. The EIA series agreed on every week checked: 1993-04-05, 1993-04-12, 1993-04-19,
2011-12-26 and 2024-04-29 *(measured)*.

**Decision.** The connector fetches the six files from one commit of the repository,
`00281e2d86058286d5548b15a7670e8eda57ef62`, and checks each against the SHA-256 recorded in
`connector.py`. A download that gives other bytes stops the build.

**Consequence.** Version 1.0.0 of this dataset is those bytes and no others, and two builds
agree to the value. A newer snapshot of the release, or the agencies' revised values, is a new
version with new digests, and never a silent change to this one.

### The files do not say what `OT` holds — **Handled**

**Problem.** Every numerical file names its target `OT`, and the release defines it in a figure
of its README as "Gasoline Prices" and "Air Quality Index". The energy file's other columns
carry full EIA headers and `OT` carries none. The paper's appendix documents no source for the
environment domain at all, and its overview table names a different agency than the one whose
file layout the release carries.

**Decision.** The connector states what each `OT` is from the agency's own data. The energy
`OT` matches EIA's "U.S. All Grades All Formulations Retail Gasoline Prices" on every week
checked, and not the regular-grade series the release's own reports quote *(measured)*. The
environment file carries the eight columns of EPA AirData's "daily AQI by CBSA" files, in EPA's
order and spelling, for CBSA 35620 *(measured)*. The spec names and the data sources say so,
and `target_signal` names the signal that holds `OT`.

**Consequence.** A consumer reads what the series is from the manifest, and not from a figure.
If the identification is wrong, one spec name and one data source are the place to change.

### A price has no unit pint can carry — **Handled**

**Problem.** A gasoline price is a currency per volume, and pint's registry, which TimeF units
go through, defines no currency. A unit the SDK's registry cannot parse would break every read
of the manifest, because the reader rebuilds a spec's unit from its name.

**Decision.** The price spec declares the dimensionless unit and states the unit in words in its
name. A currency in the shared registry is a change to the core package, and this connector does
not make it.

**Consequence.** A consumer who converts prices does it outside pint. A later core change that
defines currency units can give this spec a real unit in a new version.

### Several reports fall on one day — **Handled**

**Problem.** The 156 environment reports cover 83 distinct days, and 22 days carry more than
one, up to 21 on 2023-06-08, the day after the Canadian wildfire smoke reached New York
*(measured)*. Five rows repeat another row's days and text *(measured)*.

**Decision.** Every row becomes its own annotations, named after its position in the file, so
two reports of one day are two annotations and a repeated row is two. The connector removes
nothing the release states twice.

**Consequence.** A day with 21 report facts gets 21 caption tasks with one scope. A consumer
that wants one caption for a day, or one copy of a repeated text, dedupes by span and value.

### The release ships no license file — **Handled**

**Problem.** The repository holds no LICENSE, and its README names none. The arXiv page of the
paper shows "CC BY 4.0", which is the license of the paper's text and not of the data. The
paper's own Appendix V states the license of the dataset.

**Decision.** The card records ODC-By-1.0, as Appendix V states it: "We license our dataset by
ODC-By v1.0". The same appendix adds that the sources are "in public domain and authorized for
non-commercial distribution", and the card's citation points at the paper so a consumer can read
that sentence. The numbers come from two US federal agencies, which publish them as public
information.

**Consequence.** A consumer who redistributes the dataset attributes it, as ODC-By asks, and reads
Appendix V before a commercial use. Should the repository state another license, the card is the
one place to change.

## Decisions a consumer may revisit

Each decision above sits in one place, so a consumer who takes another one changes one thing.

| Decision | Where |
| --- | --- |
| the test share, the horizons, the stride and the target-only span of the forecasts | `TEST_SHARE` and `iter_forecasting_tasks` in `tasks.py`; `horizons` on each shape in `domains.py` |
| which facts get a caption | the key filter in `iter_caption_tasks` in `tasks.py` |
| whether text outside the values stays | `Timeline.covering` in `timeline.py` and the session end in `connector.py` |
| what counts as nothing found | `states_nothing` in `text.py` |
| whether a prediction is split into its two parts | `build_annotations` in `text.py` |
| which columns are signals, constants or ignored | the shapes in `domains.py` |

## Reproducing the build

```bash
uv run timenet-build build adityalab/time-mmd
```

The build fetches about 4.4 MB from GitHub and writes about 0.6 MB. A clone of the repository has
the layout the connector reads, so `TimeMmdConnector().convert([TimeMmdSource(root=<clone>)])`
builds the same dataset with no download. `examples/load_time_mmd.py` reads one forecast and one
caption of each record back through their spans.

```python
from timenet.client import TimeNet

dataset = TimeNet().load("adityalab/time-mmd")
dataset.describe()
```
