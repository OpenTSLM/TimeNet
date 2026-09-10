# VerbalTS

The six caption-paired corpora released with VerbalTS, a text-to-series generation benchmark. One
Google Drive folder holds 60 files: 54 NPY arrays and 6 `meta.json`. One record is one window, with
every signal of that window, its categorical attribute codes, and one or three captions that
describe it. Each caption becomes the prompt of a generation task whose answer is the window.

- **id**: `seqml/verbalts`
- **source**: https://drive.google.com/drive/folders/1N0zxkLdvpdjkwayKA2OZIJYP4nfzhOeF
- **paper**: https://proceedings.mlr.press/v267/gu25a.html, PMLR 267:20448-20476
- **code**: https://github.com/seqml/VerbalTS
- **licence**: `other`, because the six components hold six different positions. See
  "The six components hold six licence positions" below.

A number marked *(measured)* was either derived from the pinned byte table in `files.py`, which
records the exact size of all 60 files from one full download, or counted over a local copy of those
60 files whose SHA-256 all match that table. A reader can re-derive the window count, the step
count, the signal count, the attribute count and the caption width of every split from the 60 byte
counts alone, with no download. The caption count in the last column is the one number those bytes
do not give on their own: the paper states it for `Weather` and the caption planes carry it for the
rest.

| component | windows | steps | signals | attributes | captions per window |
| --- | --- | --- | --- | --- | --- |
| `synthetic_u` | 32,000 | 128 | 1 | 3 | 1 |
| `synthetic_m` | 32,000 | 128 | 2 | 4 | 1 |
| `Weather` | 13,100 | 36 | 21 | 7 | 3 |
| `BlindWays` | 1,029 | 600 | 72 | 2 | 1 |
| `ETTm1` | 16,275 | 120 | 1 | 5 | 1 |
| `istanbul_traffic` | 10,224 | 144 | 1 | 5 | 1 |

That is **104,628 records**, **471,687 series**, **70,069,656 float64 values** and **130,828 tasks**
*(all measured)*. The values plane is 560,557,248 bytes of the release's 799,997,660.

## The source of truth

Four facts have two possible sources. Which one this connector follows, and what the other one
says:

**The unit of a value.** Nothing states one, so every spec is `dimensionless`. The signal names read
as units (`p (mbar)`, `T (degC)`, `SWDR (W/m2)`) and are labels only.

**How the 72 `BlindWays` signals group.** Both papers state 24 joints of three variables each, in
the sentences quoted below. The artifact states no grouping and no axis order.

**What an attribute code means.** `meta.json` states how many codes an attribute takes and nothing
else, for 26 attributes over the six components *(measured)*. The paper's appendix prints the label
lists, and no source states their code order.

**The cadence of a component.** The paper's appendix, quoted below, except for `BlindWays`, where no
source states a rate. The artifact states no cadence and no timestamp at all.

The first and the third go to the artifact, for one reason: a reader can re-measure it and cannot
re-measure a sentence. The cadence is the opposite case and is worth reading twice. The artifact
states nothing at all about time, so a derived fact does not displace a stated one there; it fills a
hole. No cadence in `specs.py` is a reading of these bytes: two are quoted from the paper, `ETTm1`
is arithmetic on two numbers the paper states, and the `BlindWays` rate is an inference. See the two
cadence entries below.

## What the description states

The Drive folder ships no README, no data dictionary and no units file. Every fact this conversion
leans on is below, quoted where a source states it and named as an inference where none does, from
one of three sources a reader can open:

- **the paper**, Gu, Li, Jing and Ren, *VerbalTS: Generating Time Series from Texts*, PMLR
  267:20448-20476, https://proceedings.mlr.press/v267/gu25a.html. Appendix A is the dataset
  appendix.
- **the release's own loader**, `data/data.py` in https://github.com/seqml/VerbalTS.
- **the BlindWays paper**, Kim, Sengupta, Kuribayashi, Kacorri and Ohn-Bar, *Text to Blind Motion*,
  NeurIPS 2024 Datasets and Benchmarks, https://openreview.net/forum?id=QIJQ1qCGqV, which is where
  that component's motion data comes from.

**The component order.** Section 5.1 introduces the six datasets in one sentence: "(i) Full
synthetic datasets ... including Synth-U with univariate time series and Synth-M with multivariate
time series. (ii) Real-world datasets including Weather ... and BlindWays ... (iii) Augmented
real-world datasets including ETTm1 ... and Traffic". `files.COMPONENTS` follows that sentence, so
it decides the enumeration order and the order the writer walks the files in. Table 1 of the same
paper prints Synth-M before Synth-U, because it groups by multivariate and univariate setting; the
table is not the order this connector uses.

**The `Weather` cadence** is stated twice in Appendix A.3.2. "The time series data consist of 21
weather-related variables recorded at 10-minute intervals throughout the year, with timestamps
accurate to the second." And for the window: "The sequence length L is set to 36, representing 6
hours of data (10-minute intervals, 6 samples per hour, resulting in time series length of
6 × 6 = 36)." That decides one step per 600 s and leaves nothing to infer.

**The `istanbul_traffic` cadence** is stated in Appendix A.2.3, where the raw Kaggle series is
minute-level: "The raw dataset, consisting of a single long sequence with a total length of 817,769,
is first taken a sample every ten minutes". That decides one step per 600 s.

**The `ETTm1` cadence** is the one no sentence fixes. Appendix A.2.2 gives the length and the span:
"The raw dataset, consisting of a single long sequence with a total length of 69,680" and "The
dataset covers a two-year period from July 2016 to July 2018." 69,680 points over 730 days is one
point every 15.09 minutes, so the axis takes 900 s. See the cadence entry below.

**The 21 `Weather` names, and their order.** Appendix A.3.2: "The 21 weather-related variables
include: atmospheric pressure (p, mbar); temperature (T, degC); potential temperature (Tpot, K); dew
point temperature (Tdew, degC); relative humidity (rh, %); ... logarithmic temperature (Tlog, degC);
and carbon dioxide concentration (CO2, ppm)." All 21 are listed, in the order
`tables.WEATHER_SIGNALS` stores them, which is also the order of the upstream Jena header the names
are spelled from.

**The `BlindWays` layout.** Appendix A.3.1: "The motion data comprises multivariate data from 1,029
motion segments, recorded through 18 IMU sensors worn by each participant, detailing the positions,
angles, and trajectories of 24 joints in the human body. Each joint is represented and recorded
using three variables, resulting in a total of 72 variables." The BlindWays paper says the same of
its own capture: "The data is captured in the Xsens joint representation, comprising a total of 24
joints." That fixes 24 joints of three variables each. Neither sentence says the three variables are
x, y and z, nor that the three belonging to one joint sit next to each other, so the grouping and
the axis letters in `j00_x` are this connector's positional labels.

**The two `var_id` column orders.** Appendix A.2.2 lists the ETTm1 columns: "Each data point in
ETTm1 consists of 7 numerical features: High Useful Load (HUFL), High Useless Load (HULL), Middle
Useful Load (MUFL), Middle Useless Load (MULL), Low Useful Load (LUFL), Low Useless Load (LULL), and
the target variable Oil Temperature (OT)." Appendix A.2.3 lists the three Traffic columns in the
same way, as the "Traffic index Overall", the "Traffic index of Asian side" and the "Traffic index
of European side", abbreviated `TI`, `TI_An` and `TI_Av`. Appendix A.2.1 fixes which attribute holds
the code: "each time series is represented as a structured vector of indices, referred to as the
attributes index. This vector includes: Variable Index: [0,K-1] where K is the variable number.
Trend Attribute ... Seasonality Attribute ... Skewness Attribute ... Kurtosis Attribute", in the
order `meta.json` lists them, and both files do list `var_id` first. The captions then check the
decoding against the bytes; see the `var_id` entry below.

**Which `meta.json` fields are converted** is decided by the release's own loader, which reads two:
`self.attr_list = self.meta["attr_list"]` and `self.attr_n_ops = np.array(self.meta["attr_n_ops"])`.

**That a caption is not an addressable unit** comes from the same loader, which samples one caption
per access, `cap_id = random.randint(0, len(self.caps[idx])-1)`, and from Appendix A.3.2: "In the
processed dataset, all descriptions are retained, and during training, one description is randomly
sampled for each time series instance."

**That no record has a clock** comes from the loader building its own index,
`self.time_point = np.arange(self.n_steps)`, and from the artifact, which ships no timestamp.

**The task direction** is the paper's own subject: "In this paper, we introduce VerbalTS, a novel
framework for generating time series from unstructured textual descriptions".

Two places where the description and the files disagree. The code repository's README lists the
caption file as `train_caps.npy`, while the release ships `train_text_caps.npy`, which is the name
`data/data.py` itself reads. And Appendix A.1 says the 32,000 synthetic windows split "in a ratio of
6: 1: 1. Finally, we get 24000 training samples, 2400 validation samples, and 2400 test samples",
while the arrays and `meta.json` both hold 24,000 / 4,000 / 4,000, which is what a 6:1:1 split of
32,000 gives. The files win in both cases, and the connector reads only the files.

## What one record holds

One record is one window of one split of one component.

**Series.** One `TimeSeries` per signal of the window, in the order the array stores them. Every
series of a window holds the same number of steps, because the release stores each split as one
`(n_windows, n_steps, n_signals)` array. Values are the shipped float64, unchanged.

Every series is `dimensionless`. VerbalTS z-scored the four real-world components per variable
against their own train split and never published the mean and the standard deviation vectors, so
mbar, degC, kW, traffic index and metres cannot be recovered from these bytes. Appendix A.3.2 states
it for `Weather`, "z-score normalization is applied to each variable using the mean and standard
deviation computed from the training set", and A.2.2, A.2.3 and A.3.1 say the same for `ETTm1`,
`istanbul_traffic` and `BlindWays`. The bytes agree: every `Weather` and `BlindWays` signal has
train mean 0.0000 and train standard deviation 1.0000, and every `var_id` group of `ETTm1` and
`istanbul_traffic` sits within 0.003 of that *(measured)*. Appendix A.1 states no scaling for the
two synthetic sets, whose values are generated rather than measured, and their train splits sit
within 0.011 of the same mean and deviation *(measured)*. The five spec types are `synthetic`
(shared by both synthetic sets), `weather`, `pose`, `power` and `traffic`.

**Signal names** come from four different rules, because the artifact ships no name anywhere:

| component | rule | example |
| --- | --- | --- |
| `Weather` | the 21 Jena column names the paper lists, in that order | `p (mbar)`, `CO2 (ppm)` |
| `BlindWays` | signal `k` is joint `k // 3` on axis `k % 3` | `j00_x` … `j23_z` |
| `ETTm1`, `istanbul_traffic` | the window's own `var_id` code | `MULL`, `TI_Av` |
| `synthetic_u`, `synthetic_m` | position | `x0`, `x1` |

**The time axis** is per component and holds a cadence only. Both synthetic sets take an
`OrdinalAxis`: order recorded, no clock claimed. `Weather` and `istanbul_traffic` take one step per
600 s, `ETTm1` one per 900 s, and `BlindWays` 60 Hz. The first two are stated in the paper, `ETTm1`
follows from the length and the span it states, and the `BlindWays` rate is an inference this file
records as open.

**Annotations**, all of them about the whole record and none scoped to a signal:

| key | value | where it came from |
| --- | --- | --- |
| `component` | e.g. `Weather` | which folder the window was read from |
| `split` | `train`, `valid` or `test` | which array the window was read from |
| `<slug>_<attribute>` | one int64 code per attribute | `{split}_attrs_idx.npy`, by column |
| `<slug>_<attribute>_options` | how many codes it takes | `meta.json`, once per corpus |

The slug carries the component (`weather_season`, `ettm1_season`), and that prefix is load-bearing.
Weather's `season` is one of four calendar seasons and ETTm1's is a code for the dominant
periodicity, of which `meta.json` declares nine. There are 26 attributes over the six components
*(measured)*, so 26 record keys and 26 corpus-level keys. Each attribute keeps the name the release
writes it under, including the truncated `atmospher`, so Weather's atmospheric-pressure attribute
lands under `weather_atmospher`.

**No record carries a `start_time`** and `subject_ids` is empty everywhere. The release states
neither; see the two entries below.

**Record ids are positional**: `verbalts-{component}-{split}-{row:05d}`, for example
`verbalts-Weather-train-00000`. The release ships no id of its own. Two builds of this release give
the same ids. A new release of the dataset invalidates every one of them. A series id is the record
id and the signal name: `verbalts-Weather-train-00000-p (mbar)`.

## The tasks this connector builds

| the question | type | count | scope |
| --- | --- | --- | --- |
| given this caption, generate this window | `TSGenerationTask` | **130,828** | the whole record |

The caption is the `prompt` and the window is the answer, reached through `target_record_id`. Each
task lists its component's `_options` annotations in `input_annotation_ids`: they describe the
attribute space the caption was drawn from, which is context about the input and not part of the
answer.

A Weather window gets three tasks over one target, one per caption. Every other component gets one.
So 104,628 records answer 130,828 tasks. The tasks stream rather than being materialized, so a
record row states no task id of its own; see "The tasks stream out of the caption planes" below.

## Inconsistencies and decisions

### Every value is dimensionless — **Handled**

**Problem.** The components carry physical names and no units. VerbalTS z-scored the four
real-world ones per variable against their own train split and published neither the mean nor the
standard deviation vector, so no cell of the release is in a physical unit.

**Decision.** Every spec declares `ureg.dimensionless` and the shipped float64 is stored unchanged.
Rescaling toward the original instrument is impossible without the constants, and guessing them
would invent data twice over.

**Consequence.** No card, paper or downstream table may claim this dataset carries real weather,
load, congestion or motion measurements. The names `p (mbar)` and `T (degC)` are labels for which
upstream column a signal came from, and nothing more.

### Values keep their float64 width — **Handled**

**Problem.** The release ships float64. float32 would halve 560,557,248 bytes of values
*(measured)*.

**Decision.** Keep float64.

**Consequence.** The values plane is about 280 MB larger than a float32 one would be. In exchange
the stored values are bit-identical to the release, which is what makes an exact value comparison
against the source possible at all. A narrowing would make that check impossible rather than merely
lossy.

### `BlindWays` runs at 60 Hz, and no sentence states that — **Open**

**Problem.** Neither paper prints a frame rate for the motion capture.

**Decision.** Declare 60 Hz, on two arguments that agree, neither of which states the rate. The
release ships 1,029 clips of 600 frames, which is 617,400 poses *(measured)*, and the BlindWays
paper's dataset table gives 2.8 in its "Hours" column: 617,400 poses over 2.8 hours is 61.25 Hz.
That paper's prediction task covers 10 s, "the next 9.5 seconds of future motion given 0.5 seconds
of past motion", and 600 frames over 10 s is exactly 60 fps.

**Consequence.** A `BlindWays` window claims a 10 s duration on an inference and not on a stated
rate, which is why this entry is open. If the rate is wrong the durations are wrong by that factor
and the values are not. The fallback is `OrdinalAxis()`, a one-line change in `specs.py`.

### Three cadences come from the paper — **Handled**

**Problem.** `Weather`, `ETTm1` and `istanbul_traffic` ship no cadence and no timestamp.

**Decision.** Take one step per 600 s for `Weather` and `istanbul_traffic`, and one per 900 s for
`ETTm1`. The paper states the first two outright, in the sentences quoted above. `ETTm1` is
arithmetic on the paper's own two numbers: 69,680 raw points from July 2016 to July 2018 is one
point every 15.09 minutes. The same paper calls `ETTm1` "minute-level data", which those two numbers
rule out as one point per minute: two years of minutes is 1,051,200 points, not 69,680.

**Consequence.** Each cadence also has an arithmetic check inside the repo, and each passes: a
144-step `istanbul_traffic` window at 600 s is exactly 24 hours, a 36-step `Weather` window at 600 s
is exactly the 6 hours the paper prints, and a 120-step `ETTm1` window at 900 s is exactly 30 hours
*(step counts measured)*.

### `Weather` takes the 21 column names the paper lists — **Handled**

**Problem.** The artifact names none of its 21 signals.

**Decision.** Take the 21 names from Appendix A.3.2, quoted above, in the order it lists them. That
is also the upstream Jena header's order, and the count matches the shipped signal count exactly.

**Consequence.** What stays assumed is that the array's third axis follows the listed order. If the
cut reordered the columns, all 275,100 `Weather` series are mislabelled and no value moves. The
captions cannot settle it the way the `ETTm1` captions settle `var_id`, because a `Weather` caption
describes the weather in prose and never names a column. The fallback is positional names, as the
synthetic sets use.

### Three `Weather` names spell a unit in ASCII — **Handled**

**Problem.** Three of the 21 units carry a non-ASCII sign. The paper prints them as `W/m²` and
`µmol/m²/s`, and a name copied out of a header that has lost its encoding carries the Unicode
replacement character U+FFFD instead.

**Decision.** Write both signs in ASCII, as `2` and `u`: `SWDR (W/m2)`, `PAR (umol/m2/s)` and
`max. PAR (umol/m2/s)`. `tests/test_tables.py` pins that no name carries U+FFFD.

**Consequence.** Those three names spell their unit differently from the paper, and so does `rho
(g/m**3)`, which keeps the Jena header's ASCII spelling of a cube where the paper prints `g/m³`.
None of the four differs in which variable it names. A replacement character carries no information
and would follow every reader of the artifact around.

### The `BlindWays` axis letters and the synthetic names are this connector's — **Handled**

**Problem.** The artifact ships no joint list and no variable list. Both papers state 24 joints of
three variables each, which accounts for all 72 signals, and neither says what the three variables
are or in which order they sit.

**Decision.** Name `BlindWays` signal `k` as joint `k // 3` on axis `k % 3`, giving `j00_x` through
`j23_z`. That assumes the three variables of one joint sit next to each other and run x, y, z. Name
the synthetic signals by position, `x0` and `x1`.

**Consequence.** 74,088 `BlindWays` series and 96,000 synthetic series carry a name the release did
not state. The names are a stable way to address a signal, and they claim nothing about anatomy: the
mapping from joint index to body part is not in the artifact. If the three variables of a joint are
not adjacent, the grouping is wrong and no value moves.

### A one-signal window is named from its own `var_id` code — **Handled**

**Problem.** `ETTm1` and `istanbul_traffic` ship one signal per window. Which column it was cut from
sits in the window's first attribute code, and the release ships no decoding for it.

**Decision.** Index the code into the column order the paper lists: `HUFL, HULL, MUFL, MULL, LUFL,
LULL, OT` and `TI, TI_An, TI_Av`. `meta.json` declares exactly 7 and 3 codes for those two
attributes, which matches the two column counts, and both files name `var_id` first, which is where
Appendix A.2.1 puts the variable index. `convert` refuses a `meta.json` that names something else
there, because a moved `var_id` would mislabel every window whose code still lands inside the column
list.

**Consequence.** This is the one naming rule that reads a data value, so a mislabel would be silent
and per window rather than per component. The captions settle it. Every `ETTm1` caption opens "This
sequence is HUFL." and every `istanbul_traffic` caption opens "This is the variable of TI.", naming
the column in words, and over all 26,499 windows of the two components the name the caption states
agrees with the name this table decodes, with 2,325 windows behind each of the 7 `ETTm1` codes and
3,408 behind each of the 3 traffic codes *(measured)*. A code outside the column list raises rather
than converting.

### The label vocabularies are not carried — **Handled**

**Problem.** Each attribute is an int64 code. The paper's appendix prints label lists for some
attributes, and the release ships no code-to-label mapping.

**Decision.** Store the codes raw. Each corpus-level `_options` annotation carries only the number
of codes, which is what `meta.json` states.

**Consequence.** 26 label lists are absent from the artifact, and the codes themselves are exact
copies. The artifact says how many codes each attribute takes and never what one of them means. The
counts do corroborate the appendix: `Weather`'s seven attributes declare 4, 4, 6, 4, 9, 4 and 4
options and Table 10 prints exactly that many labels for each, and `BlindWays`' 2 and 3 match
Table 8 *(measured)*. What no source states is which label a code stands for, so anyone who needs
one has to bring the appendix and confirm the code order themselves. Writing 26 unconfirmed
vocabularies into the artifact would have looked like evidence.

### Four `ETTm1` windows carry a `season` code the release does not declare — **Handled**

**Problem.** `ETTm1`'s `meta.json` declares nine options for `season`, and Appendix A.2.1 gives the
range as `[0,9]`. Rows 504 to 507 of the train split hold `-1` *(measured over all 16,275 `ETTm1`
windows; no other attribute of any component holds a code outside its declared range)*.

**Decision.** Store the stated code. The build warns one time for the attribute, naming the count
and the first row, and converts every code as it stands. It does not clamp the value, drop the
window or widen the option count.

**Consequence.** `ettm1_season` holds a value that its own `ettm1_season_options` annotation does
not cover, for 4 of 104,628 records. Anyone who trains on that attribute has to decide what `-1`
means, and no source says. Clamping it to 0 would have invented a periodicity.

### An annotation key carries its component — **Handled**

**Problem.** Weather's `season` is one of four calendar seasons and ETTm1's is a periodicity index
in [0, 9]. Schema derivation refuses one key that yields two descriptors.

**Decision.** Prefix every attribute key with a per-component slug: `weather_season`,
`ettm1_season`.

**Consequence.** Every key is longer than the name the release writes, and no build aborts after the
whole download and conversion have already run. `tests/test_connector.py` pins the failure mode with
a colliding key.

### The item is the window, not the caption — **Handled**

**Problem.** A Weather window has three captions. Either the window or the caption could be the unit
of the dataset.

**Decision.** The window. 104,628 records answer 130,828 tasks.

**Consequence.** The caption is not addressable on its own. This matches the release's own loader,
which collapses a multi-caption window to one caption per access, so the caption is not an
addressable unit there either.

### A caption comparison against the release has to compare sets — **Handled**

**Problem.** The release's loader picks one caption per access with an unseeded process-global
random generator that does not depend on the item index. Two passes over the same Weather window
return different captions.

**Decision.** Nothing in the conversion. Recorded because it decides how the conversion can be
checked: compare caption **sets** per window, never a sampled caption. No per-access oracle exists.

**Consequence.** Affects the 13,100 Weather windows, the only ones with more than one caption.

### A caption rides only as a task prompt — **Handled**

**Problem.** The caption could also be stored as a `caption` annotation on the record, which would
make it queryable without reading the task table.

**Decision.** Store it once, as the task's `prompt`.

**Consequence.** `Task.prompt` has no by-reference form, so the text has to sit inline in the task
row either way. An annotation copy would be a second copy of all 130,828 captions. The release's
caption planes are fixed-width UTF-32 and total 235,831,312 bytes; the text inside them is
37,819,859 bytes, every one of them ASCII *(measured)*, so a second copy is not free. Nothing else
in the artifact makes a caption queryable, so a reader who wants one has to read the tasks table.

### Three tasks over one target, not one — **Handled**

**Problem.** A Weather window's three captions differ in wording. They could be three tasks or one
task with three phrasings.

**Decision.** Three tasks. Under generation each caption is a separate specification of what to
synthesize. Nothing in TimeF requires `target_record_id` to be unique across tasks.

**Consequence.** 130,828 tasks rather than 104,628. Collapsing them would lose two of the three
specifications of every Weather window.

### The tasks stream out of the caption planes — **Handled**

**Problem.** The build has 130,828 tasks over 104,628 records. `add_tasks` attaches one batch to one
record, so a caption-per-window release makes one call per window and the dataset ends up holding
every task, every `record_ids` tuple and every `task_ids` tuple in memory. The house rule is to
stream wherever the tasks outnumber the records.

**Decision.** `set_task_stream`. `convert` builds the records and registers the codebook
annotations, then hands over a callable that re-opens the caption planes and yields one
`TSGenerationTask` per caption. Each streamed task carries its own `record_ids`, and the callable
gives a fresh iterator on every call, so that reading it again gives the same tasks. The writer
drains it once, but `iter_tasks` is public and any consumer may read it after that.

**Consequence.** Three things change. A record row stores no `task_ids`, because no task is ever
held in memory to fill them in; `TimeFReader.read()` rebuilds that reverse link from the task rows,
so a full read still resolves a record's tasks, while a records-only read sees the column empty. The
dataset no longer runs the checks that need every task at once, which here is the duplicate-id
check: task ids are the UUIDv7 TimeF generates, and a test pins that they are distinct.
`from_tasks` is unused in this connector, so no derivation check is lost. And the caption planes are
read by the stream rather than by `convert`, so a caption plane whose row count disagrees with its
values file raises when the stream reads it and not while the records are being built. The task
count in the table above is unchanged: streaming decides where the tasks live, not how many there
are.

### The task direction is text in, series out — **Handled**

**Problem.** One window-and-caption pair supports two tasks: generate the series from the text, or
caption the series.

**Decision.** `TSGenerationTask` only. VerbalTS is a text-to-series generation benchmark, so that is
the direction the release supervises.

**Consequence.** The opposite reading derives from the same pair, so a later version can add an
unprompted captioning task without re-converting one value.

### Item order is frozen, and the id padding is what freezes it — **Handled**

**Problem.** The writer sorts series by `source_id`, and a record id is a string.

**Decision.** Enumerate component in the order Section 5.1 introduces them, then split in `train`,
`valid`, `test` order, then row ascending. Pad the row index to five digits, so lexicographic order
equals numeric order.

**Consequence.** The writer walks each of the 18 values files front to back rather than jumping
through it. Dropping the padding would keep the data correct and make every build read the 560 MB
values plane out of order. Five digits is enough for the largest split, which holds 24,000 windows
*(measured)*.

### Both synthetic sets share one spec — **Handled**

**Problem.** `synthetic_u` and `synthetic_m` differ only in their signal count.

**Decision.** One `synthetic` spec covers both, so the release declares five spec types and not six.

**Consequence.** `spec_type` is the writer's primary sort key, so the two sets are stored together.
Anyone who wants them apart selects on the `component` annotation.

### No record carries a `start_time` — **Handled**

**Problem.** No component ships a timestamp, and the release's own loader synthesises a plain index.
Weather's chronological 2014-2020 / 2021 / 2022 split and `istanbul_traffic`'s 2022-11-01 to
2024-06-16 range appear only in the paper's prose.

**Decision.** Leave `start_time` unset everywhere.

**Consequence.** A window has a duration and no place on a wall clock. The `split` annotation is the
only thing that says which period a Weather window came from, and it says it by name and not by
date.

### `subject_ids` is empty everywhere — **Handled**

**Problem.** `BlindWays` recorded 11 blind and low-vision participants. The artifact carries no
per-clip participant id.

**Decision.** Leave `subject_ids` empty.

**Consequence.** The clip-to-person mapping is lost, so `BlindWays` cannot be partitioned by
subject. Any subject-disjoint split over this component is impossible from these bytes.

### Window overlap is not recorded — **Open**

**Problem.** The paper states both slides. `ETTm1` uses "a sliding window with a sequence length of
120 and a stride of 30", `istanbul_traffic` one "with a sequence length of 144 and a stride of 24".
So consecutive windows physically duplicate most of their values, and no field of the artifact says
which windows those are.

**Decision.** Nothing. The connector copies what the release ships. TimeF has no field for "this
window is that window shifted by 30 steps", and a recorded relation would be part measurement and
part guess: the rows do sit in `var_id`-major order, and 12,978 of the 13,006 neighbouring
same-variable pairs in the `ETTm1` train split overlap by exactly 90 steps, but 28 do not
*(measured)*.

**Consequence.** 75% of an `ETTm1` window and 83% of an `istanbul_traffic` window is a copy of its
neighbour's values. Anyone measuring the information content of this dataset, rather than its size,
needs to know that. A reader should not treat 16,275 `ETTm1` windows as 16,275 independent
observations. Reconstructing the relation by matching values is possible, at the cost of those 28
pairs.

### `BlindWays` clips are all 600 frames, and no source says how — **Open**

**Problem.** Every clip is exactly 600 frames. VerbalTS says it did not change that: "For the time
series data, we maintain the same shape as the raw dataset." So the uniform length comes from the
capture, and the BlindWays paper counts "1,029 motion clips and approximately 0.6 million human
poses", which is consistent with the shipped 617,400 *(measured)*. Neither paper says whether a clip
was padded, cropped or cut to a fixed 10 s.

**Decision.** Copy the 600-frame arrays unchanged and do not try to invert a transform no source
describes.

**Consequence.** A clip's stated duration is exact only if the clips were cut rather than padded.
This component is 44.67% of the corpus bytes and 1,029 of 104,628 records *(measured)*, so the
uncertainty covers a large share of the values and a small share of the records.

### Annotation and task ids are generated, not stated — **Handled**

**Problem.** Nothing resolves an attribute annotation, a context annotation or a generation task by
id. Only the corpus-level `_options` annotations are resolved, by every task's
`input_annotation_ids`.

**Decision.** State an id only for those. Let every other annotation and every task take the
UUIDv7 that TimeF generates, as the other connectors in this repo do.

**Consequence.** Two builds of this release give the same record ids and series ids, and different
annotation and task ids. A task is addressable by its `target_record_id` and its `prompt`, which are
both stable. Anyone who needs a citable task id has to record the id from the build they used.

### `download_async` gives back one handle per component — **Handled**

**Problem.** The house shape for a connector is one `<Dataset>Source` handle, with `convert`
starting at `raw_refs[0]`. This release is six Drive folders, each with its own `meta.json` and its
own nine arrays.

**Decision.** Return six `VerbalTsComponent` handles, one per folder, and iterate them in `convert`.
The type is named after what it holds, the way `timenet/hello_world`'s `HelloWorldRecording` is. Six
is the release's own division into folders, not one handle per record.

**Consequence.** `convert` reads six `meta.json` files and needs no folder-name arithmetic, and a
reviewer reading the checklist finds the divergence stated here rather than having to spot it.

### The six components hold six licence positions — **Open**

**Problem.** `ETTm1` is CC BY-ND 4.0. `BlindWays` and both synthetic sets carry no statement at any
primary source. `Weather` is CC BY 4.0 data with Apache-2.0 captions. `istanbul_traffic` is CC0,
asserted by a third-party uploader over municipal data. The card has one licence field.

**Decision.** Declare `other` with `license_url` pointing at the VerbalTS code repository, which is
the only URL any primary source offers. Build for measurement.

**Consequence.** **Do not redistribute the converted bytes.** Three of the six components have no
licence at all, which is not the same as permissive. Somebody has to write to the authors before
this dataset is published anywhere. A per-component licence field would be the real fix, and TimeF
has no such field today.

### The SHA-256 pins describe one download — **Open**

**Problem.** Google Drive exposes no revision and no digest of its own. The 60 digests in `files.py`
were computed from one full download of the release on 2026-09-06.

**Decision.** Pin all 60 files by Drive id, byte count and SHA-256, and re-read every one of them
after a download. `check_drive_body` does that, and it also catches Drive's virus-scan interstitial,
which Drive serves as a small HTML page under HTTP 200 when `confirm=t` is dropped from the URL.

**Consequence.** The pin proves that two builds read the same bytes. It does **not** prove those are
the bytes the authors released: a replacement made before 2026-09-06 sits inside the pin, and no
independent digest exists to check it against. What has been checked is the table itself. All 60
entries were re-hashed against a full local copy of the folder, and all 60 matched on size and on
digest *(measured)*, so no entry is mistranscribed.

### The build does not record what it read from Drive — **Open**

**Problem.** `manifest.json` has no field for a source pin, so the digests live in `files.py` and
nowhere in the output.

**Decision.** Leave it. Recording the pin in the manifest needs a manifest change, which does not
belong in a connector.

**Consequence.** A reader of a built artifact cannot tell which bytes produced it without reading
this connector's source at the commit that built it.

## Warnings this build emits

One line, for one attribute of one split of the pinned release:

```text
ETTm1/train: 4 of 13013 season codes fall outside the 9 options meta.json
declares, first at row 504 with code -1; every code is converted as the
release states it
```

That is the only inconsistency the release ships that this connector can see. Every other case it
can detect is a corrupt or changed artifact, and each raises `TimeFFormatError` instead of warning:
a values file that is not three-dimensional or not float64, a caption or attribute plane that does
not match its values file, an attribute plane that disagrees with `meta.json`, a `meta.json` that
does not parse or whose two lists disagree, a `meta.json` that does not name `var_id` first for a
component whose signal name is decoded from it, a signal count that does not match the component, a
`var_id` outside its column list, and a downloaded file whose size, NPY magic or digest is wrong.

## What is not built

**The rest of `meta.json`.** `attrs_split`, `final_split` and `n_random_samples_per_attr` are not
converted. The release's own loader reads only `attr_list` and `attr_n_ops`. The split sizes are
recoverable from the array shapes, and `n_random_samples_per_attr` is 1,000 in the five files that
carry it, which is the per-combination sample count Appendix A.1 states; `Weather`'s `meta.json`
omits the field *(measured)*.

**The attribute label vocabularies**, for the reason in "The label vocabularies are not carried".

**A captioning task.** Only the generation direction is built.

**Any wall-clock time**, and any link from a `BlindWays` clip to the participant who walked it.
Neither is in the release.

**The window-overlap relation**, for the reason in "Window overlap is not recorded".

TimeF also adds a control plane the NPY release has none of: a records table, a series index, an
annotations table, a tasks table and a manifest. That is not a subtraction, but it is the one part
of the output that has no counterpart in the source.
