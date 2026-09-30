# HEARTS frozen test cases

The frozen test cases of HEARTS, a health time-series agent benchmark. One record is one test case:
the signals the reference harness hands its agent, plus the question and the ground-truth answer.
The release is one Python pickle per case in a flat `<corpus>/<task>/<index>.pkl` tree over five
redistributed upstream corpora: CGMacros (interstitial glucose), HARESPOD (respiration, oxygen
saturation and heart rate from a hypobaric chamber), Coswara and COUGHVID (respiratory audio) and
VCTK (speech). This connector converts 1,005 of the 1,455 published cases, over 21 of the 30
published task directories.

- **id**: `yang-ai-lab/hearts`
- **source**: https://huggingface.co/datasets/yang-ai-lab/HEARTS
- **licence**: `other`

The release declares no licence at any source. Most of it derives from human-subject recordings, and
at least one upstream corpus is NonCommercial-ShareAlike, so the card says `other` and this
connector makes no redistribution claim about the converted bytes.

Two pins, not one. The data comes from the Hub repository at commit
`7c18df521ae36cbc6b61e17782f1ac08dc378ea1`. The 21 prompts are not in the data at all: they are
transcribed from the release's own experiment code, GitHub `yang-ai-lab/HEARTS` at commit
`02854baf8b14db322a0d115c47f906a8ee6de1b8`. A build that reads only the Hub repository cannot
reproduce them.

## The source of truth

| fact | source of truth | the other reading |
| --- | --- | --- |
| the VCTK rate | `exp/vctk/base.py`, 16000 Hz | upstream VCTK publishes at 48 kHz |
| a HARESPOD unit | the values, scaled to the unit interval | upstream records percent and bpm |
| an audio buffer's rate | the payload's own `sr`, per file | the prompt, which says 48 kHz |
| a question's wording | `exp/<source>/<task>.py`, GitHub pin | the artifact states no question |

The artifact wins wherever it states the fact, because a reader can re-measure it. Where the
artifact is silent (the VCTK rate, every prompt, every answer vocabulary) the release's own code is
the only source there is, and this README names the commit it was read at.

## What the description states

The Hub repository ships a card, `README.md`, beside the pickles. A build does not fetch it, because
this connector asks for `*.pkl` only, but it is the release's own prose and five of the things it
says decide something here. They are quoted below verbatim, at the Hub pin, so a reader can check
them rather than take this file's word.

> This repository hosts the *fixed ("frozen") test cases* used by the HEARTS benchmark. Each file is
> a Python pickle (`.pkl`) containing one test-case payload.

Decides the record. One file is one payload is one record, and no case is merged with another or
split.

> - `<dataset>`: dataset identifier (e.g., `cgmacros`, `vitaldb`)
> - `<task>`: task identifier within that dataset
> - `N.pkl`: N-th fixed test case for that dataset/task

Decides the record id and the walk order. The corpus, the task and the case index are the release's
own names for a case, so a record id is built from all three and nothing counts across files. The
card's own example names a `vitaldb` corpus that this revision does not publish.

> Security note: Python pickles can execute code when loaded. Only unpickle files you trust.

Decides how the files are read. The release says its own format can run code, and the reference
harness calls bare `pickle.load` anyway. This connector reads every file through a restricted
unpickler instead, so the two sides do not run the same deserializer.

> Pickled objects can be code/version-dependent; if you see load/format issues, align your HEARTS
> code version with the data release.

Decides the pandas pin. The files were frozen under pandas 2.x, so this connector's
`requirements.txt` asks for pandas 2 and says why.

> These files are intended to be immutable inputs for reproducible evaluation.

Decides the two guards. The Hub revision is pinned to a commit, and a build checks the case count of
every directory it fetched against the count this connector was written against, so a moved
revision stops the build instead of converting quietly.

Everything else the conversion leans on is in the release's experiment code, at the GitHub pin
above, because the card states no question, no answer vocabulary and no sampling rate. These are the
release's own sentences from that code:

> A raw audio waveform is provided at 'input/audio.npy'. This is a mono audio signal sampled at
> {self.sample_rate} Hz.
> — the prompt of `vctk/waveform_temporal_direction_detection`, where `sample_rate` defaults to
> 16000 in `exp/vctk/base.py`

Decides the VCTK time axis. No VCTK payload holds a rate, so 16000 Hz is read from the reference
implementation and from nowhere else.

> One audio file is saved at 'input/audio.wav'. It is a cough audio sampled at 48 kHz.
> — the prompt of `coughvid/cough_detection_good_qual`

Decides nothing. Every COUGHVID and Coswara payload states its own `sr`, and the connector reads
that field per file. Over the 472 COUGHVID and Coswara cases, 272 COUGHVID cases and 187 Coswara
cases run at 48 kHz, 12 Coswara cases at 44.1 kHz and one at 192 kHz *(measured)*, so a constant
taken from this sentence would misplace thirteen of them.

> There are two columns in this csv file: one is timestamp_min containing the time of each reading
> (in integer minutes), and the other column "Libre GL" contains glucose values (mg/dL).
> — the prompt of `cgmacros/meal_time_localization`

Decides how the meal-time answer is read. The release states the readings in whole minutes and its
own output format asks for a float, so the answer is carried to the whole microsecond rather than
read as an integer. All 50 released answers are whole minutes *(measured)*.

> Each segment corresponds to one of the altitude ranges provided below:
> - 1.5k-2k meters
> - 2k-2.5k meters
> - 2.5k-3k meters
> - 3k-3.5k meters
> - 3.5k-4k meters
> — the prompt of `harespod/altitude_ranking_respiration`

Decides nothing, and it is kept as written. The task hands the agent three segments and offers five
ranges. The prompt is transcribed unchanged, because the reference harness scores the answer against
this wording.

### The two edits made to a released prompt

A prompt is transcribed from `exp/<source>/<task>.py`. Two kinds of edit are made to it, and both
are listed here.

**A file reference names a signal instead.** The release writes its agent's inputs into a sandbox
directory and asks about `input/audio.wav`, `input/audio.npy` or `input/[label].csv`. A TimeF
consumer has no such directory, so each reference names the signal that carries the same values:
`data.signal`, `waveform`, `segment_dfs.A.rsp`. Every signal named in a prompt is a signal the
converted record holds, under that exact name *(measured)*.

**A per-case value interpolated into the prompt is dropped.** One prompt is shared by every case of
its directory, so a value that differs per case cannot be in it. Two directories do this.
`vctk/waveform_temporal_direction_detection` writes this sentence, and it is gone:

> Speaker: {data["speaker_id"]} (accent: {speaker_metadata.get("accent", "unknown")}).

The speaker id is the record's subject id instead, and the accent is upstream VCTK metadata that no
payload carries, so it is not converted at all.
`coswara/cough_covid_status_classification_with_symptoms` writes `Symptoms: {symptoms_text}`, and
the transcribed prompt names the `symptoms` annotation, which carries the same map.

Nothing else in the wording changed. The release's own output key is kept exactly as the harness
parses it, so a prompt asks for `has_cough` and not for a paraphrase of it, and the release's own
wording is kept even where it reads oddly (`continuous glucose monitors (CGM) data`). Two
differences in whitespace remain: the two `harespod/altitude_ranking_*` prompts hold a line of
sixteen spaces where a blank line belongs, and the VCTK prompt ends with a newline. Neither is
carried.

## What one record holds

**Series.** One per value column of every pandas frame in the payload, plus one per bare float
array, named after the payload's own key path and the column: `respiration_dfs.respiration_A.rsp`,
`data.signal`, `waveform`. A record's series are sorted by that name, because a Python dict fixes no
order and two builds of one release must agree. Over the real release this is 1,555 series and
211,959,663 values *(measured)*.

Seven specs cover them:

| spec | unit | dtype | corpus |
| --- | --- | --- | --- |
| `cgm` | mg/dL | float64 | CGMacros |
| `respiration_norm` | dimensionless | float64 | HARESPOD |
| `spo2_norm` | dimensionless | float64 | HARESPOD |
| `heart_rate_norm` | dimensionless | float64 | HARESPOD |
| `audio_coswara` | dimensionless | float32 | Coswara |
| `audio_coughvid` | dimensionless | float32 | COUGHVID |
| `audio_vctk` | dimensionless | float32 | VCTK |

The dtypes are the release's own. Nothing is promoted or demoted.

**The time axis is decided per series, from that series' own time column, never per corpus.** A
column whose steps are all equal becomes a `RegularAxis` holding a period. Anything else becomes an
`IrregularAxis` plus an explicit int64 microsecond offset stream. HARESPOD frames step at a constant
10 ms or 1 s and take the first branch. The CGMacros windows take the second, because they are
genuinely uneven: one 36-row reference window carries 22 one-minute steps and 13 seven-minute ones
*(measured)*. An audio buffer carries no time column and takes `RegularAxis.from_rate_hz` from the
payload's `sr`, or 16000 Hz for VCTK.

**Annotations**, all record-scoped and none scoped to a signal:

| key | value | source | where it came from |
| --- | --- | --- | --- |
| `hearts_source` | the corpus directory, e.g. `harespod` | `hearts:provenance` | the file's path |
| `hearts_task` | the task directory, e.g. `hr_resp_pairing` | `hearts:provenance` | its path |
| `testcase_idx` | the case index in that directory | `hearts:provenance` | the file's name |
| `values_normalized` | `true`, on HARESPOD records only | `hearts:provenance` | the entry below |
| `audio_quality` | the integer Coswara quality rating | `hearts:provenance` | `data.quality` |
| `symptoms` | the subject's symptom map | `hearts:agent_input` | the `symptoms` key |
| `answer_options` | one directory's answer set | registered once per directory | the validators |

`symptoms` is the only payload field the reference harness shows its agent besides the series, and
it reaches one Coswara directory. `answer_options` is registered on the dataset and referenced by
the tasks of its directory, rather than copied onto each of them: 12 annotations in all
*(measured)*.

**Subject ids are namespaced by their corpus**: `cgmacros-CGMacros-007`, `vctk-p361`. The five
corpora number their subjects independently and nothing in the release says two ids from different
corpora are different people. The two-window comparison task carries a pair.

**Record ids** are `hearts-<corpus>-<task>-<index>`, with the index padded to two digits:
`hearts-harespod-hr_resp_pairing-00`. They come from the release's own directory and file names, so
two builds of one revision agree. Cases are walked corpus first, then task directory, then numeric
case index. The padding makes a text sort of the ids agree with the numeric one, so a record's
neighbours on disk are its neighbours in the benchmark.

No record carries a `start_time`. See the wall-clock entry below.

## The tasks this connector builds

One task per record, typed by the answer the reference harness scores rather than by one uniform
question-answering type.

| the question | type | count *(measured)* | scope |
| --- | --- | --- | --- |
| pick a label from a closed set | `ClassificationTask` | 565, over 12 directories | the record |
| answer with a JSON object or list | `AnswerTask` | 295, over 6 directories | the record |
| predict a number with a unit | `ScalarPredictionTask` | 95, over 2 directories | the record |
| name the minute a meal starts | `TemporalLocalizationTask` | 50, over 1 directory | the record |

The two scalar directories predict in `mg/dL` and in `mg*min/dL`.

A `ClassificationTask` carries the answer vocabulary of its directory through
`input_annotation_ids`, and its `target_schema` is the id of that same registered annotation, so the
field resolves to the set rather than naming a string nothing holds. One function builds the id and
both ends read it from there.

An answer that is not already text is mapped to the form the harness scores: a boolean becomes
`"true"` or `"false"` (172 answers), the VCTK direction index 0 or 1 becomes `"forward"` or
`"reversed"` (50 answers), and the meal comparison's status map becomes the label of the window
whose subject is normal (50 answers) *(measured)*. The remaining 293 classification answers are
already the label. Every mapped label is checked against that directory's vocabulary, so a release
that changes its wording stops the build instead of storing a label nothing accepts.

**The tasks stream, and the connector holds none of them.** `set_task_stream` gets them, so each
task carries its own `record_ids` and the stored record rows carry no `task_ids`. A consumer who
reads the whole dataset loses nothing by it: `TimeFReader.read()` rebuilds that reverse map from the
task rows. One who walks the records without their tasks reads the field empty. 1,005 tasks would
fit in a list, so this follows the repo's convention rather than a memory limit. The stream walks
the tree a second time and reads each payload again for its answer, which is why the read count
below is three and not two.

A stream also skips the checks that need every task at once, so nothing rejects a repeated task id.
A task id is its record id plus `-qa`, so the ids are distinct as long as the record ids are, and a
test pins that.

No splits are produced. The release is a single frozen test set.

## Inconsistencies and decisions

### 450 of the 1,455 published cases are not converted — **Not built**

**Problem.** The release publishes 1,455 test cases over 30 task directories. Nine of those
directories do not fit a TimeF record, for three different reasons. The arithmetic is 1,455 minus 50
meal-image cases, minus 50 symptoms-only cases, minus 350 forecasting and imputation cases, which
leaves 1,005 *(measured)*.

**Decision.** Convert the other 21 directories, and record each exclusion with its reason:

| directories | cases | bytes *(measured)* | why |
| --- | --- | --- | --- |
| `cgmacros/meal_img_classification` | 50 | 42,001,565 | 98.8% of it is JPEG meal photographs |
| `coswara/..._symptoms_only` | 50 | 8,732 | nine booleans and a label, and no array |
| the four `cgmacros/meal_forecasting*` | 200 | 14,283,338 | the answer is held out of the input |
| the three `cgmacros/non_meal_imputation*` | 150 | 1,143,450 | the same: the removed values |

The seven forecasting and imputation directories are among the smallest files in the release, so
they move the case count far more than the byte count: 350 cases, which is 24.1% of the release,
against 1.2% of its bytes *(measured)*.

**Consequence.** A build reads 1,263,571,219 of the release's 1,321,008,304 bytes, which is 95.7%
*(measured)*, and produces 69.1% of its cases. Anyone comparing against the release's own case count
will find 450 fewer. The symptoms-only cases could only have been converted by inventing a nine-step
series out of the symptom booleans, which would be worse than dropping them. A forecasting case
needs its held-out values in a second record, which breaks the one-case-to-one-record mapping the
benchmark reads.

### The HARESPOD values carry no physical unit — **Handled**

**Problem.** The release ships respiration, oxygen saturation and heart rate min-max scaled to the
unit interval, and does not publish the scaling constants. Percent saturation and beats per minute
are not recoverable from the artifact.

**Decision.** Give those three specs `dimensionless`, and put a `values_normalized` annotation on
every HARESPOD record whose description says why. Declaring percent and bpm anyway would be a claim
the data does not support.

**Consequence.** 433,073,354 bytes, 34.3% of what this connector converts, carries no physical unit
*(measured)*. A consumer cannot compare a HARESPOD value against a clinical threshold.

### `audio_type` is dropped from the artifact entirely — **Handled**

**Problem.** The Coswara payloads carry an `audio_type` field, at the top level and again inside
their `data` dict. The release computes the audio-classification answer from it:
`exp/coswara/audio_classification.py` maps an `audio_type` of `vowel-o` to the answer `speech`.

**Decision.** Do not carry the key anywhere in the converted artifact, not even withheld from the
task's inputs. Only the payload keys a definition names in `input_keys` become annotations, and no
definition names this one. A test asserts the key is absent from every annotation of the built
dataset.

**Consequence.** One short string on 200 records is not converted *(measured)*. Nothing in the
artifact hands a model the answer to that task.

### No record carries a wall-clock anchor — **Handled**

**Problem.** The CGMacros frames hold absolute local timestamps such as `2019-11-16 19:18:00`.

**Decision.** Reduce every time column to microsecond offsets from its own window's first row, and
set no `start_time`. A local wall clock with no zone is not an instant, and carrying it would put a
trace of when a named participant ate into the artifact, from a corpus of human-subject data under
NonCommercial-ShareAlike terms.

**Consequence.** 283 of the 333 CGMacros frames carry a 19-character `Timestamp` string on each of
2,061,436 rows, and none of that text is converted *(measured)*. The other 50 count minutes from the
meal and state no calendar day. This is a real content difference and not only a smaller file: the
release's own loader writes the absolute timestamps into the agent's CSV, so a consumer of this
dataset cannot recover the calendar day a window came from.

### Fields that neither place a value in time nor reach the agent are not carried — **Handled**

**Problem.** Each payload holds scalars beside its series: the CGMacros `meal_time`, `window_start`
and `window_end`, the Coswara `duration`, the VCTK `recording_id`, and the per-meal dicts of the
meal-comparison cases.

**Decision.** Do not carry them. The last of those hold the diabetes status the answer is derived
from, so carrying them would leak the answer the way `audio_type` would.

**Consequence.** A handful of scalars per record is not converted, well under 0.1% of the bytes. A
consumer cannot recover the meal's wall-clock moment or the VCTK utterance id.

### The positional frame index and the `timestamp_min` column are not series — **Handled**

**Problem.** Every pandas frame carries an integer index, and 50 CGMacros cases carry a
`timestamp_min` column that restates the frame's time column in whole minutes.

**Decision.** Drop the index, because TimeF places a value by its axis and has nowhere to put a row
label; the release's own loader drops it too when it writes the agent's CSV. Read `timestamp_min` as
part of the axis rather than as a series, because the axis already carries what it says, and check
that at convert time: a `timestamp_min` that is not the offset stream in whole minutes stops the
build and names the frame and the file.

**Consequence.** One int64 column on 50 records, and the row label of every one of the 1,033 frames,
is not converted. The two halves are not alike. `timestamp_min` is exactly recoverable: on all 50
records it equals the stored offset stream divided by 60,000,000 *(measured)*, which is what the
check enforces. Most of the index is not. 903 of the 1,033 frames carry a label that is not a range
from zero (it is the row's position in the corpus frame the window was cut from) and nothing stored
recovers it *(measured)*. The other 130 frames carry a plain range.

### A JSON answer records no dtype — **Handled**

**Problem.** 295 answers are a mapping or a list rather than a label or a number. Every one of the
50 `coughvid/mfcc_mean_std` answers is 26 statistics: 13 MFCC means and 13 standard deviations
*(measured)*.

**Decision.** Store them as canonical JSON with sorted keys. Rebuild every NumPy scalar as a plain
Python value first, because `json.dumps` writes an `np.float32` or an `np.int64` as a quoted string
otherwise. Write each number at the shortest decimal text that reads back as the source value at its
source dtype, which is what `str` gives for a NumPy scalar. Widening a float32 with `item` instead
would print the float32 representation error as digits the source never held: `np.float32(0.1)`
becomes `0.10000000149011612`, 19 bytes where 3 carry the same value.

**Consequence.** The JSON records no dtype, so a reader who wants a NumPy dtype has to know which
one to cast to. At this revision that costs nothing to recover: every number in every answer is
already a Python `float` or `int`, and the only NumPy scalar in any in-scope payload is the
`np.int64` meal-time answer of 50 cases *(measured)*. The shortest-text rule therefore protects a
repin rather than this revision. One MFCC answer is 535 to 552 bytes of JSON text against 208 bytes
of float64 binary *(measured)*.

### The meal-time answer is a count of minutes, and the release calls it a float — **Handled**

**Problem.** `cgmacros/meal_time_localization` asks for the minute a meal starts, and the release's
own output format declares that answer a `float`. A `TimePoint` stores whole microseconds, so the
minutes have to be converted, and reading them as an integer would drop a fraction of a minute
without a word.

**Decision.** Multiply the minutes by 60,000,000 and round to the whole microsecond, which is the
finest thing a `TimePoint` holds. A negative answer stops the build by name.

**Consequence.** Nothing is lost at this revision: all 50 released answers are `np.int64` whole
minutes, from 0 to 117, and every one of them lands inside the window its own record covers
*(measured)*. Half a minute in a later release would now reach the stored timeline instead of
vanishing.

### The answer vocabularies are not published — **Handled**

**Problem.** 12 of the 21 directories score against a closed set of answers. The artifact does not
publish those sets; only the release's output validators hold them.

**Decision.** Transcribe them, register one shared `answer_options` annotation per directory, and
reference it from every task in that directory rather than copying it onto each one.

**Consequence.** The vocabularies are a transcription from code at the GitHub pin, not data from the
artifact. If the release changes one, this connector's copy is wrong and the build fails on the
first answer that no longer maps.

### The pickles need pandas 2 — **Handled**

**Problem.** The files were frozen under pandas 2.x and name private internals
(`pandas._libs.internals._unpickle_block`, `pandas._libs.arrays.__pyx_unpickle_NDArrayBacked`) that
pandas 3 does not provide.

**Decision.** Pin `pandas>=2.2,<3` in this connector's `requirements.txt`, and import pandas inside
the function that needs it with a message naming that file.

**Consequence.** A build in an environment with pandas 3 fails outright rather than reading part of
the release. Every one of the 1,005 files is affected.

### The pickles are read through a restricted unpickler — **Handled**

**Problem.** A pickle can run arbitrary code while it is rebuilt. The reference harness calls bare
`pickle.load`.

**Decision.** Read every file through an unpickler that allows exactly the 15 module-and-name pairs
the release's own files ask for and refuses anything else by name. The two sides therefore do not
run the same deserializer.

**Consequence.** The allowlist is measured, not sampled. All 1,005 in-scope files were read with an
unpickler that recorded every pair it resolved: they ask for exactly these 15 pairs, every entry is
used by at least one file, none is missing, and all 1,005 are pickle protocol 4 *(measured)*. The
measurement covers those 1,005 files and nothing else. It is still a record and not a proof: a file
written by a different pandas or numpy build can name a pair that belongs there, and it will stop
the build after a 1.26 GB download. The refusal names the file and the pair, says the list is a
record of the pin, and says where to add the pair.

### A list, a tuple or a bare Series is dropped without a word — **Open**

**Problem.** The payload walk descends into dicts and nothing else. A frame becomes series, a 1-D
float array becomes series, an array of any other shape or dtype stops the build by name, and a
scalar is skipped on purpose. A list, a tuple and a bare pandas Series match no arm and vanish
silently.

**Decision.** Leave the walk as it is, on the evidence of a full sweep. That sweep read all 1,005
converted files through this connector's own unpickler and found 1,033 frames and 522 arrays, every
array 1-D float32, and every dropped node a scalar *(measured)*. It found no list, no tuple and no
bare Series anywhere, so nothing is lost at this revision and no guard was written for a case that
does not occur.

**Consequence.** A later release that ships a series inside a list would convert to a record quietly
missing a signal. Whoever repins this connector should re-run that sweep before trusting the case
counts. The same sweep is what says the refusal of a non-1-D-float array costs this revision
nothing.

### An empty frame would have no first offset to count from — **Handled**

**Problem.** Every offset stream is counted from its own frame's first row, so a frame with no rows
has nothing to count from and the arithmetic would raise an `IndexError` out of the walk.

**Decision.** Fail by name on it rather than guess: a frame with no rows raises and says which frame
in which file.

**Consequence.** No released frame is empty. The smallest CGMacros frame holds 36 rows and the
smallest HARESPOD frame holds 300, over all 1,033 frames of the 1,005 in-scope files *(measured)*.
The guard therefore covers a repin and not this revision.

### Every file is read three times — **Handled**

**Problem.** A pickle has no header. A payload's length, its rate and its axis are known only after
the whole object is rebuilt. That holds for the answer too: it cannot be reached without rebuilding
the whole payload.

**Decision.** Read each file once in `convert` to learn the shapes, once more in the task stream to
reach its answer, and let the per-series loaders re-read it when the writer drains them, backed by a
two-entry payload cache. The cache is small on purpose: the writer walks series in spec-type order,
so every series of one file that shares a spec type is contiguous in that walk and two entries serve
all of them.

**Consequence.** A build reads about 4.0 GB over 1.26 GB of files: every file three times, plus a
fourth read of `harespod/hr_resp_pairing` and `harespod/spo2_resp_pairing`, which are 216,921,012
bytes *(measured)* and spread their signals over two spec types each, so the loaders unpickle them
twice rather than once. Closing the loader gap needs about 50 resident payloads, which would hold 50
audio buffers at 48 kHz while the audio blocks are written, so the second read of a small HARESPOD
frame is the cheaper of the two. The stream's own pass is what streaming the tasks costs here; the
alternative is to keep 1,005 answers in memory from `convert` until the writer walks them.

### No non-finite value was found — **Handled**

**Problem.** A NaN or an infinity in a source would be stored as one, and no spec here is nullable.

**Decision.** Nothing to do. A scan that walks wider than the connector does, descending into the
answers and reading every frame and array, found zero non-finite values across all 1,005 converted
files *(measured)*.

**Consequence.** Every value in this dataset is finite. A later release could change that, and
nothing in the build would warn.

## Warnings this build emits

`None.` Every surprise this connector can meet is a fact about a case it cannot place, so each one
stops the build and says which: a file not named after its test-case index, a payload that is not a
dict, an unlisted pickle global, a missing key on a path a loader follows, a frame with no time
column, a frame with no rows, a time column whose dtype this connector does not read, an unknown
value column, a `timestamp_min` that is not the frame's own time column in whole minutes, an array
that is not a 1-D float buffer, a bare array in a corpus with no audio spec, an audio buffer with no
rate, an answer outside its vocabulary or in a shape its own directory does not state, a negative
meal minute, a tree holding nothing in scope, and a task directory whose case count has moved. A
refusal that read a payload names the file; one that read an answer names the directory and the
value. Nothing here is repaired in silence and nothing is logged and continued.

## What is not built

The nine out-of-scope task directories, listed in the exclusion table above. They are not downloaded
either, so a build fetches 21 directories and not 30.

The prompts and the answer vocabularies are the release's own words, but they come from its GitHub
repository and not from the Hub artifact. A build cannot reproduce them from the Hub repository
alone.

The upstream corpora themselves. This dataset is the frozen test cases of a benchmark, which are
windows and buffers cut out of five other releases. A record cannot be traced back to the recording
it was cut from: the release states the corpus and the subject and nothing finer.

The scaling constants that would turn a HARESPOD value back into percent saturation or beats per
minute. The release does not publish them.
