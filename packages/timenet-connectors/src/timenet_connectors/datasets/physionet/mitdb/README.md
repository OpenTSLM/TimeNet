# MIT-BIH Arrhythmia Database

`physionet/mitdb` holds the MIT-BIH Arrhythmia Database ([Moody and Mark, 2001][paper]): 48
half-hour, two-lead ambulatory ECG recordings from 47 subjects, recorded at Beth Israel Hospital
from 1975 to 1979, with reference beat and rhythm labels from two cardiologists in consensus.

The connector reads the one zip of [PhysioNet's `mitdb` 1.0.0][physionet], pinned by its SHA-256
(`47b26926…0a87`, 73 MB). The license is ODC-By 1.0. PhysioNet corrected `102.atr` in place in 2018
without a new version, so the digest, not the version, names the files.

There is no paper repository. The connector follows the authors' own sources instead:

| Source | What the connector takes from it |
| --- | --- |
| [PhysioNet page][physionet] | The intended uses: beat detection, beat classification, arrhythmia detection |
| The directory, `mitdbdir/` in the archive | The record selection, leads, digitization and labelling |
| The directory's `src/dbnotes.c` | How header comments, rhythm episodes and signal quality are read |
| WFDB `ecgcodes.h`, `ecgmap.h` ([wfdb][wfdb]) | The annotation codes, and which codes are beats (`isqrs`) |
| WFDB `bxb`, `rxr`, `epicmp`, `ecgeval` | How an algorithm is scored on the database (ANSI/AAMI EC38 and EC57) |

The WFDB files were read at commit `de6b1d39`. George Moody wrote both the directory and WFDB.

[paper]: https://doi.org/10.1109/51.932724
[physionet]: https://physionet.org/content/mitdb/1.0.0/
[wfdb]: https://github.com/bemoody/wfdb

## The source data

The archive's `mit-bih-arrhythmia-database-1.0.0/` directory holds, for each record in `RECORDS`:

| File | Content |
| --- | --- |
| `<record>.hea` | The header: two signals, 360 Hz, 650,000 samples, and comment lines |
| `<record>.dat` | Both signals, interleaved, in WFDB format 212 (12-bit samples) |
| `<record>.atr` | The reference annotations |
| `<record>.xws` | A WAVE viewer settings file, not data |

- **Records.** 23 records (100 to 124) were chosen at random from over 4,000 Holter tapes, and 25
  (200 to 234) to include rare but clinically important arrhythmias, so pooled results don't
  estimate prevalence. Records 201 and 202 come from the same subject.
- **Signals.** The upper signal is usually modified lead II (`MLII`), the lower usually `V1`
  (sometimes `V2`, `V5` or `V4`). Records 102 and 104 have `V5` on top, and in record 114 the two
  are swapped. The signals are named from the header, so each name is the lead. The analog tapes
  skew the two signals by up to 40 ms. Records 102, 104, 107 and 217 hold paced beats.
- **Digitization.** 11 bits over ±5 mV at 360 Hz: 200 units per mV, 1024 at 0 V.
- **Header comments.** The first line starts with the age and the sex, and the second line lists
  the medications, as `dbnotes.c` reads them. The age is -1 in records 103 and 219, which
  `dbnotes.c` prints as "not recorded". The first line's other fields, such as `1085 1629 x1`, are
  not documented. Later lines are the authors' notes on the record.
- **Annotations.** Each annotation has a sample, a code, a subtype and an optional text. `chan`
  and `num` are 0 throughout:

| Kind | Codes | Count |
| --- | --- | --- |
| Beats (`isqrs`) | `N` 75,052, `L` 8,075, `R` 7,259, `V` 7,130, `/` 7,028, `A` 2,546, `f` 982, `F` 803, `!` 472, `j` 229, `a` 150, `E` 106, `J` 83, `Q` 33, `e` 16, `S` 2 | 109,966 |
| Rhythm changes | `+`, with the rhythm as text, such as `(AFIB` | 1,291 |
| Signal quality changes | `~`, with each signal's quality in the subtype | 616 |
| Comments | `"`, with the text `MISSB` (428), `TS` (6) or `PSE` (3) | 437 |
| Other non-beats | `x` blocked APC (193), `\|` artifact (132), `[` and `]` ventricular flutter (6 each) | 337 |

- **Beat positions.** Since 1983 each beat label is at the major local extremum of the QRS, the
  R-wave peak in general. Record 117 was realigned in 1998.
- **Older files.** `102-0.atr`, the eight `.at_` files and `203.at-` are the annotations from
  before the corrections the directory lists. `x_mitdb/` holds 10-minute excerpts of 23 records,
  with the same signals. The connector reads none of them.

## The TimeNet dataset

### Records

One Record per record, in `RECORDS` order, with one Source of its two leads. Each lead is a lazy
float32 signal in millivolts on a 360 Hz axis, read with WFDB's gain and baseline. Every value is
a whole ADC step, so multiplying by 200 and rounding gives back the stored integer.

A sample's time is `floor(sample × 25000/9)` µs, which `AXIS.time_offset_us` computes and
`AXIS.index_at_or_after` inverts.

`Record.metadata` holds the shared `subject_id`, which is the record name, except that record 202
has `201`. That is our choice, from the directory's statement that the two share a subject. Its
own fields hold the record name, the leads in the header's order (upper signal first,
since a Record lists its signals by name), and every header comment line, as written.
`records.record_metadata(record)` reads them back. The notes in those comments name each record's
arrhythmias, so they stay in metadata, which is never model input.

### Annotations

Every reference label is a Record annotation, unless noted:

| Key | Span | Value |
| --- | --- | --- |
| `age` | none | Years, absent in records 103 and 219 |
| `sex` | none | `F` or `M` |
| `medications` | none | The second comment line, as written, such as `Digoxin` or `None` |
| `beat` | point | The beat's code, such as `N` |
| `rhythm` | interval | The rhythm label, such as `(AFIB`, up to the next change or the end |
| `signal_quality` | interval, on each Signal | `clean`, `noisy` or `unreadable`, up to the next change or the end |
| `event` | point | `x`, `\|`, `[` or `]` |
| `comment` | point | `MISSB`, `PSE` or `TS` |

- **Rhythm and signal quality episodes.** A change holds until the next change of its kind, or
  until the end of the record, as `dbnotes.c` counts durations. Before the first rhythm change, 0
  to 229 samples into each record, no rhythm is stated, and nothing fills that time. Signal
  quality is stated only from a record's first `~`; 10 records have none.
- **Signal quality decoding.** As `dbnotes.c` reads the subtype: bit `0x01 << i` marks signal `i`
  noisy, bit `0x10 << i` with it marks it unreadable, and `-1` marks both unreadable. The subtype
  as written is the annotation's `metadata["subtype"]`.
- **Subtypes.** Some `V`, `F` and `|` labels have a subtype from 1 to 9, which the directory
  doesn't document. It is kept as `metadata["subtype"]` where it isn't 0.

The dataset also holds `source_url`, `source_sha256`, `beat_vocabulary` (the 16 beat codes, in the
column order of the directory's beat table) and `rhythm_vocabulary` (the 19 rhythm labels
`dbnotes.c` names, 15 of which occur).

### Tasks

The authors score an algorithm on whole records: it writes an annotation file, and WFDB's
comparators match it against the reference. Each record has one task for each output they score,
and each beat a classification task:

| Task | Count | Type | Target |
| --- | --- | --- | --- |
| Beat annotation (`bxb`) | 48 | `TemporalLocalizationTask`, exhaustive | `target_annotations`: the Record's `beat` annotations |
| Rhythm annotation (`epicmp`) | 48 | `TemporalLocalizationTask`, exhaustive | `target_annotations`: the Record's `rhythm` annotations |
| Beat classification | 109,966 | `ClassificationTask`, `target_schema` `beat_vocabulary` | The beat's code, with `scope` the beat's point |

- **Beat annotation.** `bxb` matches each beat within 150 ms and compares the labels of matched
  beats, so the target is the labelled beats. The beats alone are the QRS detection target.
- **Rhythm annotation.** `epicmp` takes episodes of atrial fibrillation (`(AFIB`), atrial flutter
  (`(AFL`) and ventricular flutter from the rhythm labels. The target is every rhythm episode.
- **Beat classification.** Neither comparator scores a beat given its reference location; these
  tasks are our choice, for the beat classification PhysioNet's usage notes list. The scope names
  no signal, as the cardiologists read both leads, and the whole Record stays available.
- **Not tasks.** `rxr` compares runs of ectopic beats, which it derives from the beat labels, so
  the beat annotation task covers it.
- **No split, no prompt.** The release has no split, so no task has one. PhysioNet asks that
  records 201 and 202 fall in the same partition; group by `subject_id` to do so. The database
  asks no question in words, so no task has a prompt.

A torch item holds every task of its Record, targets included, so a model given one Record's
items sees the reference labels of the others.

## Scoring as the authors do

EC57 scoring, as WFDB's `bxb` implements it, differs from scoring every beat as written:

- **Learning period.** `bxb`, `rxr` and `epicmp` start at 5:00 by default. Beats before then are
  not scored.
- **Paced records.** WFDB's `ecgeval` scores `mitxlist`, the database without its paced records
  (102, 104, 107 and 217). The AAMI recommended practice excludes them from reporting.
- **Ventricular flutter.** `bxb` skips the labels from each `[` to its `]`, so the 472 `!` waves,
  all in record 207, are never scored. The directory's notes call `!` and `x` non-beats.
- **Classes.** `bxb` maps each code to an AAMI class with its `amap`:

| AAMI class | Codes |
| --- | --- |
| N | `N`, `L`, `R` |
| S | `A`, `a`, `J`, `S`, `j`, `e`; `N` unless SVEB statistics are asked for |
| V | `V`, `E` |
| F | `F` |
| Q | `Q`, `/`, `f` |
| not a beat | `!` and every non-beat code |

The targets keep the cardiologists' codes. Grouping them is the scorer's step.

## How the connector builds it

Each step's output carries what the next step needs:

```
download  mitdb zip ── ensure_archive, sha256 ──► mit-bih-arrhythmia-database-1.0.0/

parse     RECORDS ──► record names, each with .hea, .dat and .atr
          <record>.hea ──► header.read_header ──► RecordHeader
          <record>.atr ──► annotations.read_reference ──► ReferenceAnnotations
                            (beats, rhythm and quality episodes, events, comments)

build     RecordHeader + ReferenceAnnotations ──► records.build_record
          ──► ParsedRecord: the lazy Record, and its beat and rhythm annotations

check     the counts ──► release.PINNED_COUNTS

tasks     ParsedRecord ──► BeatAnnotationTask, RhythmAnnotationTask,
                           one BeatClassificationTask per beat
```

`connector.py` adds the provenance annotations and hands the extracted directory to
`MitdbDatasetLoader`, whose `populate` adds the Records and streams the tasks. The signals reread
their `.dat` through the PhysioNet base's lead loader, which reads format 212 with `wfdb`. A
malformed file raises `TimeFFormatError` and names the file. Counts that differ from the pinned
archive raise `TimeFValidationError`.
