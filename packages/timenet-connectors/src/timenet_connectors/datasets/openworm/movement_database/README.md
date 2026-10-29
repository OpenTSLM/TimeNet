# Open Worm Movement Database: single-worm strains

`openworm/movement-database` holds the strain benchmark that Javer et al. built on the Open Worm
Movement Database ([Javer et al., ECCV Workshops 2018][paper]): 10,476 fifteen-minute videos of
single *C. elegans* worms, each tracked as a 49-point skeleton per frame, and the task of naming the
worm's strain from its posture over time. The database itself is described in
[Javer et al., Nature Methods 2018][database].

The connector reads two sources:

- **The videos**, from the Zenodo community [open-worm-movement-database][zenodo], one record per
  video, under CC BY 4.0. Each record's `_features.hdf5` is pinned by the MD5 Zenodo publishes for
  it, listed in `zenodo_index.csv.gz`.
- **The benchmark**, from the authors' repository [ver228/worm-ts-classification][code] at
  `6d4d6b6`: its `SWDB_fold_dict.p` (each video's fold) and `SWDB_straindict.p` (the classes),
  pinned by SHA-256. The repository states no license; the connector downloads these two files at
  build time and copies no code.

[paper]: https://doi.org/10.1007/978-3-030-11024-6_35
[database]: https://doi.org/10.1038/s41592-018-0112-1
[zenodo]: https://zenodo.org/communities/open-worm-movement-database
[code]: https://github.com/ver228/worm-ts-classification/tree/6d4d6b6f04e05a93baea7a5a7550a7a180b60c94

## The source data

Each Zenodo record holds three files named after the video's base name, such as
`mec-12 (e1605) on food R_2010_10_01__12_25___3___3`:

| File | Content |
| --- | --- |
| `<base name>.hdf5` | The masked video, hundreds of MB. Not read |
| `<base name>.wcon.zip` | The skeletons in WCON, a JSON tracking format. Not read |
| `<base name>_features.hdf5` | Tierpsy Tracker's output, about 20 MB. Read |

The features file holds:

- **`experiment_info`**, a JSON string: the strain, gene, allele, sex, ventral side, timestamp,
  lab, food and more. The record's Zenodo description states the same facts.
- **`coordinates/skeletons`**, float32 of shape `(frames, 49, 2)`: per frame, 49 points from head to
  tail, each `x` and `y` in micrometers, in the plate's coordinates. A frame the tracker gave no
  skeleton, or whose skeleton its filter rejected, is NaN in all 98 values. Reading a video checks
  that no frame is partly NaN.
- **`features_timeseries`**, one row per frame. Its `timestamp` column numbers the frames 0 to
  n − 1 with no gap, and its attributes give `fps`, `xy_units: micrometers` and
  `time_units: seconds`. The frame rate is one over a whole number of 0.1 ms, such as 33.3 ms
  (30.03 fps) or 38.9 ms (25.71 fps), stored as a float.
- Contours, per-frame features, events and per-video summaries, which the benchmark doesn't use.

## The benchmark, as the authors define it

The paper (section 2) restricts the database to "15 minutes videos of young adults on food, and
where at least 50% of the frames were successfully skeletonized", which gives "10476 videos of
individual worms divided between 365 different classes". A class is a strain, "except N2 where
hermaphrodites and males are considered as separated classes", and "only classes that have at
least six videos" are kept. The authors' code publishes the outcome, not the selection:

- `SWDB_fold_dict.p` maps each of the 10,476 base names to fold 0, 1 or 2.
  `flow_helper.save_folds_dict` deals each strain's videos into the three folds in turn.
- `SWDB_straindict.p` maps each of the 365 classes, such as `AQ1031_XX` or `N2_XO`, to its index,
  in sorted order. `XX` is a hermaphrodite, `XO` a male; only N2 has an `XO` class.
- `SkelEmbeddingsFlow` tests on `fold_n_test = 0` by default, and `Trainer` never changes it. The
  paper: "Two thirds of the data are used for training and the rest for testing".

The selection itself, the database query and `strains2process.csv`, isn't published, so the folds
are the definition of the benchmark's videos.

## The TimeNet dataset

### Records

One Record per video of the folds: 10,476. Each has one Source, `Tierpsy Tracker`, with one lazy
signal, `skeleton`: per frame, a `(49, 2)` float32 tensor in micrometers, whose dimensions are
`point` (0 is the head, 48 the tail) and `coordinate` (0 is `x`, 1 is `y`), exactly as the file
stores it. A frame without a skeleton is a null timestep, so a reader sees the gap; the build fills
nothing. The axis is regular at the file's frame period in whole microseconds.

`start_time` is the experiment's `timestamp`, which carries its UTC offset. The one Record
annotation is `ventral_side`, `clockwise` or `anticlockwise`, or `unknown` for 7 videos: the
skeleton runs head to tail, so this side tells ventral bends from dorsal ones.

Record metadata is provenance and never model input: `base_name`, `zenodo_record` and the authors'
`fold`. Read it back with `records.record_metadata(record)`. The source names no worm, so
`subject_id` is unset.

The strain, sex, gene and allele are the class, so none of them is a Record annotation. The base
name spells out the gene and allele, and the Zenodo record's title the strain, so a model must not
see the metadata.

The skeletons are tensors, which only the Zarr values backend stores, so the connector sets
`values_backend = "zarr"`.

### Dataset annotations

| Key | Value |
| --- | --- |
| `source_url` | The Zenodo community |
| `benchmark_url` | The authors' repository at the pinned commit |
| `strain_vocabulary` | The 365 classes, in the authors' index order; every task's `target_schema` |

### Tasks

One `ClassificationTask` per video:

| Field | Value |
| --- | --- |
| `inputs` | The video's Record |
| `targets` | Its class, such as `AQ1031_XX` |
| `target_schema` | `strain_vocabulary` |
| `split` | `test` for fold 0 (3,609 videos), `train` for folds 1 and 2 (3,488 and 3,379) |

There is no scope: the authors test each video on all its frames. There is no prompt: the benchmark
asks no question in words.

### What the build leaves out

- **The authors' preprocessing.** Their model doesn't read the skeletons as published. Section 3.1:
  the skeletons are resampled in time to one every 0.04 s (25 fps); gaps under 0.25 s are linearly
  interpolated and longer ones set to zero; both space and time are smoothed with a Savitzky–Golay
  filter. The code that does this, which writes their `_interpolated25.hdf5` files, isn't in the
  repository. Filling and smoothing would invent values, so the build stores the raw skeletons.
- **The inputs the paper compares.** `collect_data/embeddings_helper.py` turns skeletons into 48
  tangent angles (`_h_angles`), or 6 eigenworms by projecting the angles onto the first rows of
  `collect_data/pca_components.npy` (`_h_eigenworms`). Both are functions of the stored skeletons.
- **Training-time filters.** `SkelEmbeddingsFlow` drops trajectory pieces under 250 frames and
  strains left with fewer than three videos. These act on the unpublished resampled files, so the
  build keeps every folded video.
- The masked videos, WCON files, contours, Tierpsy features and the experiments' other facts
  (lab, food, who, tracker), which the benchmark doesn't use. Each Record's Zenodo record has them.

**Scoring as the authors do.** They report top-1 and top-5 accuracy and the F1 score on fold 0,
per video (Table 1: 58.44 % top-1 with angles, 49.40 % with eigenworms). Read each target with
`timenet.types.text.read_value`.

## How the connector builds it

```text
download   SWDB_fold_dict.p, SWDB_straindict.p ── sha256 ──► benchmark/
           zenodo_index.csv.gz ──► IndexedVideo per video
             each _features.hdf5 ── md5 ──► videos/<record>.hdf5

parse      SWDB_fold_dict.p ──► folds: base name → 0, 1 or 2
           SWDB_straindict.p ──► classes, in index order
           zenodo_index.csv.gz ──► IndexedVideo
                                     base_name, record_id, md5,
                                     size, strain, sex
             join on base name with the folds

for each   <record>.hdf5 ── read_video ──► Video
IndexedVideo                                 experiment (base name,
                                             strain, sex, ventral side,
                                             timestamp), n_frames,
                                             period_us
           build_record(IndexedVideo, Video, fold) ──► Record
             one lazy skeleton signal
           class_of(strain, sex) ──► target

populate   dataset annotations, then the Records, then the task stream:
           StrainTask per video ──► ClassificationTask

write      load_skeletons(Video): the skeletons, null where NaN
```

| Module | Holds |
| --- | --- |
| `connector.py` | The download, the provenance annotations, and `convert` |
| `release.py` | The Zenodo files: their URL, layout, units, MD5 check, and the pinned counts |
| `index.py` | The pinned index of the videos' Zenodo records |
| `benchmark.py` | What the authors fix: their repository, folds, classes and test fold |
| `vocabulary.py` | The closed sets: sexes and ventral sides |
| `keys.py` | Every annotation key the connector writes |
| `videos.py` | One features file: `Experiment`, `Video`, `read_video`, `read_skeletons` |
| `records.py` | A video's Record, its lazy skeletons, and its typed metadata |
| `tasks.py` | `StrainTask`, which becomes a `ClassificationTask` |
| `loader.py` | `OpenWormDatasetLoader`, which runs the parse and streams the tasks |

### The pinned index

`zenodo_index.csv.gz` has one row per video of the folds: `base_name`, `record_id`, `md5`, `size`,
`strain` and `sex`. It was built on 2026-10-29 from the Zenodo REST API
(`/api/records?communities=open-worm-movement-database`), which then listed 15,001 records. Each
record's description gives its `base_name`, `strain` and `sex`, and its file list the features
file's MD5 and size. Every one of the 10,476 base names of the folds matched exactly one record,
whose features file is named `<base name>_features.hdf5`. All 10,476 records are CC BY 4.0.
The 365 classes have 6 to 1,448 videos (N2 hermaphrodites), as the paper's "at least six" says.

### Building

The full download is 188 GB, from 10,476 Zenodo records. Zenodo limits guest requests, so a
run can stop with HTTP 429; a rerun keeps the files already checked and fetches only the rest.

## Known issues upstream

- **Model selection on the test fold.** `Trainer` keeps `model_best.pth.tar` by the loss on fold
  0, the fold it reports, and no fold is held out for validation. No task is `validation`, and a
  fair comparison selects models on a part of the training folds.
- **Batch effects.** A strain's videos cluster on a few recording days: a class's median 22 videos come from a median of 3 recording days,
  and a day holds a median of 8 classes. The authors'
  `merge_by_week` option groups videos by week to study this. `start_time` carries the day, so a
  model shown it can learn the batch, not the behaviour.
