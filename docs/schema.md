# Schema

The data model the connector produces and the writer serializes. The builder ([TimeFDataset](timef-builder.md)) populates it in memory, the writer turns it into files on disk.

TODO: this is a qualitative explanation to wrap your head around the schema, i will also add the classes that define it on code

---

## Conceptual ladder

```
DatasetDescriptor                       (declared at construction, immutable)
└── Sample                              (one per source recording)
    ├── Signal                          (per signal dataframe, own time axis)
    │   └── Channel                     (column in signal dataframe)
    └── Annotation                      (with optional windows + signal targeting)
```

A `DatasetDescriptor` declares the dataset's identity and which signals exist. Samples are added by the connector during `convert_to_timef` and represent one source recording each. Each sample carries one DataFrame per signal. Annotations attach to one or more samples.

---

## Specs vs instances

**Specs** declare types, what kinds of signals, sensors, and annotation tasks exist. **Instances** are the per-sample data created by the connector.

| Spec             | Instance     | Lifecycle                                                           |
| ---------------- | ------------ | ------------------------------------------------------------------- |
| `SignalSpec`     | `Signal`     | Spec declared on the descriptor; one instance per sample.           |
| `AnnotationSpec` | `Annotation` | Spec interned by `(task, schema_, domains)`; one instance per call. |
| `SensorSpec`     | (n/a)        | Optional physical-sensor metadata, declared once per dataset.       |

Specs are declared once per dataset. Instances are created per call. The writer interns repeated `(task, schema_, domains)` triples so every annotation that shares a triple shares one `AnnotationSpec` id in the manifest.

---

## Channels and signals

| Term        | Meaning                                                                             |
| ----------- | ----------------------------------------------------------------------------------- |
| **Channel** | A single recorded series within a signal. Channels of one signal share a time axis. |
| **Signal**  | A kind of measurement: same sensor, same units, same sampling rate.                 |

| Dataset                   | Sample shape                                                           |
| ------------------------- | ---------------------------------------------------------------------- |
| 12-lead ECG (e.g. PTB-XL) | 1 signal `"ecg"`, 12 channels                                          |
| MIT-BIH Arrhythmia        | 1 signal `"ecg"`, 2 channels (`MLII`, `V1`)                            |
| Wearable session          | 3 signals (`ppg`, `accel`, `temp`), independent rates                  |
| Polysomnography (sleep)   | 5+ signals (`eeg`, `eog`, `emg`, `airflow`, `spo2`), independent rates |

Signals always live on a sample. There is no notion of a signal that exists outside a sample.

---

## Sample granularity

**One source recording = one stored sample.** This rule holds across all connectors so that the same source produces the same dataset shape regardless of who wrote the connector.

| Source                         | Stored samples per dataset            |
| ------------------------------ | ------------------------------------- |
| MIT-BIH (48 records)           | 48 samples, one per `.hea` record     |
| Sleep-EDF (153 nights)         | 153 samples, one per overnight `.edf` |
| Wearable corpus (10k sessions) | 10 000 samples, one per session       |

Annotations carry `windows` to label specific spans within a sample. Downstream consumers can materialize fixed-length training windows in the data-loader layer either using annotations or other criteria.

---

## Multi-rate signals

Each signal on a sample carries its own `time_s` column at its native sampling rate. The PSG case:

```
sample("subj_42_night_1")
    ├── signal "eeg"      → time_s @ 256 Hz, 6 channels
    ├── signal "eog"      → time_s @ 256 Hz, 2 channels
    ├── signal "emg"      → time_s @ 256 Hz, 1 channel
    ├── signal "airflow"  → time_s @  32 Hz, 1 channel
    └── signal "spo2"     → time_s @   1 Hz, 1 channel
```

---

## Annotations

Annotations attach to samples. They optionally:

- Carry `windows: list[tuple[float, float]]` : time spans, in the targeted signals' time axis, that the annotation pertains to. `None` means the annotation covers the full span.
- Name `signals: list[str]` : specific signals on the sample the annotation targets. `None` means whole-sample.

The `Annotation.samples` field is a `list[Sample]` (length 1 today) — the shape leaves room for a future cross-sample API without changing the schema.

| Pattern         | Example                                                           | Shape                                         |
| --------------- | ----------------------------------------------------------------- | --------------------------------------------- |
| Sample-level    | One overall diagnosis on a 30-second 12-lead ECG                  | `signals=None, windows=None`                  |
| Windowed        | Beat label at 0.42 s on MIT-BIH                                   | `signals=["ecg"], windows=[(0.21, 0.63)]`     |
| Signal-targeted | Apnea event on PSG airflow, 10 s span                             | `signals=["airflow"], windows=[(7220, 7230)]` |

Annotation IDs are auto-generated as `f"ann_{counter}"` (a dataset-wide running index) and exposed on the returned `Annotation` object.

---
