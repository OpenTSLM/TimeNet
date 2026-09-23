---
icon: lucide/target
description: "Tasks connect input Records to ordered targets, annotations, and other stored objects."
tags:
  - guide
  - concepts
---

# Tasks

A Task describes one modeling problem. It holds ordered input Records and one representation of the
expected output.

```python
from timenet.types import ClassificationTask

task = ClassificationTask(
    inputs=(record,),
    targets=("faulty",),
    target_schema="condition",
)
dataset.add_task(task=task)
```

The concrete class identifies the task type. `ClassificationTask` and `AnswerTask` are different
task types, even when both contain string targets.

## Common fields

Every Task supports these fields:

| Field | Purpose |
| --- | --- |
| `id` | Stable Task identity. |
| `inputs` | Ordered input Records. |
| `targets` | Ordered inline values or stored objects. |
| `prompt` | Optional instruction or question. |
| `scope` | Optional input region. |
| `input_annotations` | Attached Annotation occurrences supplied as context. |
| `target_annotations` | Attached Annotation occurrences used as the answer. |
| `rationale` | Optional explanation associated with the answer. |
| `from_tasks` | Parent Tasks from which this Task derives. |
| `annotations` | Annotations that describe the Task itself. |
| `metadata` | JSON-compatible Task details. |

`targets` can contain strings, numbers, Booleans, Records, Signals, or Spans. `None` means that
`target_annotations` supplies the answer. An empty tuple is an explicit empty answer.

A Task cannot set both `targets` and `target_annotations`.

## Register Tasks

Register all referenced objects before you register the Task:

```python
dataset.add_record(record=record)
dataset.add_task(task=task)
```

`add_task()` validates Record, Signal, Annotation, Span, and parent-Task references. It also attaches
the Task ID to each input Record.

Use `add_tasks()` for an atomic batch. A parent in `from_tasks` can appear in the same batch.

For a large task corpus, use `set_task_stream()`. The writer validates each streamed Task before it
writes the Task. A stream does not keep the complete task set in memory. TimeNet therefore cannot
check duplicate task IDs or `from_tasks` relationships across streamed Tasks.

## Scope an input

`scope` restricts the region supplied to the model:

```python
from timenet.types import ClassificationTask, TimeInterval

task = ClassificationTask(
    inputs=(record,),
    scope=TimeInterval.seconds(
        5.0,
        8.0,
        time_series_ids=("vibration",),
    ),
    targets=("fault_episode",),
)
```

Use `TimePoint` or `TimeInterval` for a timeline Signal. Use `StepPoint` or `StepInterval` for an
ordinal Signal.

## Built-in task types

TimeNet defines eight task classes:

| Class | Modeling operation | Typical target items |
| --- | --- | --- |
| `ClassificationTask` | Select one or more labels. | Strings. |
| `AnswerTask` | Answer a question or caption the inputs. | Strings. |
| `ScalarPredictionTask` | Predict typed quantities. | Integers or floats. |
| `TemporalLocalizationTask` | Locate events or regions. | Points or intervals. |
| `ForecastingTask` | Predict future values. | Records, Signals, or intervals. |
| `TSEditingTask` | Transform input time-series data. | Records or Signals. |
| `TSGenerationTask` | Generate time-series data from a prompt. | Records or Signals. |
| `TSCorrespondenceTask` | Match inputs to candidates. | Records or Signals. |

The target collection stays uniform across these classes. Each subclass adds only the configuration
that its operation needs.

### Classification

```python
ClassificationTask(
    inputs=(record,),
    targets=("healthy",),
    target_schema="condition",
)
```

`target_schema` can name a registered Annotation that holds the label vocabulary.

### Answer and rationale

```python
AnswerTask(
    inputs=(record,),
    prompt="Does this trace show a bearing fault?",
    rationale="The repeated impacts match a bearing fault.",
    targets=("Yes",),
)
```

A missing prompt makes the Task an unprompted caption or description.

### Scalar prediction

```python
ScalarPredictionTask(
    inputs=(record,),
    targets=(62.0,),
    unit="bpm",
    target_name="mean_heart_rate",
)
```

The class keeps the quantity name and physical unit separate from the numeric target.

### Temporal localization

```python
from timenet.types import (
    LocalizationMode,
    TemporalLocalizationTask,
    TimePoint,
)

TemporalLocalizationTask(
    inputs=(record,),
    prompt="Locate all R-peaks in lead II.",
    mode=LocalizationMode.SPARSE,
    targets=(
        TimePoint.seconds(1.20, time_series_ids=("II",)),
        TimePoint.seconds(2.05, time_series_ids=("II",)),
    ),
)
```

`SPARSE` says that unmarked time is unspecified. `EXHAUSTIVE` says that the targets describe the
complete region of interest.

### Forecasting, editing, and generation

These Tasks can target stored Records or Signals directly:

```python
ForecastingTask(
    inputs=(history,),
    targets=(future,),
)

TSEditingTask(
    inputs=(raw_record,),
    prompt="Remove the baseline wander.",
    targets=(clean_record,),
)

TSGenerationTask(
    prompt="Generate 10 seconds of a 150 bpm ECG.",
    targets=(synthetic_record,),
)
```

A generation Task can have no input Records.

### Correspondence

```python
TSCorrespondenceTask(
    inputs=(query,),
    candidate_records=(record_a, record_b, record_c),
    targets=(record_b,),
)
```

An empty `candidate_records` tuple means that the candidate pool is dataset-wide.

## Compose Tasks

`from_tasks` records derivation between Tasks:

```python
label = ClassificationTask(
    inputs=(record,),
    targets=("faulty",),
)
answer = AnswerTask(
    inputs=(record,),
    prompt="Is this machine healthy?",
    targets=("No",),
    from_tasks=(label,),
)
dataset.add_tasks(tasks=(label, answer))
```

TimeNet rejects missing parents, self-references, and derivation cycles.
