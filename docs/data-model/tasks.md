---
icon: lucide/target
description: "Tasks: a prompt, the inputs it gives a model, and the target it expects."
tags:
  - guide
  - concepts
---

# Tasks

A task says what a model is asked to do with some records, and what the right answer is. It has
three parts: a `prompt`, an ordered list of `inputs`, and an ordered list of `target` items. Both
lists mix text and [records](records.md), so one shape covers classification, question answering,
captioning, forecasting, and everything else a corpus asks for.

```python
from timenet.control_plane import Task

Task(
    id="diagnosis-record-123",
    prompt="Diagnose this patient.",
    inputs=[record],
    target=["The patient is stable."],
)
```

Order is preserved on both sides, so a task with two inputs keeps which one came first. That matters
whenever the question is about a comparison.

```python
Task(
    id="compare-001",
    prompt="Which recording shows the higher vibration amplitude?",
    inputs=[record_a, record_b],
    target=["The first."],
)
```

A target can hold a record too. Forecasting is a task whose input is the context window and whose
target is the horizon, both of them records.

## Referring to a record by id

A streaming build yields its records and drops them, so by the time it builds the tasks the record
objects are gone. `RecordRef` names one by id instead:

```python
from timenet.control_plane import RecordRef, Task

Task(
    id="caption-000123",
    prompt="Describe this window.",
    inputs=[RecordRef("window-000123")],
    target=["A slow drift upward, then a step."],
)
```

The reference stays a record reference rather than degrading into a text item, and the writer still
rejects an id that names no record. `dataset.add_task(task)` also adds any `Record` object the task
holds that is not in the dataset yet.

## Tasks point at records

A task names the records it uses; a record does not list its tasks. There is no reverse index,
because the forward table already holds the answer: `reader.tasks_for_record(external_id)` reads it
directly. That removes a table, the code that writes it, and the consistency check that would guard
it against drift.

Reading a task rebuilds every record it names in full:

```python
task = reader.task("diagnosis-record-123")
task.prompt                       # "Diagnose this patient."
task.inputs[0].signals()          # the record's signals, spec and axis resolved
task.target                       # ["The patient is stable."]
```

`render_task(reader, external_id)` turns all of that into the text a training example starts from:
the prompt, the inputs and targets in order, each record's source tree, and the annotations at every
level.

## Tasks are optional

A record can be added on its own. An unlabelled pretraining corpus is records with no tasks, and it
needs no invented prompt to exist.

## Annotations on a task

A task carries [annotations](annotations.md) like any other object, which is where the metadata
about the task goes: its kind, its split, who wrote it, how confident they were.

```python
task.annotate(Annotation.static(name="task_kind", value="diagnosis"))
```

## Context or target

The same fact can be either, and which one it is depends on the task, not on the storage. A
`condition=healthy` annotation on a record is context when the model is asked to forecast, and the
target when it is asked to classify. Put the fact on the record as an annotation, then let each task
decide what to do with it.
