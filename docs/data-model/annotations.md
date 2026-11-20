---
icon: lucide/highlighter
description: "Annotations attach typed context to TimeF objects and optional timeline regions."
tags:
  - guide
  - concepts
---

# Annotations

An `Annotation` stores reusable content. An annotation occurrence attaches that content to a Dataset,
Task, Record, Source, Signal, or label set.

```python
from timenet.dataset import Record, TimeFDataset
from timenet.types import Annotation

condition = Annotation(
    key="condition",
    value="healthy",
    description="Condition assigned by the source dataset.",
)
record = Record(sources=(source,), annotations=[condition])
```

`key` can be positional: `Annotation("condition", value="healthy")`. All other fields are keyword-only.

## Attach annotations

Pass `annotations=[...]` when you build the owner. A dataset, a Task, a Record, a Source, and a
Signal all take this argument. It is a list, or any other iterable, of `Annotation` objects:

```python
dataset = TimeFDataset(
    metadata=metadata,
    annotations=[
        Annotation("source_repository", value="org/name"),
        Annotation("source_revision", value="4f2a9c1"),
    ],
)
```

The dataset attaches each annotation in order, as `dataset.annotate()` does. A dataset annotation
cannot have a `span`. Every owner attaches a new occurrence of each annotation that you pass, as
`annotate()` does. So you can pass one `Annotation` object to several owners.

Use `annotate()` in these cases:

- The owner exists already, for example a Record that you import from a parent dataset.
- A Task refers to the annotation. Keep the occurrence that `annotate()` returns.

```python
occurrence = record.annotate(condition)
```

`annotate()` returns the attached occurrence. A Task must refer to this occurrence, not the unattached
content object. For an annotation that you passed to the constructor, take the occurrence from the
owner, for example `record.annotations[0]`.

## Content and occurrence fields

Reusable content includes these fields:

- `key`, `value`, `unit`, and `description`
- `metadata`, which stores JSON-compatible content details

Each occurrence adds its own `source`, `confidence`, and `occurrence_metadata`. This design lets
several objects share one long description without sharing occurrence details. The writer stores
equal content once. Neither the content nor an occurrence has an ID in Python.

## Reuse content

`Annotation` is final, so it can't be subclassed. To reuse content, keep a constant. For content
that varies, write a builder function that returns an `Annotation`:

```python
from typing import Literal

from timenet.types import Annotation

# A constant to reuse
HEALTHY_ANNOTATION = Annotation("condition", value="healthy")


def quality_annotation(level: Literal["low", "medium", "high"]) -> Annotation:
    return Annotation("quality", value=level)


record.annotate(HEALTHY_ANNOTATION)
record.annotate(quality_annotation("high"))
```

Every occurrence of a constant shares its content. A builder call creates a new `Annotation`, and
calls with equal arguments give equal content. The writer stores equal content once either way, and
the reader rebuilds a plain `Annotation` that equals the one written. Large text read back from a
release is compared by where it is stored, without a read. So annotations read from one content
row are stored as one row again, but stored text and the same text in memory become two rows.

## Static context

An Annotation without a `span` describes its complete owner:

```python
record = Record(
    sources=(source,),
    annotations=[
        Annotation("age", value=54, unit="years"),
        Annotation("condition", value="healthy"),
    ],
)
```

Use `metadata` for untyped implementation details. Use an Annotation when consumers need a named,
typed fact.

## Points and intervals

A `TimePoint` marks one offset. A `TimeInterval` covers a half-open range, `[start, end)`.

```python
from timenet.types import Annotation, TimeInterval, TimePoint

vibration.annotate(
    Annotation(
        key="impact",
        span=TimePoint.seconds(
            4.2,
        ),
    )
)

vibration.annotate(
    Annotation(
        key="fault",
        value="bearing fault",
        span=TimeInterval.seconds(
            5.0,
            8.0,
        ),
    )
)
```

The owner determines scope. A Signal annotation covers that Signal. A
[Source annotation](sources.md#annotate-a-source) covers its subtree. A Record annotation covers
the Record. An annotation span cannot set `time_series`, because the owner sets the scope.

Use `source.select(signals=(ecg, spo2)).annotate(annotation)` for a subset. Each selected Signal gets
a separate occurrence with shared content. Other Signals remain unchanged.

Time spans use whole microseconds on the Record timeline. Step spans are reserved for
Tasks on ordinal Signals.

## Label set annotations

A `LabelSet` takes annotations like the other owners. Pass them to the constructor, or call
`annotate()` later. Use them for a name, a description, or other context. No key is required.

```python
from timenet.types import Annotation, LabelSet

SLEEP_STAGE = LabelSet(
    ["W", "N1", "R"],
    annotations=[Annotation("name", value="sleep_stage")],
)
SLEEP_STAGE.annotate(Annotation(key="description", value="Sleep stage of one epoch."))
```

A label set annotation has no span. Annotations are part of the identity of a set. Two sets with the
same labels and different annotations are two sets, and TimeF stores each of them. TimeF stores sets
with the same labels and the same annotations once.

## Valid values

An annotation value can be a string, integer, float, Boolean, or list of strings. For example, a
list of strings can hold the names of activities in a recording. To define the allowed labels for a
classification target, use a `LabelSet`. See [Tasks](tasks.md).

An Annotation must have a value, a span, or both. A marker can omit its value when its span carries
the meaning.

## Use annotations in Tasks

A Task can use attached occurrences as input context or as its expected output:

```python
task = AnswerTask(
    inputs=(record,),
    prompt="What condition does the source report?",
    target_annotations=(occurrence,),
)
```

Use `input_annotations` for context. Use `target_annotations` for the answer. A Task cannot combine
inline `targets` with `target_annotations`.

The occurrences must lie on the dataset itself or on a Record that the Task references, as
[Records that a Task references](tasks.md#records-that-a-task-references) describes. `add_task()`,
`add_tasks()`, and the check of streamed Tasks reject an occurrence on any other Record.
