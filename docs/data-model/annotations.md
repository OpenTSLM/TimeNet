---
icon: lucide/highlighter
description: "Annotations attach typed context to TimeF objects and optional timeline regions."
tags:
  - guide
  - concepts
---

# Annotations

An `Annotation` stores reusable content. An annotation occurrence attaches that content to a Dataset,
Task, Record, Source, or Signal.

```python
from timenet.types import Annotation

condition = Annotation(
    key="condition",
    value="healthy",
    description="Condition assigned by the source dataset.",
)
occurrence = record.annotate(condition)
```

`annotate()` returns the attached occurrence. A Task must refer to this occurrence, not the unattached
content object.

## Content and occurrence fields

Reusable content includes these fields:

- `key`, `value`, `unit`, and `description`
- `id`, which identifies the reusable content
- `metadata`, which stores JSON-compatible content details

Each occurrence adds its own identity, source, confidence, and metadata. This design lets several
objects share one long description without sharing occurrence details.

## Static context

An Annotation without a `span` describes its complete owner:

```python
record.annotate(Annotation(key="age", value=54, unit="years"))
record.annotate(Annotation(key="condition", value="healthy"))
```

Use `metadata` for untyped implementation details. Use an Annotation when consumers need a named,
typed fact.

## Points and intervals

A `TimePoint` marks one offset. A `TimeInterval` covers a half-open range, `[start, end)`.

```python
from timenet.types import Annotation, TimeInterval, TimePoint

record.annotate(
    Annotation(
        key="impact",
        span=TimePoint.seconds(
            4.2,
            time_series_ids=("vibration",),
        ),
    )
)

record.annotate(
    Annotation(
        key="fault",
        value="bearing fault",
        span=TimeInterval.seconds(
            5.0,
            8.0,
            time_series_ids=("vibration",),
        ),
    )
)
```

Leave `time_series_ids=None` when the span covers the complete owner. Supply Signal IDs to restrict
the span to specific Signals.

Time spans use whole microseconds on the source recording timeline. Step spans are reserved for
Tasks on ordinal Signals.

## Valid values

An annotation value can be a string, integer, float, Boolean, or list of strings. A list can define a
closed label vocabulary.

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
