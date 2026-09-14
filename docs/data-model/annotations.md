---
icon: lucide/highlighter
description: "Annotations: statements attached to any entity, placed in time as static, point, or interval."
tags:
  - guide
  - concepts
---

# Annotations

An annotation is one statement about an object, plus where in time it applies. It is a `(name,
value, unit)` triple with a span type, and it attaches to a dataset, a [task](tasks.md), a
[record](records.md), a source, or a [signal](signals.md).

```python
from timenet.control_plane import Annotation

Annotation.static(name="condition", value="healthy")
Annotation.point(name="impact", value="shock", at_us=4_200_000)
Annotation.interval(name="fault", value="bearing fault",
                    start_us=5_000_000, end_us=8_000_000)
```

Every value is stored as text. `unit` says what a quantity is measured in. `provenance` says who or
what made the statement, and `confidence` how sure they were, between 0 and 1.

Attach one with `annotate`, which returns the object so calls chain:

```python
record.annotate(Annotation.static(name="age", value=54, unit="years"))
signal.annotate(Annotation.point(name="lead_status", value="Lead fell off",
                                 at_us=6_000_000))
```

A record has no metadata field, so record-level context belongs here: a subject's attributes, the
machine id, its firmware version, an operating mode, a condition label. The fact then travels with
every window drawn later from the record.

## Three time shapes

**Static** applies to the whole object and has no place in time. It is the default.

<figure markdown="span">
  ![A band over the whole recording marking a record-level fact](../assets/figures/annotation-static.svg)
</figure>

**Point** applies at one instant, `at_us` microseconds from the record's relative zero. This fits
discrete events: a shock, a valve actuation, a detected spike. Points are cheap to store, detectors
produce them in bulk, and they anchor downstream windowing.

<figure markdown="span">
  ![A marker at one time offset on a signal](../assets/figures/annotation-point.svg)
</figure>

**Interval** covers `[start_us, end_us]`, and building one whose end precedes its start raises
`TimeFValidationError`. This is the most common shape, and its `value` can carry a full reading of
what happens in that window rather than a label.

<figure markdown="span">
  ![A shaded window on a signal](../assets/figures/annotation-interval.svg)
</figure>

## One payload, many attachments

The same `Annotation` object attached to many objects is stored once. The writer keeps the `(name,
value, unit)` payload in one table and one short attachment row per use, so `patient_sex=male` on
100,000 records costs one payload and 100,000 rows rather than 100,000 copies of the string.

```python
male = Annotation.static(name="patient_sex", value="male")
for record in records:
    record.annotate(male)
```

Time placement lives on the attachment, not the payload. The same statement can therefore be static
on one object and an interval on another.

## Across signals

An annotation is not tied to one signal. A single event can span a vibration sensor, a temperature
probe, and a current sensor together; the shared timing is what says it is one physical process
rather than a per-sensor artifact. `Source.select(signal_names=...)` attaches one annotation to
several signals beneath a source in one call.

<figure markdown="span">
  ![Three stacked signals sharing one window](../assets/figures/cross-sensor.svg)
</figure>

Windows can overlap or nest, and all three shapes can sit on one signal at once.

## Reading them back

A view carries the annotations attached to it, resolved into `ResolvedAnnotation` objects that hold
payload and placement together. `to_text()` renders one as a line for a prompt.

| Call | What it answers |
| --- | --- |
| `reader.annotations_for(object_type, object_id)` | Every annotation on one object. |
| `reader.objects_with(name, value=None)` | Every object carrying an annotation, starting from the annotation. |
| `reader.records_with(name, value=None)` | Every record carrying one, directly or on one of its sources or signals. |

The reverse lookups are what an annotation-first query needs: "which records are labelled
`condition=healthy`" without walking the corpus.
