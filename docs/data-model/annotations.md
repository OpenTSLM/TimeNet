---
icon: lucide/highlighter
description: "Annotations: scoped side-information on a sample, in three shapes, that become task context or targets."
tags:
  - guide
  - concepts
---

# Annotations

An annotation is side-information attached to a [sample](samples.md). Every annotation has two parts: a
**scope** (which channels, and which point or window in time it refers to) and a free-text **content**
that can be as short as a tag or as long as a paragraph of reasoning. The three shapes differ only in
how they sit in time. All are keyword-only frozen dataclasses sharing `key`, `value`, `unit`, and
`description` fields, so a connector authors them directly (or subclasses with field defaults for reuse).

## StaticAnnotation

A `StaticAnnotation` has no time reference: it describes the whole recording. It requires a `value` and
carries no time fields. Sample-level facts live here: the machine id, its firmware version, an operating
mode, or a single condition label that should travel with every window later drawn from the sample.

```python
from timenet.types import StaticAnnotation

StaticAnnotation(key="condition", value="healthy")
StaticAnnotation(key="operating_hours", value=1200, unit="hours")
```

<figure markdown="span">
  ![A band over the whole recording marking a sample-level fact](../assets/figures/annotation-static.svg)
</figure>

## TemporalAnnotation

A `TemporalAnnotation` marks one instant on one or more channels. It is the right shape for discrete
events: a shock, a valve actuation, a detected spike. Points are cheap to store, are often produced in
bulk by detectors, and then serve as anchors for downstream windowing. Pass `time_series_ids` to target
specific channels, or leave it `None` for the whole sample.

```python
from timenet.types import TemporalAnnotation

TemporalAnnotation(key="impact", span=PointSpan.seconds(4.2, time_series_ids=("vibration",)))
```

<figure markdown="span">
  ![A marker at one instant on a channel](../assets/figures/annotation-point.svg)
</figure>

## TemporalAnnotation

An `TemporalAnnotation` covers a start-to-end window on one or more channels, and it is the workhorse.
Its `span` is an `IntervalSpan`, whose end must exceed its start. Everything expressive about the scoping grammar, which channels
by which time range, lives here, and the `value` and free-text `description` can carry the full reading
of what happens in that window.

```python
from timenet.types import TemporalAnnotation

TemporalAnnotation(
    key="fault",
    value="bearing fault",
    span=IntervalSpan.seconds(5.0, 8.0, time_series_ids=("vibration",)),
)
```

<figure markdown="span">
  ![A shaded window on a channel](../assets/figures/annotation-interval.svg)
</figure>

## Across channels

Annotations are not tied to one channel. A single event can span a vibration sensor, a temperature
probe, and a current sensor together, and the shared timing points at one physical process rather than a
per-channel artifact.

<figure markdown="span">
  ![Three stacked channels sharing one window](../assets/figures/cross-sensor.svg)
</figure>

A sample holds a *list* of annotations, so several spans can sit on one channel, windows can overlap or
nest, and the three shapes can coexist on one signal. Because the text field is free-form, an annotation
can carry a multi-sentence reading rather than just a label, which is what lets it become a reasoning
target. That is the bridge to [tasks](tasks.md).
