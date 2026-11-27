---
icon: lucide/message-square-text
description: "Render a TimeNet task into the text and series that a model receives."
tags:
  - usage
  - prompts
---

# Render prompts

A task holds data: its input Records, a scope, the targets and a label set. A model needs text
and series in a fixed order. `timenet.prompts` renders a task with a prompt renderer into an
ordered list of parts. Each model then turns the parts into its own format, for example with its
chat template.

The first version renders `ClassificationTask` only. Other task types fail with a
`TimeFValidationError`.

## Render a task

A renderer has two functions: one for the input and one for the expected reply. `render_input()`
renders only the input, for inference. It never reads the targets or the rationale. `render()`
renders both sections, for training.

```python
import timenet
from timenet.prompts import LabelQuestionRenderer, TextPart, render
from timenet.types import ClassificationTask

client = timenet.TimeNet(sources=[".timenet-registry"])
dataset = client.load("timenet/hello-world")
task = dataset.tasks_of(ClassificationTask)[0]

prompt = render(task, LabelQuestionRenderer(), dataset=dataset.metadata)
for part in prompt.input:
    if isinstance(part, TextPart):
        print(part.text)
    else:
        print(f"[series {part.series.name}, {part.series.n_values} values]")
print(prompt.target)
```

The renderer returns three kinds of parts:

| Part | Content |
| --- | --- |
| `TextPart` | Text |
| `SeriesPart` | One channel's window, as a Signal |
| `SeriesTextPart` | One channel's window, with the text of the selected annotations on its Signal |

Pass the dataset card as `dataset` if a renderer uses the dataset's name or description.

## Default renderers

TimeNet has four default renderers for classification tasks. Your code chooses the renderer. No
renderer chooses between the task's own prompt and a question from the label set.

| Renderer | Question |
| --- | --- |
| `LabelQuestionRenderer` | "Which label matches the input?" with the options on one line. It ignores the task's prompt. |
| `PromptQuestionRenderer` | The task's own prompt. It fails for a task without a prompt. |
| `PromptWithOptionsRenderer` | The task's own prompt, then the options. It fails for a task without a prompt. |
| `DescribedRenderer` | The window, each Source and channel with its quantity, unit and rate, then the options with their descriptions. |

All four reply with the rationale and an empty line, if the task has a rationale, then one label
per line. `DEFAULT_RENDERERS` holds all four.

The renderers are part of TimeNet, so the TimeNet version fixes their text. Log the renderer's
name, such as `classification-described`, and the installed TimeNet version with a training run,
so that you know which text the model saw. Pin your TimeNet version if you need fixed prompts.

## What a renderer sees

A renderer never sees the task. It sees a view that TimeNet builds from the task, and the view
holds only plain values. The input view has these fields:

| Field | Content |
| --- | --- |
| `dataset.name`, `dataset.description` | From the dataset card, or empty without a card |
| `window.duration_s` | The duration of the task's scope, or empty without a scope or for a scope in steps |
| `input_records` | One entry for each input Record, in the task's order |
| `record.sources` | The Sources that have a channel in the window, each with `name`, `metadata` and `channels` |
| `record.channels` | The channels of all Sources of the Record |
| `channel` | `name`, `spec_name`, `spec_type`, `unit`, `unit_symbol`, `sampling_rate_hz`, `n_values` and `context` |
| `record.context` | The selected input annotations on the Record, its Sources, and its Signals outside the window |
| `context` | The selected input annotations outside the input Records |
| `prompt` | The task's own prompt, or empty |
| `label_set` | `name`, `description`, `multi_label`, `labels` and `options_line` |

Each `context` entry has a `key`, a `text` and a `line`, such as `age_band: 38-52`. The target
view adds `targets` and `rationale`.

The view leaves out what a model must not see:

- Annotations that the task does not select in `input_annotations`. In many datasets, the
  annotations on a Source hold the answers.
- The split, IDs, Record metadata and task metadata.
- Source metadata other than `device` and `placement`, because other keys can hold IDs.
- The start and end of the window. In a recording of a day, the start time is close to the time of
  day and can give away the answer.

A series part holds a stripped copy of the Signal: the values in the window, no annotations, and
only the allowed metadata. Its times count from the start of the task's window, or without a time
scope from the earliest value of all channels. All channels of a task share this zero, so they stay
aligned.

## Write a renderer

Subclass `PromptRenderer`, set its `name`, and override `render_input_prompt()` and
`render_target_prompt()`. Each method gets a view and returns a `PromptBuilder` with text and
series. This renderer shows each channel's series, then asks with the options:

```python
from timenet.prompts import PromptBuilder, PromptRenderer
from timenet.prompts.views import (
    ClassificationInputView,
    ClassificationTargetView,
)


class Short(PromptRenderer):
    name = "short"

    def render_input_prompt(
        self, view: ClassificationInputView
    ) -> PromptBuilder:
        prompt = PromptBuilder()
        for record in view.input_records:
            for channel in record.channels:
                prompt.add_series(channel)
        return prompt.add_text(f"Pick one: {view.label_set.options_line}.")

    def render_target_prompt(
        self, view: ClassificationTargetView
    ) -> PromptBuilder:
        return PromptBuilder().add_text(", ".join(view.targets))
```

Pass an instance, such as `render(task, Short())`.

`PromptBuilder` has two methods. Both return the builder, so calls can be chained:

- `add_text(*lines)` adds lines of text. All lines between two series become one `TextPart`, joined
  with line breaks and stripped. Blank text is left out.
- `add_series(channel, text="")` adds a channel of the view. `render()` replaces it with the
  channel's window, as a `SeriesPart`, or a `SeriesTextPart` if `text` is not empty. A channel
  that is not from the view fails with a `TimeFValidationError`.

`render_input_prompt()` gets a `ClassificationInputView`, which has no `targets` or `rationale`,
so it cannot reach the answer. `render_target_prompt()` gets a `ClassificationTargetView`, which
adds both.
