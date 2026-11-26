---
icon: lucide/message-square-text
description: "Render a TimeNet task into the text and series that a model receives."
tags:
  - usage
  - prompts
---

# Render prompts

A task holds data: its input Records, a scope, the targets and a label set. A model needs text
and series in a fixed order. `timenet.prompts` renders a task with a prompt template into an
ordered list of parts. Each model then turns the parts into its own format, for example with its
chat template.

The first version renders `ClassificationTask` only. Other task types fail with a
`TimeFValidationError`.

## Render a task

A template has two sections: one for the input and one for the expected reply. `render_input()`
renders only the input, for inference. It never reads the targets or the rationale. `render()`
renders both sections, for training.

```python
import timenet
from timenet.prompts import LABEL_QUESTION, TextPart, render
from timenet.types import ClassificationTask

client = timenet.TimeNet(sources=[".timenet-registry"])
dataset = client.load("timenet/hello-world")
task = dataset.tasks_of(ClassificationTask)[0]

prompt = render(task, LABEL_QUESTION, dataset=dataset.metadata)
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

Pass the dataset card as `dataset` if a template uses the dataset's name or description.

## Default templates

TimeNet has four default templates for classification tasks. Your code chooses the template. No
template chooses between the task's own prompt and a question from the label set.

| Template | Question |
| --- | --- |
| `LABEL_QUESTION` | "Which label matches the input?" with the options on one line. It ignores the task's prompt. |
| `PROMPT_QUESTION` | The task's own prompt. It fails for a task without a prompt. |
| `PROMPT_WITH_OPTIONS` | The task's own prompt, then the options. It fails for a task without a prompt. |
| `DESCRIBED` | The window, each Source and channel with its quantity, unit and rate, then the options with their descriptions. |

All four reply with the rationale and an empty line, if the task has a rationale, then one label
per line. `DEFAULT_TEMPLATES` holds all four.

Each template name ends with a version, such as `classification-described-v1`. A change to the
text of a template gets a new version. Log the name with a training run, so that you know which
text the model saw. A new TimeNet release can add versions, so pin your TimeNet version if you need
fixed prompts.

## What a template sees

A template never sees the task. It sees a view that TimeNet builds from the task, and the view
holds only plain values. The input view has these fields:

| Field | Content |
| --- | --- |
| `dataset.name`, `dataset.description` | From the dataset card, or empty without a card |
| `window.duration_s` | The duration of the task's scope, or empty without a scope or for a scope in steps |
| `records` | One entry for each input Record, in the task's order |
| `record.sources` | The Sources that have a channel in the window, each with `name`, `metadata` and `channels` |
| `record.channels` | The channels of all Sources of the Record |
| `channel` | `name`, `spec_name`, `spec_type`, `unit`, `unit_symbol`, `sampling_rate_hz`, `n_values` and `context` |
| `record.context` | The selected input annotations on the Record and its Sources |
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

## Write a template

A `PromptTemplate` has a `name`, an `input` and a `target` section, both in Jinja. This template
shows each channel's series, then asks with the options:

```python
from timenet.prompts import PromptTemplate

SHORT = PromptTemplate(
    name="short-v1",
    input=(
        "{% for record in records %}"
        "{% for channel in record.channels %}"
        "{{ series(channel) }}"
        "{% endfor %}{% endfor %}"
        "Pick one: {{ label_set.options_line }}."
    ),
    target="{{ targets | join(', ') }}",
)
```

Templates can use two helpers:

- `series(channel, text="")` places the channel's window as a series part. With `text`, it gives a
  `SeriesTextPart`.
- The filter `required(value, what)` fails with a clear error if a field is empty, for example
  `{{ prompt | required("a task prompt") }}`.

TimeNet renders templates in Jinja's immutable sandbox with strict undefined values. A template that
uses a field that the view does not have fails with a `TimeFValidationError`. The input section has
no `targets` or `rationale`, so it cannot reach the answer. Text between two series is stripped,
and empty text is left out.

Connectors cannot ship their own templates yet.
