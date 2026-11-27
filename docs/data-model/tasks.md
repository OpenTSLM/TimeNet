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
from timenet.types import ClassificationTask, LabelSet

CONDITION = LabelSet(["healthy", "faulty"])

task = ClassificationTask(
    inputs=(record,),
    targets="faulty",
    label_set=CONDITION,
)
dataset.add_task(task)
```

The concrete class identifies the task type. `ClassificationTask` and `AnswerTask` are different
task types, even when both contain string targets.

## Common fields

Every Task supports these fields:

| Field | Purpose |
| --- | --- |
| `id` | Read-only int64 Task identity. `add_task()` sets it. |
| `inputs` | Ordered input Records. |
| `targets` | Ordered inline values or stored objects. One bare value, such as `targets="healthy"`, is one target. |
| `prompt` | Optional instruction or question. |
| `input_modalities` | Optional override for the input kinds. `None` enables inference. |
| `split` | Optional partition: `train`, `validation`, or `test`. `None` when the dataset does not partition its Tasks. |
| `scope` | Optional input region. |
| `input_annotations` | Attached Annotation occurrences supplied as context. |
| `target_annotations` | Attached Annotation occurrences used as the answer. |
| `rationale` | Optional explanation associated with the answer. |
| `from_tasks` | Parent Tasks from which this Task derives. |
| `annotations` | Annotations that describe the Task itself. |
| `metadata` | JSON-compatible Task details. |

The collection fields (`inputs`, `input_annotations`, `target_annotations`, `from_tasks`, `annotations`)
accept any iterable, such as a list, and store a tuple. A string, a mapping, or a single item raises
`TimeFValidationError`.

`targets` can contain strings, numbers, Booleans, Records, Signals, or Spans. `None` means that
`target_annotations` supplies the answer. An empty tuple is an explicit empty answer.

A Task cannot set both `targets` and `target_annotations`.

## Input modalities

`None` by default, in this case the value is inferred.
Inference uses each input Signal's `spec.modality`, restricted to Signals named in `scope`, if any.
It also adds `TEXT` for a non-empty prompt or text input annotations. It ignores targets, target
annotations, rationale, and metadata. It does not load Signal values.

To override inference, pass a non-empty set:

```python
from timenet.types import AnswerTask, InputModality

task = AnswerTask(
    inputs=(record,),
    prompt="Describe this image.",
    input_modalities=frozenset({InputModality.IMAGE, InputModality.TEXT}),
)
```

The override remains subject to validation. For example, a Task with a prompt must declare `TEXT`.
The writer stores the resolved set in DuckDB, so SQL filters need no inference or Signal reads.
Read-back Tasks carry that stored set as an explicit declaration.

## Splits

A dataset that ships a partition marks each Task with its `split`. Records carry no split of their
own: one Record may serve Tasks of several splits.

```python
from timenet.types import ClassificationTask, LabelSet, Split

ACTIVITY = LabelSet(["walking", "sitting"])

dataset.add_task(
    ClassificationTask(
        inputs=(record,),
        targets="walking",
        label_set=ACTIVITY,
        split=Split.TRAIN,
    )
)
```

`TimeFDataset.get_train()`, `get_validation()`, and `get_test()` return the Tasks of one split, and
`get_all()` returns every Task. Asking a split of a dataset that assigns none raises, since that is a
mistake rather than an empty selection. `iter_tasks(split=...)` on a dataset or a `TimeFReader`
filters the same way; the reader applies the filter in DuckDB, so a Task outside the split is never
built. `to_features_and_targets(split=...)` builds the `(X, y)` pair of one split.

To train on train and validation together while keeping test separate, select their union:

```python
training_tasks = dataset.get_splits(Split.TRAIN, Split.VALIDATION)
test_tasks = dataset.get_test()

# Or consume the selection lazily from a TimeFDataset:
for task in dataset.iter_tasks(split=(Split.TRAIN, Split.VALIDATION)):
    process(task)
```

Selection preserves dataset order and each Task's original `split`. Repeated split names do not
duplicate Tasks, and Tasks with `split=None` are excluded. `get_splits()` materializes its result
in one pass and raises if the dataset has no split assignments. The dataset iterator remains lazy
and simply yields no matches in that case. Both selectors require at least one valid split;
pass `split=None`, omit the argument, or use `get_all()` to include every Task. The dataset
iterator's `split` argument accepts an enum, a string, or an iterable of either. Modality filters
can be combined with split selection.

## Register Tasks

Register all referenced objects before you register the Task:

```python
dataset.add_record(record)
dataset.add_task(task)
```

`add_task()` validates Record, Signal, Annotation, Span, and parent-Task references. It does not
change the input Records. Use `dataset.tasks_for(record)` to find the Tasks that take a Record as
input.

Use `add_tasks()` for an atomic batch. A parent in `from_tasks` can appear in the same batch.

For a large task corpus, use `set_task_stream()`. The writer validates each streamed Task before it
writes the Task. A stream does not keep the complete task set in memory. TimeNet therefore cannot
validate `from_tasks` relationships across streamed Tasks.

## Records that a Task references

A Task references these Records:

- Its inputs.
- Its candidate Records.
- Its Record targets.
- Each Record that holds a Signal that the Task references, for example a Signal target.

Only the inputs are Records that the Task takes as input. `tasks_for(record)` returns the Task only
for its inputs, and `read_tasks(inputs=)` selects the Task only through its inputs.

## Scope an input

`scope` limits the region that the model gets. A scope holds the Signal objects that it covers:

```python
from timenet.types import ClassificationTask, TimeInterval

task = ClassificationTask(
    inputs=(record,),
    scope=TimeInterval.seconds(
        5.0,
        8.0,
        time_series=(vibration,),
    ),
    targets="faulty",
    label_set=CONDITION,
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

The set is closed. TimeF stores each Task's type as a tag, and the reader maps that tag back to one
of these classes. Each class is therefore final, and `Task` refuses subclasses defined outside
TimeNet. To reuse a prompt or other settings, write a builder function that returns a built-in
Task:

```python
from timenet.dataset import Record
from timenet.types import AnswerTask


def age_question(record: Record, age: int) -> AnswerTask:
    return AnswerTask(
        inputs=(record,),
        prompt="How old is the subject?",
        targets=str(age),
    )


dataset.add_task(age_question(record, 64))
```

### Classification

```python
from timenet.types import Label, LabelSet

CONDITION = LabelSet(["healthy", Label(name="faulty", description="A bearing fault.")])
FINDINGS = LabelSet(["AF", "PVC", "LBBB"], multi_label=True)

ClassificationTask(inputs=(record,), targets="healthy", label_set=CONDITION)
ClassificationTask(inputs=(record,), targets=("PVC", "AF"), label_set=FINDINGS)
```

A `LabelSet` defines the allowed labels for a classification target. Like other owners, it can carry
[annotations](annotations.md#label-set-annotations). The `multi_label` flag controls how many labels
a target can contain.

If `multi_label=False`, the default, a target must contain exactly one label. If `multi_label=True`,
a target contains all labels that apply. Omitted labels do not apply. An empty target means that no
label applies.

When the dataset adds or streams a task, it makes sure that the target follows the rules of its
label set. Each label must be in the set. No label can repeat. A target with `multi_label=False`
must contain exactly one label.

Labels come from the targets stored directly on the task or from the text of its target
annotations.

A label set has no name. TimeF stores equal sets once. For this, two sets are equal if they have the
same labels, label descriptions, `multi_label`, `ordered`, and annotations. The order of labels
matters only for an ordered set. The order of annotations does not matter. `==` follows the same rule.

Annotations are equal if all their fields are equal, the value included. So these two sets are two
stored sets, each with its own id:

```python
LabelSet(["yes", "no"], annotations=[Annotation("name", value="afib_present")])
LabelSet(["yes", "no"], annotations=[Annotation("name", value="signal_noisy")])
```

By default the order of labels does not matter. Use `ordered=True` when it has a meaning, for example
for ordinal labels such as `low`, `medium` and `high`. Then each order is its own set, and the stored
set keeps it. The order in which a prompt shows its answer options is not a reason for an ordered set.
It belongs in the [template](#templates) of the task. An unordered set keeps the
label order of the first equal set that the writer meets. Pass the labels as a list or a tuple. A
label set rejects a Python set, because its order changes from one run to the next.

The stored set gets one integer id, `label_set_id`. In one read, the reader gives tasks that use the
same set a shared object.

#### Templates

A classification task stores its prompt and its targets as the source publishes them. In several
datasets one key is the correct answer much more often than the others, so a model can learn the key
instead of the option. A `template` lets a consumer show the same task with its options in another
order and with other keys.

A template is written in a small part of the [Jinja](https://jinja.palletsprojects.com/) language.
It has a `prompt` block and an `answer` block:

```python
from timenet.types import (
    ClassificationEnumerator,
    ClassificationTask,
    ClassificationTemplate,
    LabelSet,
    OptionKeyStyle,
    PromptTemplate,
    TemplateOption,
)

SOURCE = (
    "{% block prompt %}Which trend?\n"
    "{% for option in options %}{{ option.marker }} {{ option.text }}\n"
    "{% endfor %}"
    'Answer with the {{ key_noun }} ({{ keys | join(" or ") }}).'
    "{% endblock %}"
    "{% block answer %}{{ option.key }}{% endblock %}"
)

task = ClassificationTask(
    inputs=(record,),
    prompt="Which trend?\nA: up\nB: down\nAnswer with the letter (A or B).",
    targets="B",
    label_set=LabelSet(["A", "B"]),
    template=ClassificationTemplate(
        prompt_template=PromptTemplate(source=SOURCE),
        options=[
            TemplateOption(label="A", text="up"),
            TemplateOption(label="B", text="down"),
        ],
        key_style=OptionKeyStyle(
            enumerator=ClassificationEnumerator.LATIN_UPPERCASE,
            marker_suffix=":",
        ),
        reproduces_published_prompt=True,
    ),
)
```

`get_option_presentation()` writes the prompt and the answer from the template:

```python
shown = task.get_option_presentation()
shown.prompt
# For example: 'Which trend?\n1) up\n2) down\nAnswer with the number (1 or 2).'
shown.answers
# ('2',)

published = task.get_option_presentation(use_shuffled_label_key=False)
published.prompt
# 'Which trend?\nA: up\nB: down\nAnswer with the letter (A or B).'
published.answers
# ('B',)
```

By default the options are in a random order and the keys in a random style. The random choices
depend only on `seed` and the task ID, so the same call gives the same presentation. Pass another
seed, for example the number of the epoch, for another one. The task itself does not change.

The `prompt` block writes the text for the model. The `answer` block writes one item of the answer.
It is written once for each correct option, which it gets as `option`. Both blocks get the same
facts:

| Fact | Value |
| --- | --- |
| `options` | The options in the order in which they are shown. Each has a `label` and a `text`. With keys, each also has a `key`, such as `B`, and a `marker`, such as `B:`. |
| `keys` | The key of each option, in the same order. Only with keys. |
| `key_noun` | The word for a key: `letter`, `number`, or `Roman numeral`. Only with keys. |
| `multi_label` | Whether more than one option can be correct. |
| `prompt` | The stored prompt of the task. Only if the task has a prompt. |

A template can use values, loops, conditions, blocks and a few filters such as `join`. It cannot call
a function, set a variable, load another template, or use a random filter. A block inside the
`prompt` block names one section of the prompt, for example `{% block options %}`.

The `options` of a `ClassificationTemplate` are in the order of the source, and its `key_style` says
how the source writes a key: the `ClassificationEnumerator` counts the options with Latin letters,
numbers, or Roman numerals, and the style adds the text before and after the symbol. Set `key_style`
to `None` for options without keys. A consumer then changes only their order.

Two fields restrict what a consumer can change. Set `pinned=True` on an option that must keep its
place, such as `None of the above`. Set `allowable_enumerators` if the template fits only some
enumerators.

When the dataset adds or streams a task, it makes sure that the template fits the task:

- Each option stands for a label of the label set, and each target label has an option.
- The template can be written with the options and the keys of the source.
- With `reproduces_published_prompt=True`, the template writes the stored prompt and the stored
  targets exactly. Set `False` for a template that adds text to the stored prompt, for example a list
  of the options, and for a task without a prompt.
- If the options have keys, the task has no rationale. A rationale can name a key, and TimeNet does
  not find such a name.

`get_label_of_answer()` gives the label of the option that one item of an answer names, so the answer
of a model can be compared with the target labels:

```python
shown.get_label_of_answer("2") in ("B",)
# True
```

### Answer and rationale

```python
AnswerTask(
    inputs=(record,),
    prompt="Does this trace show a bearing fault?",
    rationale="The repeated impacts match a bearing fault.",
    targets="Yes",
)
```

A missing prompt makes the Task an unprompted caption or description.

### Scalar prediction

```python
ScalarPredictionTask(
    inputs=(record,),
    targets=62.0,
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
        TimePoint.seconds(1.20, time_series=(lead_ii,)),
        TimePoint.seconds(2.05, time_series=(lead_ii,)),
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
    targets=future,
)

TSEditingTask(
    inputs=(raw_record,),
    prompt="Remove the baseline wander.",
    targets=clean_record,
)

TSGenerationTask(
    prompt="Generate 10 seconds of a 150 bpm ECG.",
    targets=synthetic_record,
)
```

A generation Task can have no input Records.

### Correspondence

```python
TSCorrespondenceTask(
    inputs=(query,),
    candidate_records=(record_a, record_b, record_c),
    targets=record_b,
)
```

An empty `candidate_records` tuple means that the candidate pool is dataset-wide.

## Compose Tasks

`from_tasks` records derivation between Tasks:

```python
label = ClassificationTask(
    inputs=(record,),
    targets="faulty",
    label_set=CONDITION,
)
answer = AnswerTask(
    inputs=(record,),
    prompt="Is this machine healthy?",
    targets="No",
    from_tasks=(label,),
)
dataset.add_tasks((label, answer))
```

TimeNet rejects missing parents, self-references, and derivation cycles.
