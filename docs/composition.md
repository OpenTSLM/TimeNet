# Dataset composition

A composed connector imports records from exact parent releases and adds its own tasks.
The child has separate record hierarchies and annotations. Its value handles still refer to the
parent payloads. The writer records those payload sources through the standard TimeF storage API.

Declare each parent by alias in the dataset card:

```yaml
parents:
  ptbxl: physionet/ptb-xl@1.0.0
```

The connector builder prepares missing parents before it builds the child. A parent version must
match the exact version in the card. A missing builder, a version mismatch, or a dependency cycle
stops the build.

Root connectors implement `convert(raw_refs)`. A composed connector also implements
`compose(raw_refs, context)`. The engine passes a `BuildContext` with the declared parents and keeps
its readers open until writing finishes.

```python
from timenet.composition import BuildContext

def compose(self, raw_refs, context: BuildContext):
    dataset = context.dataset()
    for record in context.parent("ptbxl").iter_records():
        imported = dataset.import_record(record, parent="ptbxl")
        # Add child tasks that refer to imported.
    return dataset
```

`iter_records()` reads record hierarchies with lazy values. An optional list selects records by
numeric entity id or by the connector's `source_record_id` metadata. Upstream identifiers must be
unique within a parent. TimeNet assigns fresh numeric entity ids within the child dataset.

Repeated imports of the same parent record return the same child record. Changes to a child
record's annotations do not change the parent record. Parent tasks are not imported.

PTB-XL provides reusable ECG records for ECG-QA CoT. The five HEARTS corpus connectors provide
records for HEARTS tasks. Each corpus connector can also build independently.
