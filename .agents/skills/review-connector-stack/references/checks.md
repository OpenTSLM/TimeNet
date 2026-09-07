# The checks

Every check is written as a claim the diff has to prove. Cite `file:line` or record
**not reviewed**. Anchors: `AGENTS.md`, the `add-dataset-connector` references
(`fidelity.md` for the data, `layout.md` for the code), and the two worked connectors,
`physionet/sleep_edfx` and `physionet/ecg_qa_cot`. Where a document disagrees with the code, the
code wins and the document is the finding.

Every check is dataset-agnostic; it applies to whatever connector is under review. The `file:line`
references are **citations**, not examples: they point at code in this repo that shows the
convention is real and followed, so you can open one and compare. Nothing here says the connector
you are reviewing must resemble the connector cited.

## Contents

- [1. Imports and dependencies](#1-imports-and-dependencies)
- [2. I/O apart from meaning](#2-io-apart-from-meaning)
- [3. The data is a faithful copy](#3-the-data-is-a-faithful-copy)
- [4. Ids](#4-ids)
- [5. Annotations and tasks](#5-annotations-and-tasks)
- [6. Errors](#6-errors)
- [7. Naming and shape](#7-naming-and-shape)
- [8. Tests](#8-tests)
- [9. Docs and prose](#9-docs-and-prose)
- [10. Reuse](#10-reuse)
- [11. The stack itself](#11-the-stack-itself)
- [12. Checks that must have been run](#12-checks-that-must-have-been-run)

## 1. Imports and dependencies

- **A base module that every connector imports imports its library lazily.** Inside the function,
  with `# noqa: PLC0415`, and an error whose message names the connector's `requirements.txt` and
  the `--no-isolation` escape hatch. `bases/huggingface.py:47`, `bases/physionet.py:84`,
  `download/s3.py:40`. `discovery.available()` (`discovery.py:157-178`) imports **every** module
  under `datasets/` to read its `CONNECTOR`, so a top-level import here breaks dataset listing for
  every connector that never touches the library.
- **The raised type is not uniform, and that is not a finding.** `huggingface.py` and
  `physionet.py` re-raise `ImportError`; `download/s3.py:45` raises `TimeNetBuildError`. Both name
  the fix in the message, which is the part that matters. Do not report either one.
- **A base only the declaring connector imports may import at the top.** `bases/edf/reader.py`
  imports `edfio`; `bases/excel.py` imports `xlrd`. Both are fine. Do not "fix" them into lazy
  imports.
- **A connector defers exactly the libraries its own `requirements.txt` declares.** The deciding
  question is the same one the base answers: `discovery.available()` imports every connector
  module, so a top-level import of a library the package does not depend on breaks listing for
  everybody. A library the package already has — `pyarrow` is a core `timenet` dependency
  (`packages/timenet/pyproject.toml:28`), and so are the stdlib, `timenet` and `timenet_connectors`
  — goes at the top, and deferring it is the finding. A library named only in this connector's
  `requirements.txt` goes inside the function, with `# noqa: PLC0415`, a reason, and an
  `ImportError` naming that `requirements.txt` and `--no-isolation`.
- **A new dependency is declared twice.** The connector's `requirements.txt`, and the `dev` group
  of the root `pyproject.toml`. One without the other leaves `ty` unresolved and the connector's
  tests unrunnable after `make sync`.
- **`requirements.txt` says why each line is there**, in a comment above it, in one sentence.
- **`uv.lock` is in the diff** if `pyproject.toml` changed.
- **No `typing.Final`.** `AGENTS.md § Python Habits` states it. It is used nowhere in either
  package, so a first use is a convention change and belongs in its own discussion.
- **No new top-level import of a heavy library in `connector.py`** that the connector does not
  declare.

## 2. I/O apart from meaning

- **`connector.py` orchestrates and nothing below it drives.** It walks the release, opens what it
  needs, and hands what it opened downward. The modules below open nothing and walk nothing.
- **A function touches the disk or builds a value, never both.** `tables.py` and `metadata.py`
  take values and give values. If a helper module imports `pathlib` for anything but a type hint,
  ask why.
- **A file is read one time.** A function that takes a `Path` while its caller already holds the
  open file re-reads what has been read. `reader.open_edf(path)` parses the header once;
  `read_signal(file, index)` takes the open file.
- **A lazy loader closes over the open file**, so one sample's signals share one open file and one
  header parse. Note the cost in the review if the release is large: whatever a loader captures
  stays alive until it is called, and it is called after `convert` returns.
- **`convert` holds the loop, and the loop states what a record is made of.** A `convert` that
  calls one helper hiding the whole record is worse, not tidier. `sleep_edfx` suppresses `PLR0914`
  for exactly this, with the reason in prose above the `def`.
- **The library's own types stay inside the base that wraps it.** No `edfio` or `wfdb` type
  crosses into a connector module.

## 3. The data is a faithful copy

From `fidelity.md`. Each of these is a fail if the diff does the opposite without a README entry:

- Nothing the source states is dropped. A record that serves no task is still evidence, and every
  record the release ships is present — the short one, the badly scored one, and the outlier.
- **An annotation naming a series the recording does not hold means the series is built, not the
  annotation dropped.** A span refuses a `time_series_id` that resolves to nothing, so the tempting
  fix is to drop the annotation; the source claims the signal exists, so the record carries it.
  Built from what the source recorded, never from invented values.
- **The README quotes the description sentences the design relies on**, beside the card's
  `source_url`. No header states which series an annotation was scored from, so if the connector
  scopes annotations to particular series, the prose that justifies it must be quoted and checkable.
- No resampling, no interpolation, no gap filling, no normalising, no rounding. Every series keeps
  its own axis, so mixed rates need none of it.
- The gain, the rate and the block duration come from the header of the file being read, never
  from a constant. Check for a hard-coded record rate.
- Signal names, labels and units are the source's. A spec says what kind of thing it is; it does
  not rename the signal.
- A derived fact is added beside a stated one, never in place of it. Where two sources disagree,
  both are kept and the README says which wins and why.
- A broken artifact raises. A silent repair by the underlying library is caught and turned into
  `TimeFFormatError` — `bases/edf/reader.py` matches the text of `edfio`'s truncation warning.
- An inconsistency the source ships warns rather than raises. The carve-out is an **unknown
  signal name**, which `fidelity.md § Errors and warnings` requires to raise `TimeFFormatError`:
  a release that grew a signal must fail loudly, not convert to a record quietly missing a signal.
  Cite that section rather than the compressed form.
- **A warning is emitted once per kind, with a count and one example**, through a module logger
  (`_LOG = logging.getLogger(__name__)`), never `warnings.warn` — a warning raised inside a lazy
  loader never reaches whoever started the build. The carve-out is re-emitting a warning the
  underlying library raised, which a reader still needs to see: `bases/edf/reader.py:115` passes one
  back through `warnings.warn_explicit` after inspecting it. That is not a finding.
- **No warning duplicates one TimeF already gives.** `add_annotation` emits
  `SpanOutsideWindowWarning` itself. A connector that warned about the same overrun doubled the
  output.
- **`warn_when_outside=False` is not a mute, it is the raise.** The default `True` warns and keeps
  the span (`record.py:142`, `:188`); `False` restores `TimeFValidationError` (`:145`, `:191`). A
  diff that passes it is choosing an error over a warning, and should say why.
- Every inconsistency has a `README.md` entry beside the connector: the evidence, the decision,
  and the state. A number the author measured is marked *(measured)*.
- **Every *(measured)* number in the README has a counterpart in the plan's phase-5 table.** A
  reviewer who has not downloaded the release cannot re-measure one, and that is honest; what the
  reviewer *can* check is whether the number was measured against a build at all. The plan records
  the predicted number, the measured number and which was wrong. A README number with no row there
  was never measured, and it is a finding, not a "not reviewed". One release shipped a wrong unit on
  2640 records because the dataset card was believed and nothing measured it.

## 4. Ids

- **No `id=` unless something resolves the object by that id.** `fidelity.md § Ids` is the rule;
  this is the checkable form of it. Everything defaults to a UUIDv7 from `new_id()`. Four reads
  make an id load-bearing:

  | the read | where |
  | --- | --- |
  | a dataset keys its records by `record_id` to validate a streamed task | `dataset/dataset.py:264` |
  | an annotation's `time_series_ids` resolves against `TimeSeries.time_series_id` | `bases/edf/timeseries.py:58` builds it; `sleep_edfx/annotations.py:39,99` reads it |
  | a `ClassificationTask.target_schema` equals a registered vocabulary annotation's id | `sleep_edfx/tasks.py` builds both from one string |
  | a dataset keys its tasks by `id` to resolve `Record.task_ids` — resolved from the **generated** id, so still do not pass one | read at `dataset/dataset.py:670`; written at `:334`; streamed tasks never reach it, `:217` |

  A **streamed** task never reaches `Record.task_ids`, so nothing resolves its id and it must not
  name one.
- **A test asserting an id is not a read.** If the finding is a needless id, the assertion goes
  with it.
- **`record_id` is passed, never generated**, and built from one module-level `_ID_PREFIX`, so two
  builds of one archive give one set of ids.
- **A subject id is qualified** by whatever the release numbers separately, so that two parts of a
  release each numbering subjects from one cannot collide and silently merge two people. A bare
  ordinal is the finding.
- **`start_time` is set only when the source states a real instant.** A local wall clock with no
  zone becomes an annotation. Inventing a timezone is inventing data.

## 5. Annotations and tasks

- **Order: series, then the record, then annotations, then tasks.** `add_annotation` resolves a
  span's `time_series_ids` against the record, so the series must exist first.
- **Annotations attach in one batch.** `add_annotations` validates the batch before attaching any
  of it.
- **A scoped span is measured against the intersection of the named series' windows. An unscoped
  span is measured against the record's `time_span` when it declares one, and against the union of
  its series windows when it does not** — and a span landing in a *gap* between two merged windows
  is outside it (`dataset/record.py:94-116`, `_reject_outside_union`). A review that expects
  warnings should predict which kind they are, and a record with no `time_span` is the case that
  catches people out.
- **A closed set is one annotation whose value is the list**, registered with
  `register_annotations`, not one annotation per member.
- **`target_schema` equals the id of that annotation**, and one function builds both from one
  string.
- **Streamed tasks**: `source` is a callable giving a fresh iterator on every call, and each task
  carries its own `record_ids` and states no id. A streamed task **is** validated per task —
  `dataset.py:267` calls `_validate_streamed_task` on every one, so an unknown record or an
  out-of-window span still raises. What it skips is the *cross-task* checks that need every task at
  once. Do not report a streamed task as unvalidated.
- **A stream may re-open a file, and that is not a finding.** `set_task_stream` needs a source it
  can read again (`dataset/dataset.py:222-224`), which for a release larger than memory means
  re-reading a file; `physionet/ecg_qa_cot/connector.py:297` streams straight out of its CoT CSVs.
  What the stream must not do is hold the tasks alive between calls, or read something that can
  answer differently the second time. Report a stream that keeps state, not one that opens a file.
- **The dedupe of a repeated annotation goes through a holder object** (`MetadataAnnotation`)
  where the annotation and its consumer are in one pass. A value-derived id (`_qtype_id`) is the
  fallback only when a stream reads it back without holding it, and both ends route through one
  function.
- **The connector invents no prompt** where the release states no question in words.
- **A run-length expansion is exact, or the README says what was decided.** An entry divides into a
  whole number of windows only if every onset sits on a boundary and every duration is a multiple of
  one. Where it does not, the choice of where a window starts is a decision and needs an entry.
- **Unlabelled time gets no task.** It is not the negative class and not the default label. A task
  built over a stretch the source never labelled invents a label.
- **`target` and `target_annotation_ids` are exclusive**, and `add_task` enforces it. `target`
  carries the answer inline; `target_annotation_ids` says the answer *is* those stored annotations.
- **A task's scope names no signal** unless the release says a model may read only those. A
  scoped annotation records what the annotator looked at; a task states what a model must answer,
  and they are different things.

## 6. Errors

- The type matches the table in `AGENTS.md § Conventions`. In a connector that is mostly
  `TimeFFormatError` for a bad artifact and `TimeNetDownloadError` for a fetch that arrived
  without the parts promised.
- No raw `ValueError` or `Exception`. But `ImportError` for a missing optional extra, `LookupError`
  for an unresolvable connector id, `FileNotFoundError` under a searched root, and
  `typer.BadParameter` for CLI input are deliberate. Do not report them.
- The message names the file, recording, or cell a reader has to go and open, and `!r`-quotes the
  value that failed.
- Two branches that share a message bind it to a local; they do not repeat the literal.
- `# noqa: DOC502 (reason)` is the sanctioned escape when a documented `Raises:` comes from a
  helper. Parenthesised, lowercase, no trailing period.

## 7. Naming and shape

- **The handle type is named `<Dataset>Source`, and `download_async` returns a list holding exactly
  one of them** (`sleep_edfx/connector.py:228` returns `list[SleepEdfxSource]`, built at `:262`).
  Not a list with an entry per record. `convert` starts `source = raw_refs[0]` and does not iterate
  `raw_refs`.
- **The generator is `_iter_<plural>`, the per-ref builder `_<plural>_for`, the id helper
  `name_<thing>`, the parser `_parse_<thing>`.** A module's single public builder is `build`; one of
  several is `build_<thing>`.
- **One module-level `_ID_PREFIX`**, and every id the connector writes is built on it.
- **`__init__.py` re-exports with explicit self-aliases** (`CONNECTOR as CONNECTOR`), so a
  re-exported name is unambiguously public to a type checker rather than an incidental import.
- **An archive is located with `find_dir_containing`**, not by guessing the extracted layout, and
  fetched with `ensure_archive`.
- **A module belongs in `bases/` when a second connector could import it unchanged.** Lift on the
  second copy, not the third. A value whose meaning changes between two parts of one release stays
  in the connector.
- **Two modules writing the same annotation key means a `keys.py` of `StrEnum`.** One module may
  keep bare literals.
- **A key's description sits in one dict keyed by the key**, not in each call that builds an
  annotation.
- **A helper module that needs the shape of a value `connector.py` owns takes a `Protocol`**, so the
  import stays one way.
- **A constant used once lives at its use site**; a lookup table and a source URL stay at module
  level.

## 8. Tests

- **No checked-in fixture bytes.** There is no `fixtures/` directory under `datasets/`. The test
  writes what it needs.
- **The fixture comment says it is invented and how a reader can tell.** The best form gives the
  reader a property that no real record has — an id outside the range the release uses, a date
  outside its span — and says so in the comment. `sleep_edfx` numbers synthetic subjects above 89
  because the release numbers none that high.
- **A module that takes values is tested with values** — no temp file, no library import.
  `test_tables.py` passes literal tuples and never imports `xlrd`.
- Tests live at `datasets/<org>/<name>/tests/`, and `make test` excludes them; they run under
  `make test-connectors`.
- **`ty` narrowing, never a cast.** `Annotation.span`, `Task.scope`, `TimeSeries.time_axis` and
  `span_us` are unions. In tests, small `assert isinstance(...)`-and-return helpers.
- **`RUF069` bans `==` between floats**, tests included. Compare the stored integer microseconds
  or use `pytest.approx`.

## 9. Docs and prose

- Module docstring states what the module does **not** do: "This module reads no file."
- Comments explain the release, not the code.
- A complexity suppression takes a prose reason above the `def`, not inside the `noqa`.
- Google-style docstrings on every public module, class and function; `DOC` checks that documented
  args and returns match the signature.
- Prose hard-wrapped near 100 columns; code inside a fence at 80. Every fence tagged with its
  language unless column alignment carries meaning.
- Docstring voice matches the connector being edited. `layout.md` marks this unsettled — do not
  report a divergence as a finding.

## 10. Reuse

Reuse is judged against the whole tree, which is what an author writing one connector cannot see.
That is why it is the reviewer's group and not only the author's.

- **The connector imports every base that fits.** A release read through EDF, WFDB, Excel or the
  Hub has a base for it under `bases/`. A connector that opens one of those formats itself, rather
  than through the base, is a finding unless it says why the base does not fit.
- **A base that does not fit is named, with the reason.** `BaseHuggingFaceConnector` returns a
  list holding every row of the release (`bases/huggingface.py:30,71-72`), so a large Hub dataset
  has to write its own `download`. That is a correct reason. "I did not look" is not one, and the
  absence of any reason is the finding.
- **The connector defines nothing a base already defines.** A second copy of `find_dir_containing`,
  of an archive fetch, or of a header parse belongs in `bases/` or `download/`.
- **A helper a second connector would want is in `bases/`, not here.** The test is
  `layout.md § The folder`: could a second connector import it unchanged? Lift on the second copy,
  not the third. A value whose meaning changes between two parts of one release stays put.
- **An existing connector already answered the modelling question.** A row-shaped release with no
  id and no stated rate is `chengsenwang/tsqa`; an answer stored once and referenced by many tasks
  is `physionet/ecg_qa_cot:227`. A connector that invented one of those has a finding against it,
  and the plan's reuse table is where it should have been caught.

## 11. The stack itself

- **Each PR stands alone.** No comment, docstring or README line that a later PR in the stack
  deletes. No forward-looking chatter ("the next PR adds…").
- **Each PR is one subject.** A change and its opt-out belong together; two unrelated subjects do
  not. For a connector the subjects are, bottom to top: the card; the pure modules with their
  tests; `connector.py` with its tests; the `README.md`. Each is reviewable without the ones above
  it, so a PR holding two of them is a finding.
- **Each PR is 100 to 500 changed lines.** A PR of 1290 lines is not reviewable, whatever else is
  true of it. Where a connector will not fit, the seam above is where it divides.
- **The commit subject and the PR title name the dataset and say what landed.** `feat(slip): read
  the SLIP pretraining corpus` tells a reader who has never met SLIP nothing: not that this adds a
  connector, not what SLIP is, not where it comes from. A scope nobody recognises needs the title
  to carry the rest.
- **A mid-stack change lives on the branch that owns it**, propagated with
  `gh stack rebase --upstack`, not folded into a higher branch.
- **Commit messages** are Conventional Commits with the connector as the scope:
  `feat(sleep-edfx): read the EDF container`. Branches are Conventional Branch.
- **No gratuitous renames.** A function or test renamed for no reason in the diff is a finding
  against the diff, not against the old name. Prose that is still true stays.
- **PR body is Problem, then Changelog**, in simple English, ticket reference last —
  `AGENTS.md § Stacked PRs`.
- **No `--no-verify`.** A hook that failed is fixed, not skipped.
- A PR whose commit no longer describes what the code does needs the amend called out.

## 12. Checks that must have been run

`make check`, `make test`, and — for any connector change — `make test-connectors`. `AGENTS.md`
requires the final summary to say which ran and which could not.

- **The pipeline is green on the PR.** Run `gh pr checks <n>` for every PR of the stack and record
  the result. A red or pending pipeline is treated exactly like a local check that was never run:
  the review does not clear it.
- **A local pass on one interpreter is not a pass.** The workflow runs a matrix, and the dev
  environment is one row of it. `typing.override` is 3.12 and later, so a connector using it
  imported cleanly in development and failed on every connector's `test_cards.py` under 3.11
  *(this happened, and two reviews missed it)*. The matrix is the check.
