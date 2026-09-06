---
name: add-dataset-connector
description: Use when adding a new TimeNet dataset connector, i.e. converting an external dataset (from a HuggingFace repo, PhysioNet, or another source given by a link or reference) into the TimeF format. Runs six phases: read the card and the source's own words, take a head of each file type and census the release, draw the map and write the plan, get the plan approved, build the connector, then prove the build against what the plan predicted.
---

# Adding a dataset connector

A connector fetches a dataset's raw source and converts it into a `TimeFDataset`. It implements the
`BaseConnector` contract (in the `timenet` package) and lives in `timenet-connectors`. You write two
things, `download` and `convert`. The build calls them and stores what `convert` returns;
`timenet-build build <id>` runs that and writes the result into a registry.

This skill takes a link and gives back a built connector. It works in six phases with one gate. Do
not skip a phase, and do not write connector code before the gate.

## The two documents

- **`docs/notes/connectors/<org>/<name>/plan.md`** is the working document. It holds the heads, the
  census, the map, the record design and the open assumptions. It is scratch. It is untracked, and
  you never `git add` it.
- **`packages/.../datasets/<org>/<name>/README.md`** ships with the connector. It holds the
  assumptions the user ruled on, and every inconsistency the release contains. It is the document
  somebody reads a year later.

The assumptions live in the plan until the gate, then move to the README. After that the README is
the only copy. Do not keep both.

## Where to start

Look for `docs/notes/connectors/<org>/<name>/plan.md`.

- **It does not exist.** Start at phase 0.
- **It exists.** Read the phase ledger at its top and resume at the first phase that is not `done`.
  Phase 1 walks the whole release, so never repeat it when the plan already holds its census.

Every phase writes its result into the plan and updates the ledger before the next one starts.

## Phase 0 — the card and the source's own words

The card comes first, because the source's prose decides things no file header states.

1. Derive the `org/name` id from the link. Lowercase, hyphens allowed in the leaf. On disk hyphens
   become underscores, so `physionet/ecg-qa-cot` maps to `datasets/physionet/ecg_qa_cot/`. The id is
   validated as `^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$`, and no segment may start with `.`.
2. Draft `dataset.yaml` from the source page and **ask the user to confirm it**. `license` and
   `domains` are enums, and a wrong licence is a legal claim, not a typo. Never guess it.
3. Read the source's description. Keep the sentences the design will rely on, with the URL they came
   from. These are evidence, not metadata: they go in the plan now and in the README later, beside
   the card's `source_url`.

The description states what no header states, so read it for all of these:

- **Which series an annotation was derived from.** Scope the annotation to those series and no
  others. The standard a study cites is not a safe guess for what the study did.
- The epoch length, the rater, and the equipment.
- The units, the ranges, and what a label means.

Phase 0 is done when the card loads without `TimeNetInvalidCardError` and the quotes are in the plan.

**Tell the user:** the id, the card as confirmed, and the sentences of the description the design
will lean on.

**Next:** phase 1, which reads the release. Say so and continue; this hand-off needs no approval.

## Phase 1 — a head of each file type, then a census of all of them

Read `references/discovery.md` before you start this phase.

Two reads, with different jobs.

**Write one `head()` for each kind of file the release ships**, in `heads.py` beside the connector.
A head opens one file, reads a small part of it, and gives that part back as readable text. It writes
nothing. Most sources are binary — EDF, WFDB, parquet, xls, HDF5 — so `cat` is not an option, and
the head function is the deliverable. It ships with the connector, so it also runs against the next
release of the dataset and shows a changed shape at once.

**Then census the whole release with a script.** A head shows the shape of one file. It cannot show
you what is odd, because odd is a fact about the set: one file in a hundred writing a different
block length, a scaling factor that varies per file where you assumed a constant, a handful of table
rows disagreeing with their headers. You find those by counting, never by reading.

**Hold this budget:** the context cost of phase 1 does not grow with the size of the release. One
head per file *type*, not per file. The census walks every file but only its table enters the
conversation. Run the walk in a subagent and take back the table alone.

Produce, into the plan:

- the file inventory, including the files that belong to no record,
- one head per file type,
- the census table,
- the map: a mermaid diagram in four columns — what ships, what pairs it into records, what reads
  the container, what each part means,
- what one record is, with real values from the heads,
- the task type, the count of tasks per record, and therefore whether tasks are added or streamed,
- the module skeleton and the test plan,
- the numbers the build should produce,
- the assumptions and open questions.

`references/plan-template.md` is the format. Use it as written, so every connector's plan reads the
same. `references/fidelity.md` decides what the design may and may not do to the data;
`references/layout.md` decides the module skeleton. Read both before you write the plan.

**Tell the user:** what the release holds — the count of records, the shapes the census found, and
anything odd — and that the plan is ready to read.

**Next:** phase 2, the gate. This one **stops**. Do not begin phase 3 until the user has ruled.

## Phase 2 — the gate

**STOP. Write no connector code.** Give the user the plan and wait for explicit approval of:

- the map and what one record is,
- the task type and how tasks are counted,
- the module skeleton and the test plan,
- **every open assumption**, one at a time, each with the evidence behind it.

An assumption is a question the source does not answer. Say what the source states, say what you
would do, and let the user rule. Do not resolve one silently.

If the user changes the design, revise the plan and ask again. Only continue on an explicit yes.

**Next:** on a yes, phase 3, which writes the rulings into the connector's README.

## Phase 3 — the assumptions become the README

Write `README.md` beside the connector from `references/readme-template.md`. One entry for each
assumption and each inconsistency, with three parts: the evidence, the decision, and the state —
**Handled**, **Not built**, or **Open**. An open entry is worth more than a tidy file, because it
names what nobody has decided.

Mark any number you measured over the release yourself as *(measured)*, so a reader can tell it from
one copied off the dataset's page. Quote the description sentences from phase 0 beside `source_url`.

Use the `simple-english` skill on the prose. Do not run it over the head output or the census table;
those are evidence, and rewording them destroys them.

Then delete the assumptions from the plan. The README owns them now.

**Tell the user:** the README path, and how many entries are **Open** — an open entry is a decision
still owed, and it should not be a surprise at review time.

**Next:** phase 4, the build.

## Phase 4 — build

Read `references/connector-anatomy.md` for the contract, the base connectors and the task types.
Follow the skeleton the plan states and the rules in `references/fidelity.md` and
`references/layout.md`. Build in the order the format forces: series, then the record, then
annotations, then tasks.

Write the tests the plan named as you go. The modules the plan marked pure need no fixture, which is
the whole point of keeping them pure.

**Next:** phase 5, which proves the build against the numbers phase 1 predicted.

Run a build as you write it:

```bash
uv run timenet-build build <org>/<name> \
    --no-isolation --keep-cache --out ./out
```

- `--no-isolation` runs in the current interpreter. The default builds an environment from the
  connector's `requirements.txt`, which is right for CI and slow while you write.
- `--keep-cache` keeps the raw download. Without it the build deletes the cache directory it
  created after a successful build, and the next run downloads the release again. It deletes only a
  cache it created; a `cache_dir` a caller passed is user-owned and left alone.

**Seed the cache from a copy you already have** rather than downloading twice. The cache directory is
`~/.cache/timenet/cache/<dataset_id>/`, and `ensure_archive` skips the download when it finds a
marker file there named `.<first 8 hex characters of sha256(url)>-<archive name>.extracted`. Put the
extracted release in that directory, create that marker, and the build reads it. `find_dir_containing`
searches with `rglob`, so the tree can sit at any depth.

## Phase 5 — prove the build

Run in this order and stop at the first failure:

1. `make check`, `make test`, `make test-connectors`.
2. `uv run timenet-build build <id> --no-isolation --keep-cache --out ./out`
3. Load it back: `TimeNet(registry="./out").load("<id>").describe()`.
4. Compare the result against the numbers the plan predicted: records, series per record, tasks, and
   the warning count with its reason. A number that does not match means the plan is wrong or the
   code is. Find out which and say so.

**A build can fail after `convert` returned cleanly**, and the first time it happens it is
confusing. `convert` gives back a description; the values and the tasks are read afterwards. So a
loader that raises when called, a task naming a record that does not exist, a span outside its
window, and a stream that yields nothing on a second pass all surface *after* the step that looks
responsible has finished. If a failure names none of your own modules, it is that stage: check what
your loaders do when called, and what your task stream gives on a second call.

**Tell the user:** which checks ran, and which predicted numbers matched.

## Rules that hold in every phase

- **Never `git add` the plan.** `docs/notes/` stays local, like `docs/openspec/`.
- **Never read the data to learn about the data.** Heads show shape, censuses show anomalies.
- **Write down what you measured, and mark it *(measured)*.** A number nobody can re-measure is a
  claim, not evidence.
- Follow AGENTS.md: conventional commits, no `--no-verify`, TimeNet's own errors from
  `timenet.errors`, Google-style docstrings, type hints.

## Further reading

- `references/discovery.md` — heads, censuses, and the map.
- `references/fidelity.md` — what a connector may and may not do to its source.
- `references/layout.md` — how the modules of a connector divide.
- `references/connector-anatomy.md` — the contract, the bases, the task types, a worked example.
- `references/plan-template.md`, `references/readme-template.md` — the two standard formats.

Repo docs: `docs/connectors.md` and `docs/build.md` are a design proposal. Where they describe a
flat-file `datasets/<org>/<name>.py` layout, the real code uses the folder-package layout. Follow
the code.
