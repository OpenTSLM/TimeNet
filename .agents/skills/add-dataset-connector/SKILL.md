---
name: add-dataset-connector
description: Use when adding a new TimeNet dataset connector, i.e. converting an external dataset (from a HuggingFace repo, PhysioNet, or another source given by a link or reference) into the TimeF format. Runs six phases: read the card and the source's own words, take a head of each file type and survey the release, draw the map and write the plan, get the plan approved, build the connector, then prove the build against what the plan predicted.
---

# Adding a dataset connector

A connector fetches a dataset's raw source and converts it into a `TimeFDataset`. It implements the
`BaseConnector` contract (in the `timenet` package) and lives in `timenet-connectors`. You write two
things, `download` and `convert`. The build calls them and stores what `convert` returns.
`timenet-build build <id>` runs that and writes the result into a registry.

This skill takes a link and gives back a built connector. It works in six phases with one gate. Do
not skip a phase, and do not write connector code before the gate.

**A TimeF dataset is a faithful copy of its source, not a cleaned one.** Convert what the source
states, in the shape it states it. Record every assumption you had to make in the connector's
README. That one rule decides more of this work than any other. `references/fidelity.md` carries it
in full.

## Contents

- [What this is for](#what-this-is-for)
- [The two documents](#the-two-documents)
- [The checklist](#the-checklist)
- [Talk to the user before a phase, not only after it](#talk-to-the-user-before-a-phase-not-only-after-it)
- [Where to start](#where-to-start)
- [Phase 0 — the card and the source's own words](#phase-0--the-card-and-the-sources-own-words)
- [Phase 1 — a head of each file type, then a survey of all of them](#phase-1--a-head-of-each-file-type-then-a-survey-of-all-of-them)
- [Phase 2 — the gate](#phase-2--the-gate)
- [Phase 3 — the assumptions become the README](#phase-3--the-assumptions-become-the-readme)
- [Phase 4 — build](#phase-4--build)
- [Phase 5 — prove the build](#phase-5--prove-the-build)
- [Rules that hold in every phase](#rules-that-hold-in-every-phase)
- [Further reading](#further-reading)

## What this is for

Converting a dataset is a modelling problem before it is a coding one. The files do not state what
counts as one record. They do not state which signals a record holds, what an annotation is scoped
to, or what question a task asks. If you get it wrong, you produce a dataset that loads fine and
answers the wrong question.

So the job is to **agree a model with a person**, on evidence, before you write anything:

1. **Gather evidence with tools, not assumptions.** A `head()` per file type shows the shape. A
   survey over every file shows what is odd. Both produce output a person can read.
2. **Propose a model**: what one record is, its signals, its annotations, its tasks — each part
   beside the evidence that produced it.
3. **Let the user argue with it.** Every decision must be contestable, which means the evidence that
   justifies it has to be next to it. A user can point at one line and say "that is
   wrong, the head shows something else". A plan they can only accept or reject is not agreement.
4. **Write the agreement down** where it ships with the connector. A reader a year later then sees
   what was decided and why.
5. **Then implement it**, and review the result against the repository's conventions.

The plan is the artifact that carries all of this. It is not a summary of work done. It is the thing
being agreed, and the code follows from it.

## The two documents

- **`docs/notes/connectors/<org>/<name>/plan.md`** is the working document. It holds the heads, the
  survey, the map, the record design and the open assumptions. It is scratch. It is untracked, and
  you never `git add` it.
- **`packages/.../datasets/<org>/<name>/README.md`** ships with the connector. It holds the
  assumptions the user ruled on, and every inconsistency the release contains. It is the document
  somebody reads a year later.

The assumptions live in the plan until the gate, then move to the README. After that the README is
the only copy. Do not keep both.

`heads.py` sits beside the plan, in the same untracked folder, and ships with nothing. The raw
release sits in the build cache at `~/.cache/timenet/cache/<dataset_id>/`, outside the working tree.
Neither is ever committed.

## The checklist

**Post this in your first reply, before phase 0 does any work.** Then post it again with its ticks
at every phase boundary. A checklist sent one time is a header. A checklist re-sent is a position.

```markdown
- [ ] 0. Plan created, card confirmed by the user, description quotes captured
- [ ] 1. A head() per file type, survey over every file, map, model, predicted numbers
- [ ] 2. GATE — every part of the model and every assumption ruled on by the user
- [ ] 3. Rulings written into the connector's README
- [ ] 4. Built against the approved skeleton, tests written
- [ ] 5. Build proved against the predicted numbers, review run, findings fixed
```

## Talk to the user before a phase, not only after it

This skill guides a person. A person who sees nothing until a phase ends is not guided. That person
gets a report instead. Four rules, and they cost one message each:

- **Say what a phase will do before it runs.** What it will do, what it will produce, and what it
  needs from the user. Every phase below carries that as **Before you start**.
- **Announce a long step when it begins**, and say what it counts and what it will give back. The
  survey walks the whole release and takes minutes, and the skill runs it in a subagent, which is
  what makes it invisible. Say it is running.
- **Show the diagrams in the conversation**, not only in the plan. The record model is the thing a
  person can argue with. A diagram filed in a document nobody is watching was never shown.
- **Re-post the checklist with its ticks at every phase boundary.**

As a result, a phase closes with the lines it needs. **Tell the user** is a message, not a plan
section. **Write it down** names what goes into the plan. **Next** hands off to the phase that
follows. Phase 2 is itself a conversation, so it carries no separate report, and phase 5 ends the
run rather than handing off.

## Where to start

Look for `docs/notes/connectors/<org>/<name>/plan.md`.

- **It does not exist.** Start at phase 0.
- **It exists.** Read the phase ledger at its top and resume at the first phase that is not `done`.
  Phase 1 walks the whole release, so when the plan already holds its survey, never repeat it.

**Check the plan's skill ref before you continue anything.** Section 0 of the plan names the branch
the skill was read from. Run `git show <skill ref>:.agents/skills/add-dataset-connector/SKILL.md` and
compare it against the skill you are running. If the ref is missing, or the skill it holds is not
this one, **stop and ask the user**. A plan written by six phases and a gate, resumed by a skill with
neither, is a plan whose ledger the new run cannot honour.

Every phase writes its result into the plan and updates the ledger before the next one starts.

## Phase 0 — the card and the source's own words

The card comes first, because the source's prose decides things no file header states.

**Before you start, tell the user:** phase 0 reads the source page, writes the card and the plan,
and fetches the release. It needs two things from them. They say how many datasets the link holds,
and they confirm the licence.

1. **Ask how many datasets the link holds.** A release that bundles a corpus with its own benchmarks
   is common, and one `org/name` id cannot address both. Which one you are converting changes the
   record model, the task type and the survey. If the link holds more than one, produce **one plan
   and one card per dataset**. Say which plan is which. Do not fold two datasets into one id.
2. **Create the plan.** Copy `references/plan-template.md` to
   `docs/notes/connectors/<org>/<name>/plan.md`. That file is the template itself, so copying it is
   the whole step. It exists from here on, and every later phase writes into it.
3. **Record where the code is built.** Section 0 takes three lines and all three are load-bearing:
   the **base ref** this connector branches from, **why** it is not `main`, and the **skill ref**,
   the branch that holds this skill. If you build on the wrong ref, you write code that contradicts
   the skill on every line. The skill's own references are also unreadable from a branch that does
   not carry them — see **Rules that hold in every phase**. Tell the user all three.
4. Derive the `org/name` id from the link. Lowercase, hyphens allowed in the leaf. On disk hyphens
   become underscores, so `physionet/ecg-qa-cot` maps to `datasets/physionet/ecg_qa_cot/`. The id is
   validated as `^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$`, and a segment must not start with `.`.
5. **Fetch the release, into the build cache and never into the working tree.** Phase 1 opens files,
   so the files must be on disk. Nothing later in this skill fetches them for you. The destination
   is `~/.cache/timenet/cache/<dataset_id>/`, which is where phase 4 looks for a seeded cache.
   A 12 GB download inside the working tree is one `git add -A` away from a commit that cannot be
   undone cheaply.

   | source | how to fetch it |
   | --- | --- |
   | Hugging Face | `git clone git@hf.co:datasets/<org>/<name>` — needs `git lfs`, or the clone leaves pointer files behind |
   | PhysioNet | `wget -r -N -c -np <url>` |
   | S3 | `download.ensure_archive`, or the AWS CLI with `--no-sign-request` |

   Record the command, the resulting size and how long it took, in section 0. Phase 1's budget rule
   needs the size.
6. Draft `dataset.yaml` from the source page and **ask the user to confirm it**. `license` and
   `domains` are enums, and a wrong licence is a legal claim, not a typo. Never guess it.
7. Read the source's description. Keep the sentences the design will rely on, with the URL they came
   from. These are evidence, not metadata: they go in the plan now and in the README later, beside
   the card's `source_url`.
8. **Read the files that describe the release before you open one that holds it.** These are a
   `dataset_info.json`, a manifest, a data dictionary, a `state.json`. Write every claim they make
   into section 0. Those claims are what phase 1's heads then check. A claim that turns out false is
   an inconsistency the README must carry. One release declares a schema its own shards do not ship.
   The two missing columns are the two a record id would have come from.

The description states what no header states, so read it for all of these:

- **Which series an annotation was derived from.** Scope the annotation to those series and no
  others. The standard a study cites is not a safe guess for what the study did.
- The epoch length, the rater, and the equipment.
- The units, the ranges, and what a label means.

Phase 0 is done when the card loads without `TimeNetInvalidCardError` and the quotes are in the
plan. The release must also be on disk outside the working tree, and section 0 must name the base
ref and the skill ref.

**Tell the user:** the id, the card as confirmed, the base ref and the skill ref, where the release
was fetched to and how big it is, and the sentences of the description the design will lean on.

**Write it down:** section 0 of the plan, and set the ledger row to `done`.

**Next:** phase 1, which reads the release. Say so and continue. This hand-off needs no approval.

## Phase 1 — a head of each file type, then a survey of all of them

Read `references/discovery.md` before you start this phase.

**Before you start, tell the user:** phase 1 opens the release and counts it. It produces the heads,
the survey table, two diagrams and the record design. The survey walks every file and takes minutes,
and it runs in a subagent. Say when it starts and what it is counting.

Two reads, with different jobs.

**Write one `head()` for each kind of file the release ships**, at
`docs/notes/connectors/<org>/<name>/heads.py`. A head opens one file, reads a small part of it, and
gives that part back as readable text. It writes nothing. Most sources are binary — EDF, WFDB,
parquet, xls — so `cat` is not an option, and the head is what makes the release readable. **A head
does not ship with the connector**: it is a discovery tool, it lives with this plan, and
`discovery.md § Where a head lives` says why. Give it a `__main__`, and record the command beside its
output. The user can then run it again on any file they like.

**Then survey the whole release with a script.** A head shows the shape of one file. It cannot show
you what is odd, because odd is a fact about the set:

- One file in a hundred that writes a different block length.
- A scaling factor that varies per file where you assumed a constant.
- A handful of table rows that disagree with their headers.

You find those by counting, never by reading.

**Hold this budget:** the context cost of phase 1 does not grow with the size of the release. One
head per file *type*, not per file. The survey walks every file but only its table enters the
conversation. Run the walk in a subagent and take back the table alone.

Produce, into the plan:

- the file inventory, including the files that belong to no record,
- one head per file type, each with the command that produced it,
- the survey table, including every file checked against whatever the release declares about itself,
- **two mermaid diagrams, both required**. The map has four columns: what ships, what pairs it into
  records, what opens the container, and what each part means. The record model traces every
  object TimeF will hold back to the exact part of the source that states it,
- what one record is, with real values from the heads, and each signal's source dtype beside the
  dtype its spec will declare,
- the task type, the count of tasks per record, and thus whether tasks are added or streamed,
- **which facility stores each answer, annotation and task**, from
  `connector-anatomy.md § Where an answer, an annotation and a task can live`, and why,
- **what already exists for each part of the connector**, and whether this one reuses it,
- the module skeleton and the test plan,
- the numbers the build will produce, each derived from a survey row,
- the decisions taken and the questions genuinely left open.

**Show both diagrams in the conversation.** The record model is the artifact the user accepts or
rejects, which is the whole reason the skill requires it. A diagram written only into the plan was
never shown.

**List what already exists before you name a module.** The parts of the connector are the fetch, the
opener, the base, and the modelling questions the release raises. For each part, name the base or
the existing connector that already does it. Then say whether this one reuses it. Where it does not,
say why not. An author writing one connector cannot see the whole tree, and this is the step that
makes them look.

**Decide, do not collect.** Before you write an assumption, check whether `fidelity.md`, `layout.md`
or an existing connector already answers it. Where one does, **take the decision, cite the rule, and
record it as a decision**. Section 5 of the plan has two headings for exactly this: the decisions
taken with the rule that took each, and the questions genuinely left. Putting a question the repo
already answers in front of the user is not caution. It buries the two questions that are real.

**Derive every predicted number from the survey, and show the derivation.** A predicted record count
that contradicts a table two sections above it is a mistake nobody has to make. One run predicted
55 187 records where summing the eleven per-folder counts in its own survey gives 91 094.

`references/plan-template.md` is the format. Use it as written, so every connector's plan reads the
same. `references/fidelity.md` decides what the design can and cannot do to the data.
`references/layout.md` decides the module skeleton. Read both before you write the plan.

**Tell the user:** what the release holds — the count of records, the shapes the survey found, and
anything odd. Show them the two diagrams, and say that the plan is ready to read.

**Phase 1 is done when** all of these hold. Both diagrams are drawn and shown. Every file in the
inventory is either a node in the map or named as belonging to no record. Every node of the record
model has an inbound edge. The survey covers every file rather than a sample of them. The plan
states a number for records, tasks per record, and expected warnings, each derived from the survey.
A count you cannot state is phase 1 unfinished, and a count that does not follow from the evidence
above it is worse than none.

**Write it down:** sections 1 to 5 of the plan, and set the ledger row to `done`.

**Next:** phase 2, the gate. This one **stops**. Do not begin phase 3 until the user rules.

## Phase 2 — the gate

**Before you start, tell the user:** phase 2 walks the model with them and stops until they rule. It
shows the decisions already taken as one list to object to, and then the open questions one at a
time.

**STOP. Write no connector code.** This is where the model gets agreed, and agreement is not the
same as approval. Walk the model with the user part by part, each part beside its evidence. Then any
one of them can be rejected on its own:

- **The record model diagram**, node by node. It is the fastest way to disagree with a design: an
  object with no arrow into it is invented, and a source with no arrow out is undecided.
- **What one record is**, and the file inventory it comes from.
- **The signals**, their units and their rates, and which header field each came from.
- **The annotations**, and above all **what each is scoped to**. That comes from prose, not from a
  header, so show the sentence.
- **The tasks**: what one question asks, how many there are, and why that is the question the source
  supports.
- **The decisions taken**, as one short list with the rule that took each. The user reads the list
  and overturns anything they disagree with. Do not walk these one at a time.
- **Every open question**, one at a time.

**A question is gate-worthy when all three hold**: the source is silent, **and** no repo rule covers
it, **and** two defensible answers lead to different records, tasks or counts. A question that fails
that test is a decision the author takes and the user can overturn on reading it. One run put 22
questions through this test: 19 were decided against existing rules and 3 reached the user.

For a question that passes, say what the source states. Say what each answer changes, and say what
you recommend. Then let the user rule. Do not resolve one silently.

**Invite disagreement rather than confirmation.** "Does this look right?" gets a yes. Ask instead
which part looks wrong, and be ready to show the head or survey line behind any of them. If the user
cannot check a claim against evidence you produced, that claim is not ready to be agreed.

If the user changes the model, revise the plan and walk it again. Only continue on an explicit yes.

**Phase 2 is done when** all of these hold. The user ruled on every open question individually, and
did not approve the plan as a whole. The user saw the list of decisions taken. No line of the model
is left that the user did not see the evidence for.

**Write it down:** each ruling against the assumption it settles, and the ledger row as
`approved <date>`.

**Next:** on a yes, phase 3, which writes the rulings into the connector's README.

## Phase 3 — the assumptions become the README

**Before you start, tell the user:** phase 3 moves every ruling out of the plan and into the
connector's README, and it needs nothing from them.

Write `README.md` beside the connector from `references/readme-template.md`. One entry for each
assumption and each inconsistency, with three parts: the evidence, the decision, and the state —
**Handled**, **Not built**, or **Open**. An open entry is worth more than a tidy file, because it
names what nobody decided.

Mark any number you measured over the release yourself as *(measured)*. A reader can then tell it
from one copied off the dataset's page. Quote the description sentences from phase 0 beside
`source_url`.

Use the `simple-english` skill on the prose. Do not run it over the head output or the survey table.
Those are evidence, and rewording them destroys them.

Then delete the assumptions from the plan. The README owns them now.

**Tell the user:** the README path, and how many entries are **Open**. An open entry is a decision
still owed, and it is not a surprise at review time.

**Phase 3 is done when** both of these hold. Every assumption and every inconsistency has a README
entry with a state. The plan's assumptions section is empty, because the README owns them now.

**Write it down:** the README path in the plan's ledger row, and the row as `done`.

**Next:** phase 4, the build.

## Phase 4 — build

**Before you start, tell the user:** phase 4 writes the modules the plan named and their tests, and
runs a build. Name the modules, and say how long the first build is likely to take.

Read `references/connector-anatomy.md` for the contract, the base connectors and the task types.
Follow the skeleton the plan states and the rules in `references/fidelity.md` and
`references/layout.md`. Build in the order the format forces: series, then the record, then
annotations, then tasks.

Write the tests the plan named as you go. The modules the plan marked pure need no fixture, which is
the whole point of keeping them pure.

**Phase 4 is done when** all of these hold. Every module the plan named exists. Every test the plan
named is written and passes. `make check`, `make test` and `make test-connectors` are green.

Run a build as you write it:

```bash
uv run timenet-build build <org>/<name> \
    --no-isolation --keep-cache --out ./out
```

- `--no-isolation` runs in the current interpreter. The default builds an environment from the
  connector's `requirements.txt`, which is right for CI and slow while you write.
- `--keep-cache` keeps the raw download. Without it the build deletes the cache directory it
  created after a successful build, and the next run downloads the release again. It deletes only a
  cache it created. A `cache_dir` a caller passed is user-owned and left alone.

**Seed the cache from a copy you already have** rather than downloading twice. The cache directory is
`~/.cache/timenet/cache/<dataset_id>/`, and `ensure_archive` skips the download when it finds a
marker file there named `.<first 8 hex characters of sha256(url)>-<archive name>.extracted`. Put the
extracted release in that directory, create that marker, and the build reads it. `find_dir_containing`
searches with `rglob`, so the tree can sit at any depth.

**Write it down:** any place the built skeleton differs from the planned one, and why. The ledger
row goes to `done`.

**Next:** phase 5, which proves the build against the numbers phase 1 predicted.

## Phase 5 — prove the build

Run in this order and stop at the first failure:

1. `make check`, `make test`, `make test-connectors`.
2. `uv run timenet-build build <id> --no-isolation --keep-cache --out ./out`
3. Load it back: `TimeNet(registry="./out").load("<id>").describe()`.
4. Compare the result against the numbers the plan predicted: records, series per record, tasks, and
   the warning count with its reason. A number that does not match means the plan is wrong or the
   code is. Find out which and say so.

**Write the measured numbers into the plan, beside the predicted ones.** A prediction nobody wrote
the outcome against was never a test. Where the two differ, say which was wrong in the same line.

**A build can fail after `convert` returned cleanly**, and the first time it happens it is
confusing. `convert` gives back a description. The values and the tasks are read afterwards. So
these all surface *after* the step that looks responsible finishes:

- A loader that raises when called.
- A task that names a record that does not exist.
- A span outside its window.
- A stream that yields nothing on a second pass.

If a failure names none of your own modules, it is that stage: check what your loaders do when
called, and what your task stream gives on a second call.

**Phase 5 is done when** every predicted number has a measured number beside it in the plan, and
each difference is explained. A prediction with no outcome written against it was never a test.

**Tell the user:** which checks ran, and which predicted numbers matched.

**Then ship it.** Read `references/shipping.md`. A connector goes up as a stack of small pull
requests. That file states the size to aim for, the seam to cut on, and what never enters the stack.

## Rules that hold in every phase

- **Read every reference with `git show <skill ref>:<path>`, never by working-tree path.** The
  connector is built on a branch chosen for its API, and that branch does not carry this skill. So
  the reference files are simply not on disk. Section 0 of the plan holds the skill ref:

  ```bash
  git show <skill ref>:.agents/skills/add-dataset-connector/references/fidelity.md
  ```

  Reading relative to the skill's own base directory does **not** solve this. That directory mirrors
  the checked-out branch and swaps when the branch swaps.
- **Never `git add` the plan.** `docs/notes/` stays local, like `docs/openspec/`. The head and the
  raw release live there or in the build cache, and neither is ever committed.
- **Never read the data to learn about the data.** Heads show shape, surveys show anomalies.
- **Write down what you measured, and mark it *(measured)*.** A number nobody can re-measure is a
  claim, not evidence.
- Follow AGENTS.md: conventional commits, no `--no-verify`, TimeNet's own errors from
  `timenet.errors`, Google-style docstrings, type hints.

## Further reading

- `references/discovery.md` — heads, surveys, and the map.
- `references/fidelity.md` — what a connector can and cannot do to its source.
- `references/layout.md` — how the modules of a connector divide.
- `references/connector-anatomy.md` — the contract, the bases, the task types, a worked example.
- `references/shipping.md` — how a connector ships, as a stack of small pull requests.
- `references/plan-template.md`, `references/readme-template.md` — the two standard formats.

Repo docs: `docs/connectors.md` and `docs/build.md` are a design proposal. Where they describe a
flat-file `datasets/<org>/<name>.py` layout, the real code uses the folder-package layout. Follow
the code.
