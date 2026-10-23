# Verification: `org/name`

<!--
Copy this file into the connector folder as VERIFICATION.md, in the same commit as the connector.
Replace every TODO. docs/connector-verification.md explains each item and how to catch it, and
test_every_connector_ships_a_verification_record checks the format.
-->

- **Checklist version:** 2

Status is one of ✅ Done, ☑️ Done once, ⚠️ Open or ⚪ N/A, or a combination such as
✅ Done / ⚠️ Open. Every row needs evidence, and a ✅ Done row names its test or build check in
backticks. "Choosing a status" in `docs/connector-verification.md` gives the rules. When in doubt,
pick the weaker status.

## References

TODO: the independent copies of the data that exist for this source, such as another release, the
authors' preprocessed arrays or code, or the upstream data.

## Fields

Every file, column and field of the source, and every input the authors' code reads, with its role
in the build or the reason it is dropped. Then every value the connector added that the source
doesn't hold, with its reference, or "assumption". Items 11, 12 and 48 cite this table.

| Field | Where it comes from | Role in the build | Reference, or why it is dropped |
| --- | --- | --- | --- |
| TODO | TODO | TODO | TODO |

## Checklist

### Step 1. Source

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 1 | Wrong release or version | TODO | TODO |
| 2 | Unpinned input | TODO | TODO |
| 3 | Partial, corrupt or changed download | TODO | TODO |
| 4 | Wrong files from the archive | TODO | TODO |
| 5 | Source structure misread | TODO | TODO |

### Step 2. Coverage

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 6 | Counts differ from the source | TODO | TODO |
| 7 | Rows dropped silently | TODO | TODO |
| 8 | Wrong filter or threshold | TODO | TODO |
| 9 | Duplicate records | TODO | TODO |
| 10 | Missing or extra channels | TODO | TODO |
| 11 | Source information left unused without a note | TODO | TODO |
| 12 | Information the connector made up | TODO | TODO |

### Step 3. Read path

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 13 | Lazy loader reads the wrong bytes or record | TODO | TODO |
| 14 | Precision lost on a dtype change | TODO | TODO |
| 15 | Overflow or wraparound in integer conversion | TODO | TODO |
| 16 | Write and read don't round-trip | TODO | TODO |
| 17 | Build isn't deterministic | TODO | TODO |

### Step 4. Channel meaning

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 18 | Raw ADC counts stored without gain and baseline | TODO | TODO |
| 19 | Wrong units or scale | TODO | TODO |
| 20 | Swapped or mislabeled channels | TODO | TODO |
| 21 | Flipped sign convention | TODO | TODO |
| 22 | Missing-value markers stored as values | TODO | TODO |
| 23 | Wrong categorical codebook | TODO | TODO |
| 24 | Implausible upstream values passed through without a note | TODO | TODO |

### Step 5. Time

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 25 | Wrong sample rate | TODO | TODO |
| 26 | Wrong start time or epoch | TODO | TODO |
| 27 | Wrong time zone | TODO | TODO |
| 28 | Daylight-saving errors | TODO | TODO |
| 29 | Gaps or missing values filled or dropped silently | TODO | TODO |
| 30 | Artifacts from resampling or interpolation | TODO | TODO |
| 31 | Off-by-one sample | TODO | TODO |
| 32 | Sources or channels out of sync | TODO | TODO |

### Step 6. Identity and joins

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 33 | Join relies on row order | TODO | TODO |
| 34 | Off-by-one id | TODO | TODO |
| 35 | Ids or join keys not unique | TODO | TODO |
| 36 | Subject metadata on the wrong recording | TODO | TODO |
| 37 | Join silently drops rows | TODO | TODO |

### Step 7. Labels and targets

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 38 | Wrong mapping from class code to name, or wrong vocabulary order | TODO | TODO |
| 39 | Label distribution differs from the paper | TODO | TODO |
| 40 | Wrong granularity, such as per record instead of per epoch | TODO | TODO |
| 41 | Derived target computed wrongly | TODO | TODO |
| 42 | Label misaligned with, or contradicted by, its signal | TODO | TODO |

### Step 8. Splits

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 43 | Not the authors' splits | TODO | TODO |
| 44 | Leakage across splits | TODO | TODO |
| 45 | Splits regenerated without a fixed seed | TODO | TODO |

### Step 9. Task construction

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 46 | Wrong task type | TODO | TODO |
| 47 | Task can't be solved from its inputs | TODO | TODO |
| 48 | A field in the wrong role | TODO | TODO |
| 49 | Wrong prompt, or one without a source | TODO | TODO |
| 50 | Series placed at the wrong point in the prompt | TODO | TODO |
| 51 | An input reveals the target | TODO | TODO |
| 52 | The target leaks through an id, metadata or another field | TODO | TODO |
| 53 | An input carries information from after the target | TODO | TODO |
| 54 | Answer format doesn't match what the evaluation expects | TODO | TODO |

### Step 10. Card and provenance

| # | Error | Status | Evidence, or what's left |
| --- | --- | --- | --- |
| 55 | Wrong license | TODO | TODO |
| 56 | Wrong domain, tags or description | TODO | TODO |
| 57 | A README number that doesn't match the build | TODO | TODO |
| 58 | A README claim the data doesn't support | TODO | TODO |
| 59 | Known limitations left undocumented | TODO | TODO |

## Open items

TODO: the open items, most important first, including decisions for maintainers.

## Plan

TODO: the checks that would close the open items.
