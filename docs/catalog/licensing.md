---
icon: lucide/scale
description: "How TimeNet's code license differs from the licenses of the datasets it fetches."
tags:
  - catalog
  - licensing
---

# Dataset licensing

TimeNet's own code is MIT licensed. That license does not extend to the datasets TimeNet fetches and
converts. Each dataset keeps the license set by whoever published it.

## Where a dataset's license lives

Every dataset ships a card (`dataset.yaml`) with fields that tell you what applies:

- `license`: an SPDX identifier, such as `MIT`, `Apache-2.0`, or `CC-BY-4.0`. The dataset-card
  schema limits this field to a fixed set of values, kept in sync with `timenet.types.License`. Most
  are license identifiers you can look up on [SPDX](https://spdx.org/licenses/). Two are escape
  hatches:
    - `OTHER`: a known license outside the SPDX list, such as a Hippocratic License 3.0 build with
      modules. Pair it with a `license_url`.
    - `NONE`: SPDX's value for a source whose publisher states no license at all. It does not mean
      the data is free of restrictions: its terms of use are unknown, so check with the publisher
      before you use or redistribute it.
- `license_url`: where to read the full license text. Required when `license` is `OTHER`.
- `source_url`: where the data comes from. Open it to read the upstream terms in full. This field is
  optional, so a purely synthetic dataset may leave it out.

The [dataset catalog](datasets.md) lists both fields per dataset.

## Before you download

Some sources only grant credentialed access. PhysioNet, for example, asks you to accept a data use
agreement and sign in before you pull a record. TimeNet does not remove that step. When a dataset's
`source_url` points at a gated source, follow that source's terms and use your own credentials.

## Redistribution

TimeNet does not vendor dataset bytes. It fetches each dataset from its `source_url`, so you get the
data under the license the source grants. If you redistribute a converted dataset, keep its original
license with it.
