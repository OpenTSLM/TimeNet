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

Every dataset ships a card (`dataset.yaml`) with two fields that tell you what applies:

- `license`: an SPDX identifier, such as `MIT`, `Apache-2.0`, or `CC-BY-4.0`. The dataset-card schema
  limits this field to a fixed set of SPDX identifiers, kept in sync with `timenet.types.License`, so
  the value is always one you can look up on [SPDX](https://spdx.org/licenses/).
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
