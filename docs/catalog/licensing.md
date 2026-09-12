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

- `license`: an SPDX identifier, such as `MIT`, `Apache-2.0`, or `CC-BY-4.0`, or `other` for a license
  outside that list. The dataset-card schema limits this field to a fixed set of SPDX identifiers plus
  `other`, kept in sync with `timenet.types.License`, so a listed value is always one you can look up
  on [SPDX](https://spdx.org/licenses/). `other` requires the card's `license_url` field, a link to
  the full license text.
- `source_url`: where the data comes from. Open it to read the upstream terms in full. This field is
  optional, so a purely synthetic dataset may leave it out.
- `access`: whether the data is `open`, `credentialed`, or `restricted` (`timenet.types.Access`).
  Defaults to `open`.
- `access_url`: where to get access, such as a data use agreement or credentialing page. Required
  when `access` is not `open`.

The [dataset catalog](datasets.md) lists the license and source per dataset.

## Before you download

Some sources only grant credentialed access. PhysioNet, for example, asks you to accept a data use
agreement and sign in before you pull a record. A dataset built from such a source sets `access` to
`credentialed` (or `restricted` for a case-by-case grant, such as an AWS-IAM grant or a direct
request) and points `access_url` at where to get it. TimeNet enforces this in code: `load` and
`download` raise `TimeNetAccessError` when a credentialed or restricted dataset is read from a hosted
registry (S3, HTTP, or the `timenet://` service) instead of a local one. Follow `access_url` to get
your own credentials, then build the dataset locally.

## Redistribution

Whether TimeNet vendors a dataset's bytes depends on its `access`:

- Open datasets: a registry (local, S3, HTTP, or the hosted `timenet://` service) hosts the converted
  TimeF bytes, so `load` and `download` read from the registry rather than refetching `source_url` on
  each call. If you redistribute a converted dataset, keep its original license with it.
- Credentialed and restricted datasets: TimeNet never hosts or redistributes these. `load` and
  `download` raise `TimeNetAccessError` for any registry but a local one, so you build the dataset
  yourself after getting your own access. The source's terms, such as a PhysioNet data use agreement,
  may forbid redistribution outright, not just require attribution.
