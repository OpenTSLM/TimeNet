# timenet-connectors

Dataset connectors for [TimeNet](../timenet). A connector fetches a dataset's raw artifacts and converts
them into the shared TimeF format. The connector contract itself lives in the `timenet` package
(`timenet.connectors.BaseConnector`).

## Layout

Each concrete connector is a package folder at `datasets/<org>/<name>/`: a `connector.py` exposing a
module-level `CONNECTOR`, an `__init__.py` that re-exports it, and a `dataset.yaml` card beside them
(lowercase names; hyphens in the name become underscores on disk). They are discovered lazily by
dataset id, so there is no central registry. Reusable bases (for the HuggingFace Hub and PhysioNet)
live under `bases/`.

## Build

```bash
timenet-build build timenet/hello-world        # synthetic demo
timenet-build build chengsenwang/tsqa          # a real dataset
```

A connector declares the libraries that its source needs in a `requirements.txt` beside its
`dataset.yaml`. Curation installs them into the environment where the build runs. This keeps
`timenet-connectors` small.
