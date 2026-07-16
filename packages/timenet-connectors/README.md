# timenet-connectors

Dataset connectors for [TimeNet](../timenet). A connector fetches a dataset's raw artifacts and converts
them into the shared TimeF format. The connector contract itself lives in the `timenet` package
(`timenet.connectors.BaseConnector`).

## Layout

Concrete connectors live at `datasets/<org>/<name>.py` (lowercase) and each exposes a module-level
`CONNECTOR`. They are discovered lazily by dataset id, so there is no central registry. Reusable bases
(e.g. for the HuggingFace Hub) live under `bases/`.

## Curate

```bash
timenet-curate build timenet/hello-world        # synthetic demo
timenet-curate build chengsenwang/tsqa          # a real dataset
```

Some connectors need optional extras, e.g. `pip install 'timenet-connectors[huggingface]'`.
