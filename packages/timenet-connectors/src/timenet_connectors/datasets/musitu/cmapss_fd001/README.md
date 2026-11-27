# NASA C-MAPSS FD001 simulated-engine predictive maintenance

**Connector ID:** `musitu/cmapss-fd001` — TimeNet Round 1 contribution by MUSITU AXIOM Sentinel. **This is a research benchmark, not an operational aircraft or industrial safety system.**

## Source and access

NASA Prognostics Center of Excellence, C-MAPSS turbofan degradation simulation (Saxena et al., PHM 2008). Public provenance: <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>. Versioned source archive: <https://zenodo.org/records/15346912>, expected MD5 `79a22f36e80606c69d0e9e4da5bb2b7a`.

The connector downloads and verifies the source archive on demand; it does **not** ship raw telemetry or the checkpoint weights. For offline research, point `CMAPSS_DATA_DIR` to a directory containing `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`. **Explicit local files bypass the remote checksum**, so the person supplying them must verify provenance separately. If remote downloads are blocked, the download method raises rather than silently substituting synthetic data.

The Zenodo record does not assert a specific SPDX license. The dataset card uses `OTHER`; public access does not automatically establish redistribution rights. Confirm the applicable source terms before redistributing a transformed copy.

## Task and representation

- One `Record` per **engine × 30-observation window**, sampled every 15 engine cycles for the 100 simulated training engines. Each record contains 24 input signals: three operating settings and 21 sensor channels, with one value per cycle on an ordinal axis. All signals use only information observed at or before the endpoint.
- One `ScalarPredictionTask` per record; target is remaining useful life in cycles (RUL), measured from the official engine lifetime for training engines.
- **Training and validation are grouped by engine**, not window: every fifth development engine (`sorted(train_ids)[::5]`) is reserved for validation; no development engine appears in both partitions. Each of the 100 official NASA test engines contributes **only its last observed endpoint**. The official RUL offset labels are used only as targets for test tasks.
- A record's annotations identify engine ID, cohort and observation cycle, not input model signals. The source may have differing cycle lengths; groups must have complete, ordered cycles starting at 1. Missing/invalid source values, duplicate identifiers, nonintegral cycles and invalid official RUL labels cause explicit errors.
- `STRIDE=15` and `WINDOW=30` define the connector's benchmark representation. The separate Sentinel model experiments may use different stride and validation group choices; their reported scores should not be mistaken for a TimeNet-trained model result.

## Reproducibility and verification

In the checked-out TimeNet workspace:

```sh
uv sync --locked --all-packages
uv run python -m pytest -q packages/timenet-connectors/src/timenet_connectors/datasets/musitu/cmapss_fd001/tests
uv run timenet-build build musitu/cmapss-fd001
```

Fixture tests use **entirely synthetic telemetry** and do not fetch or redistribute NASA data. They cover split isolation, scalar RUL tasks, exact past-only sensor values, malformed source rows, invalid official labels and no-network local input. A separate [Sentinel GitHub Actions run](https://github.com/evansmusitu/musitu-timenet-sentinel/actions/runs/44984278576) has run real-source conversion on the official TimeNet API; an upstream full build and licensing review remain open.

## Evidence and limitations

- [Interactive Sentinel demo](https://evansmusitu.github.io/musitu-timenet-sentinel/)
- [Full reproducible project](https://github.com/evansmusitu/musitu-timenet-sentinel)
- [Genuine OpenTime 5B pilot and limitations](https://github.com/evansmusitu/musitu-timenet-sentinel/blob/main/docs/OPENTIME_5B_VERIFIED_GPU_RESULTS.md) (F1 from 0.000 to 0.324 on 100 official test engines after one epoch/16 training windows)

The separate compact temporal QA model achieved warning-head F1 0.960 on the official 100-engine endpoint set, while a classical baseline achieved F1 0.889. **Neither result is an OpenTime 5B result**, and neither proves operational performance. The OpenTime pilot remained worse than the independent baselines. No real-aircraft or cross-fleet validation is claimed.
