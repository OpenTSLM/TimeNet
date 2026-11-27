# SWOT signed coastal river surface slope

`wienkers/swot-river-slope@0.2.0` packages signed SWOT river fall and water-surface elevation from 320 SWORD reaches. Negative fall indicates a higher water surface downstream. A negative measurement alone cannot establish backwater or reversed flow.

```bash
uv run timenet-build build wienkers/swot-river-slope@0.2.0
uv run timenet info wienkers/swot-river-slope@0.2.0
```

The connector fetches two SHA-256-pinned assets from the [versioned data release](https://github.com/wienkers/revir/releases/tag/swot-river-slope-v0.2.0). It builds 21,958 pre-issue histories and 87,832 tasks. The 5,133 training issues date to 2023 and the 16,825 validation issues to 2024, with disjoint SWORD regions. Validation histories can include earlier observations on those same validation reaches. The 2025 test split is reserved.

## Observations and quality

The snapshot contains 20,083 observations with distinct `(reach_id, time_str)` pairs across 2023 and 2024. The benchmark accepts `reach_q` 0 or 1. Seven passes are flagged 0 (good) and 20,076 are flagged 1 (suspect, with potentially large errors). Retention requires finite values within these benchmark bounds.

| Field | Accepted interval |
| --- | --- |
| Water-surface elevation | −450 to 8,000 m |
| Elevation uncertainty | 10⁻⁶ to 0.5 m |
| Slope | −0.05 to 0.05 m/m |
| Slope uncertainty | 10⁻¹⁰ to 5 × 10⁻⁵ m/m |

The benchmark intentionally keeps signed measurements, including 14 slopes below the RiverSP product's published minimum of −0.001 m/m. Two of those passes supply five validation targets. Water-surface elevation and fitted slope reference the EGM2008 geoid. Gauge comparisons require datum alignment. Fall and its standard uncertainty multiply native m/m values by 10,000. TimeF uses the equivalent unit `decimeter / kilometer` to preserve the scale of metres per 10 km. [RiverSP product description](https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-docs/web-misc/swot_mission_docs/pdd/D-56413_SWOT_Product_Description_L2_HR_RiverSP_20250224a_RevC_clean_sig_final.pdf) · [PO.DAAC quality flags](https://podaac.github.io/tutorials/notebooks/datasets/SWOT_quality_flag_demo.html)

The reach index defines three disjoint sampling strata. Each split uses 64 coastal reaches within 50 km of the Natural Earth 1:50m coastline, 32 confluence reaches with more than one upstream SWORD reach, and 64 low-gradient reaches. The low-gradient flag uses the bottom quartile of nonnegative 2023 reach-median slopes in the sampling frame. Geographic flags may overlap.

| Cohort | Observations | Negative | Below −1.96u | Below −3u | Reach-days lost / observed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coastal stratum | 6,964 | 222 | 53 | 43 | 218 / 6,818 |
| Low-gradient stratum | 8,149 | 318 | 137 | 112 | 310 / 8,066 |
| Confluence stratum | 4,970 | 46 | 21 | 19 | 43 / 4,899 |
| All strata | 20,083 | 586 | 211 | 174 | 571 / 19,783 |
| Overlapping low-gradient flag | 10,745 | 519 | 177 | 142 | 508 / 10,628 |
| Overlapping confluence flag | 5,275 | 52 | 25 | 23 | 49 / 5,204 |

Here `u` is the reported standard uncertainty in fall. Requiring nonnegative fall drops a reach-day when every retained pass that day has negative fall.

## Forecast tasks

Each UTC day's representative is the retained observation with the smallest slope uncertainty, breaking ties by acquisition time. Select the first daily representative whose acquisition timestamp is 2 to 21 days after a three-day-grid issue, with both lead limits inclusive. This endpoint is conditional on a retained observation. Reversal thresholds are fall below zero and fall plus one standard uncertainty below zero. Steepening requires more than 0.20 m per 10 km above the median of the four latest nonnegative daily visits.

All tasks use pre-issue observations and input annotations for issue time, location, reach length, stratum and baseline. The numeric forecast includes the selected pass time and its target values. The forecast and yes/no prompts describe the same daily observation. The `model_inputs` helper, available from `timenet_connectors.datasets.wienkers.swot_river_slope`, exports the prompt, input context and scoped history.

Acquisition UTC proxies availability time. The dataset includes calibration-orbit observations from before the start of the science orbit on 21 July 2023. The nominal overpass hint uses the 20.86455-day repeat period and past science-orbit passes where available. The M2, S2, K1 and O1 angles and spring index describe astronomical forcing. Local tide phase requires gauge data or a calibrated hydrodynamic prediction. [SWOT science orbit](https://www.aviso.altimetry.fr/en/missions/current-missions/swot/orbit/science-orbit.html)

## Source provenance

Hydrocron queries used the `SWOT_L2_HR_RiverSP_D` collection and the reach IDs in the index, with acquisitions from 1 January 2023 to 31 December 2024. The release CSV retains `reach_id,time_str,wse,wse_u,slope,slope_u,reach_q` and deduplicates identical reach–timestamp pairs by lowest slope uncertainty. The snapshot and reach index are the reproducible build inputs. A later Hydrocron query can return reprocessed measurements. [Hydrocron API](https://podaac.github.io/hydrocron/timeseries/) · [SWOT RiverSP Version D](https://doi.org/10.5067/SWOT-RIVERSP-D) · [SWORD v17b](https://doi.org/10.5281/zenodo.15299138)

The release's [data terms and attribution](https://github.com/wienkers/revir/releases/download/swot-river-slope-v0.2.0/DATA_LICENSE.md) cover NASA/CNES SWOT observations and the adapted SWORD v17b index. The connector code uses TimeNet's MIT licence.
