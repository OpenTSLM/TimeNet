# scooby-team/swiss-grid-anomaly-cot

Weekly windows of the Swiss electricity system, 2020-2025, with explained load anomalies. Every
label is computed by rules from public data; no label comes from a model.

## Records

One Record per ISO week (312 weeks), with two sources of hourly signals:

- **Swissgrid, Swiss control block** (MW): end-user consumption, total consumption, production,
  and the net import from Austria, Germany, France and Italy. The connector sums Swissgrid's
  quarter-hourly energies (kWh) into hourly mean power.
- **MeteoSwiss SwissMetNet** (12 stations): air temperature (°C) and global radiation (W/m²), each
  a mean weighted by the 2023 BFS population of the cantons around each station.

A week has 168 hours, or 167 / 169 in the weeks of a clock change. Everything is stored in UTC.

## Tasks

| task | type | target |
| --- | --- | --- |
| cause | `ClassificationTask` | cause of the week's largest anomaly, or `none` (9 labels) |
| days | `TemporalLocalizationTask` | every anomalous day of the week, possibly none |
| exchange | `AnswerTask` | net importer or exporter, and through which borders |

Each task carries a rationale written by fixed templates from the computed facts. A check rejects
any number in the text that is not one of the facts, which the record keeps as a JSON annotation.

**The prompts name no holiday.** They give the week's dates and describe the series. Most anomalous
days are holidays, and rules that read only the holiday calendar, with no load data, reach a cause
accuracy of 0.98 and a day F1 of 0.81 on the 2025 test weeks: with the calendar in the prompt, a
model could answer without reading the series. The holidays stay in each record as
`public_holiday` annotations, and the rationales name them.

## How the labels are computed

1. A daily linear baseline (weekday type × month, heating and cooling degree days, lagged heating
   degree days, radiation, trend), fitted on the ordinary days of 2020-2023 only, gives the
   expected end-user consumption. Its MAPE is 2.3 % on the ordinary days of the 2025 test year.
2. A day is anomalous when its robust z-score is at least 3 in absolute value.
3. Ordered rules give the cause: `covid_lockdown`, `christmas_newyear` (22 Dec - 6 Jan),
   `public_holiday` (a holiday for at least 20 % of the population), `bridge_day`,
   `summer_holidays` (14 Jul - 15 Aug). Holidays come from `holidays` 0.108 per canton.

No day of 2020-2025 lies above expectation at this threshold, so `cold_snap`, `heatwave` and
`unexplained` are in the label set but never occur.

## Splits

208 train weeks (2020-2023), 52 validation (2024) and 52 test (2025). The baseline is fitted on
train days only and never sees a validation or test day.

## Source quirks the connector handles

- Swissgrid writes Swiss civil time. In 2020-2024 each label is the **end** of its quarter hour;
  in 2025 it is the **start**. The convention is detected from the first label of each file.
- On the fall-back day the 02:xx labels appear twice. 2020 writes the fall-back instant as 03:00
  CEST, 2021-2024 as 02:00 CET. Every label is checked against the wall-clock reading of its
  instant, so a missing, extra or shuffled row raises.
- MeteoSwiss writes UTC, labelled at the end of each hour.

## Limitations

- **License.** Swissgrid states no license for these files, so the card says `NONE`. MeteoSwiss
  data is CC BY 4.0.
- **Few causes in the test year.** 2025 holds 3 of the 9 causes (`none`, `public_holiday`,
  `christmas_newyear`), and 19 anomalous days in 8 weeks, so scores on the days task are noisy.
- **Numbers in the rationales.** Some quoted numbers (the expected load, z-scores, the weather
  effect) come from the baseline and cannot be read exactly from the signals.
- **Revisions.** Swissgrid re-published 2023 in March 2026, and MeteoSwiss rewrites the station
  files once a year. A revised file fails its SHA-256 pin, so the build stops instead of silently
  changing the labels.
