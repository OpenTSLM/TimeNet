# Hill of Towie SCADA Forecasting

`res/hill-of-towie-scada@2.0.0` covers all 21 turbines from January 2016 through April 2026.
It uses annual ten-minute SCADA archives from Zenodo release 2.1.0. Each task predicts six measured
active-power values in kW from the previous 36 ten-minute readings of active power, active wind
speed, and generator RPM. [Source release and CC BY 4.0 attribution](https://zenodo.org/records/22662930)

## Dataset split

All boundaries are UTC, with an inclusive start and exclusive end for physical measurement
intervals. Source timestamps label the **end** of each ten-minute interval, so a reading stamped
at a split boundary belongs to the preceding interval. An hourly task is eligible when its entire
six-hour input and one-hour target fit in one calendar month and one row of the table below. The
same boundaries apply to every turbine. An input needs all three channels, and a target needs
only power. Source gaps and blanks remain nulls in the shared records.

| TimeF split | `evaluation_cohort` | Start UTC | End UTC | Windows | Rationale |
| --- | --- | --- | --- | ---: | --- |
| Train | `historical_train` | 2016-01-01 00:00 | 2024-10-01 00:00 | 1,564,202 | Retains several seasons, the AeroUp rollout, and TuneUp. Ends after the documented late-summer 2024 curtailment period. |
| Validation | `post_tuneup_validation` | 2024-10-01 00:00 | 2025-01-01 00:00 | 45,511 | Tunes on later data following the TuneUp campaign, ahead of the 2025 holdout. |
| Test | `pre_commissioning` | 2025-01-01 00:00 | 2025-10-28 13:00 | 141,619 | Main score for forecasting before documented Dynamic Yaw commissioning. |
| Test | `dynamic_yaw_commissioning` | 2025-10-28 13:00 | 2026-01-07 13:00 | 34,857 | Covers commissioning, tests, and changes to turbine roles. |
| Test | `dynamic_yaw_west_trial` | 2026-01-07 13:00 | 2026-02-03 18:00 | 13,434 | Starts at the formal west trial boundary in the RES configuration. |
| Test | `dynamic_yaw_west_and_east_trial` | 2026-02-03 18:00 | 2026-05-01 00:00 | 42,614 | Starts at the formal east cohort boundary and ends with the archive. |

Use the main test cohort for the headline score and report the three later cohorts separately.
September 2025 lost 44.34% of expected turbine intervals, including a 12.73-day gap across every
turbine. Report a main-test sensitivity score with September omitted to show how that gap affects
coverage. The Dynamic Yaw schedule alternates 50 minutes on and 50 minutes off. The cohort labels
describe dates. Per-turbine exposure, controller state and wake effects require the RES role
definitions and controller telemetry. [Trial configuration](https://github.com/resgroup/hill-of-towie-open-source-analysis/blob/main/scripts/wfc_analysis_2026/wind_up_config/HOT_dynamic_yaw.yaml)
· [Commissioning and curtailment dates](https://github.com/resgroup/hill-of-towie-open-source-analysis/blob/main/scripts/wfc_analysis_2026/wind_up_config/exclusion_periods/HOT_DY.yaml)

## Measurement and operating changes

| When | What changed |
| --- | --- |
| August 2016 | The turbine SCADA schema removed the low-speed power reduction fields. |
| July 2017 | The schema added boost availability. |
| August 1, 2017 and April 10–11, 2018 | The secondary anemometer correction started on T14 and T17 first. Most other turbines followed on April 10, 2018, with T01 and T06 on April 11. The historical audit found a median +11.91 percentage-point step in raw secondary wind speed relative to primary. The active-to-primary ratio was unchanged to two decimal places. |
| September 23–29, 2021 | T13 underwent the first AeroUp retrofit. |
| April 2022 | The schema added actual wind direction, boost activation, and turbine operating state. The three forecast channels span the earlier and later schemas. |
| July 14, 2022–May 26, 2023 | RES rolled out AeroUp to the rest of the fleet. Use the per-turbine first and last work dates to distinguish pre-upgrade, installation, and post-upgrade windows. |
| March 14–August 31, 2024 | TuneUp began on T02, T03, T06, T08, T09, T13, T16, and T20 on March 14. T21 joined on May 2. RES also flags heavy curtailment from August 9 through September 28. |
| October 28, 2025 onward | Dynamic Yaw commissioning began. RES lists turbine role changes and controller configuration tests in October, November, and December. |
| January 7 and February 3, 2026 | The formal west and east trial boundaries were at 13:00 and 18:00 UTC, respectively. Dynamic Yaw changes the relationship between inflow, yaw, and measured power. |

RES records farm-wide pre-campaign tests from October 28 at 13:00 to November 17 at 13:45 UTC.
It records a deliberate north-offset test on T15 and T18 on November 19–20, a change in T06's
role on December 2, and changes in T19's and T21's roles on December 11. These dates are why the
main test stops in October 2025. [RES commissioning log](https://github.com/resgroup/hill-of-towie-open-source-analysis/blob/main/scripts/wfc_analysis_2026/wind_up_config/exclusion_periods/HOT_DY.yaml)

The dataset targets actual delivered power, including valid readings during curtailment and
changes in operation. The connector uses the active wind-speed channel throughout. Use the same
validation period to compare full-history training with a June 2023 onward training run, after
the AeroUp rollout. [SCADA data dictionary](https://zenodo.org/records/22662930/preview/Hill_of_Towie_data_dictionary.md?include_deleted=0)
· [AeroUp dates by turbine](https://zenodo.org/records/22662930/files/Hill_of_Towie_AeroUp_install_dates.csv?download=1)
· [RES campaign summary](https://zenodo.org/records/22662930)

Zenodo release 2.0.0 extended SCADA from August 2024 to April 2026 and added separate fastlog,
LiDAR, and controller telemetry. Release 2.1.0 updated the documentation. The forecasting
records here use the ten-minute SCADA tables. [Source changelog](https://zenodo.org/records/22662930/preview/CHANGELOG.md?include_deleted=0)

## Build and read the forecast windows

From the TimeNet checkout, run `make sync`, then build into a local registry:

```bash
uv run timenet-build build res/hill-of-towie-scada@2.0.0 --no-isolation
```

The source download is approximately 15 GB. The connector reads two CSV tables per month directly
from the pinned archives. In the resulting TimeF release, a record holds one turbine and one
month segment. Each forecast carries an input `scope` for the 36 historical readings and a
target interval for the six observed power values. Apply the input scope before passing values to
a model. Reading a whole shared record would include readings after that task's cutoff.

```python
from timenet import TimeNet
from timenet.types import TimeInterval

with TimeNet().open("res/hill-of-towie-scada@2.0.0") as reader:
    task = next(reader.iter_tasks(split="train", batch_size=1))
    assert isinstance(task.scope, TimeInterval)
    assert task.targets is not None
    (target,) = task.targets
    assert isinstance(target, TimeInterval) and target.time_series
    (power,) = target.time_series
    context = {
        signal.name: signal.slice(task.scope).to_numpy()
        for signal in task.inputs[0].signals
    }
    observed_power_kw = power.slice(target).to_numpy()
```
