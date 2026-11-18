# Lima hourly PM2.5 forecasting

This connector builds `airaware/lima-pm25-forecast@0.1.0` from SENAMHI's
public export, *Monitoreo de los contaminantes del aire en Lima Metropolitana*.
SENAMHI licenses the source database under
[ODC-By-1.0](https://opendatacommons.org/licenses/by/1-0/). AirAware provides
the hourly deduplication, chronological benchmark, and forecasting tasks.
The connector code uses the repository's MIT license.

## Tasks and splits

The model receives 24 consecutive hourly PM2.5 measurements from one of seven
provisional station identities and predicts the next six values. TimeF records
contain only the history. Respective forecast values are ordered numeric targets.
The source CSV uses a pinned SHA-256, as specified on the dataset card. Exact
repeat readings share a single retained reading. Conflicting finite reports for
the same hour are excluded along with windows that require missing values.

Validation starts on March 16, 2022 at 08:00 UTC and test starts on February
25, 2023 at 18:00 UTC. Windows crossing either boundary are purged. The published
benchmark contains 237,397 training, 50,706 validation, and 50,789 test tasks.
Use validation for model selection and reserve test for final evaluation.

The timestamps use published hourly labels interpreted as UTC under AirAware's
operating policy. The concentration unit in µg/m³ is inferred. The position of
the hour within the reporting interval, per-reading quality flags, and publication
delays are unknown. The benchmark covers the monitored sites in the pinned export
and supports retrospective evaluation.

## Build and verify

With the TimeNet development environment installed:

```sh
timenet-build build airaware/lima-pm25-forecast@0.1.0 \
  --local-registry /path/to/local-registry
```

For a build without network access, place the verified original at
`${TIMENET_CACHE}/downloads/airaware/lima-pm25-forecast/senamhi-lima.csv`,
use the same local registry as the remote registry in the build command, and
pass `--no-isolation` to use the already installed environment. The converter
checks the exact CSV bytes independently of the cache check. Open the result
with `TimeFReader` and run `verify()` before using the release.
