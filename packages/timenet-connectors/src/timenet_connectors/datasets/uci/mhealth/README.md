# MHEALTH

`uci/mhealth` preserves each official `mHealth_subject<N>.log` as one 50 Hz record.  Its 23
measurement columns are retained as float64 streams: chest, left-ankle, and right-lower-arm
acceleration; two chest ECG leads; left-ankle and right-lower-arm gyroscope; and both three-axis
magnetometers.  Source label `0` is retained alongside labels `1` through `12` as timed activity
annotations and scoped classification tasks.

The official README specifies acceleration in m/s², gyroscope in deg/s, and ECG in mV.  It calls
magnetometer values “local”; this connector retains their raw values without claiming a physical
magnetic-field unit.  The official column list repeats column 13; the connector uses the observed
24-column order and preserves all three left-ankle magnetometer components.

The source is [UCI MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset), licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
