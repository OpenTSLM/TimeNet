# PAMAP2

`uci/pamap2` keeps each official Protocol or Optional `.dat` file as one record. It retains the
source timestamp, activity including `0`, heart rate, and all 51 IMU columns as 52 lazy float64
measurement streams. The named streams retain the source units: heart rate in bpm; temperature in
°C; both ±16g and ±6g accelerometers in m/s²; gyroscopes in rad/s; and magnetometers in μT. Source
`NaN` values and the published invalid orientation columns are retained. The source documents 100 Hz
data, while source timestamps remain authoritative.
