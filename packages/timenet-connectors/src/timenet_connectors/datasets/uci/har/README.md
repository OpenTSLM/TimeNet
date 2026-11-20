# UCI-HAR mapping

The connector preserves the UCI-HAR train/test window split. Each source row becomes one
record with nine source-named signals: total acceleration, body acceleration, and body
gyroscope on the x, y, and z axes. Values remain raw source values as float64. Acceleration
uses standard gravity and gyroscope uses radians per second; every window has 128 samples at
50 Hz with a relative origin of zero.

Records retain the source split, one-based row, native subject id, and numeric activity id.
Tasks use the exact activity names from `activity_labels.txt`, referring to the registered
six-label vocabulary.

The dataset card uses CC BY 4.0 as listed on the current official UCI page linked in
`license_url`. Legacy release material contains an older non-commercial notice; the card
follows the current repository listing.

The download cache preserves the official outer archive as `uci-240.zip`. Its fixed nested
`UCI HAR Dataset.zip` member is streamed to a temporary file, then only the files needed for
conversion are written into the cache.
