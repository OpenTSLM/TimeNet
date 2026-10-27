# Shoaib mapping

Each native participant CSV is one record. It has the five original `time_stamp` columns as string
series, sixty sensor channels, and the native per-row activity label as a string series. Timestamp
tokens such as `1.39E+12` are retained verbatim; they are rounded source tokens and do not define an
invented precise wall-clock timeline. The measurement axis is the source README's nominal 50 Hz
relative sample axis.

Android's primary sensor documentation defines accelerometer and linear acceleration as m/s²,
gyroscope as rad/s, and magnetic field as μT. Values are never rescaled. The official archive did
not provide a machine-readable redistribution license in the inspected material, so the card uses
`other` and links to the University of Twente source.
