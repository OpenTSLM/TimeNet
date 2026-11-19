# Fixture provenance

The numerical files contain three observations from source revision
`00281e2d86058286d5548b15a7670e8eda57ef62`. Economy uses January through March 1987,
reordered March/January/February to exercise chronological sorting. Its OT values
match the source and equal Exports minus Imports. Energy uses April 5, 12 and 19,
1993, with all native numerical columns.

All four textual files are synthetic, source-shaped parser fixtures. They exercise
quoted commas/quotes, embedded newlines, repeated coverage dates, missing markers,
empty fields and text that predates measured history. They are not historical reports
and provide no publication or availability evidence.

Task tests generate 65 synthetic observations using these column shapes. Their
controlled OT values test QA boundaries and withheld forecast targets. The other
copied channel values are placeholders, not realistic multivariate observations.
Source parity is checked separately against all six complete pinned CSVs by the
auxiliary reproduction script.
