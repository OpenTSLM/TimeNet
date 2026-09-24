# CAPTURE-24 mapping

Each participant CSV.GZ is one native record. X/Y/Z remain float64 acceleration in g at the source
100 Hz, and original timestamp and annotation text remain string series. Contiguous annotation runs
become whole-source sample scoped classification tasks; no windows, resampling, or label aggregation
is added. Participant age group/sex and all six dictionary mappings are preserved as annotations.
