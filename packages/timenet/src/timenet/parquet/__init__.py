"""The Parquet machinery the values plane writes through.

Encodings are pinned by column role and, for the waveform values column, chosen per modality from
measured cardinality. Parts rotate on a byte budget. Both are the reason the values plane stays on
pyarrow rather than moving into the database with the control plane: no other writer in reach
exposes per-column physical encoding control.
"""
