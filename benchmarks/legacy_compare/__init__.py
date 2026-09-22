"""Compare the v0.1 Parquet control plane with the DuckDB control plane on real datasets.

The scripts here run under two environments: the ``main`` branch (v0.1) and this stack. Only
:mod:`read_bench` runs under both; it touches nothing but the public reader API that the two share.
"""
