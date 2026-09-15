"""The paper measurement harness, ported into the spike to compare three representations.

The lane protocol and the six Original lanes are copied unchanged from the reference harness so
the timing, the delivered item and the checksum are the same code on both sides of the
comparison. The Original lanes read their releases directly and depend on no SDK, so they give
the same numbers whichever environment runs them, which is what makes a stock-TimeF run and a
DuckDB run comparable: Original appears in both and calibrates them.
"""
