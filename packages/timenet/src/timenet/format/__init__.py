"""The TimeF on-disk format contract: filenames, layout templates, and pinned Arrow schemas.

The writer and the reader both use this module. They are otherwise independent. The definitions that
both sides of the round-trip must agree on live here, so neither side imports the other.
"""
