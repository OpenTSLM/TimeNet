"""The TimeF on-disk format contract: filenames, layout templates, and pinned Arrow schemas.

Shared by the writer and the reader, which are otherwise independent of each other. Definitions that
both halves of the round-trip must agree on live here so neither has to import the other.
"""
