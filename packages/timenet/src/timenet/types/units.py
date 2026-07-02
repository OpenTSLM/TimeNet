"""The shared pint unit registry.

TimeNet uses `pint <https://pint.readthedocs.io>`_ for all physical units instead of a hand-rolled
system. One process-wide :data:`ureg` owns every definition and conversion; units built from any other
registry will not compare or convert cleanly, so always reference units through :data:`ureg`
(``ureg.hertz``, ``ureg.millivolt``, ...).
"""

import pint


ureg = pint.UnitRegistry()

# Non-physical units pint does not ship. `beat` is a dimensionless count given its own base dimension
# so heart-rate units stay distinct from plain frequencies.
ureg.define("beat = [beat]")
ureg.define("bpm = beat / minute")

# Make `ureg` the registry that unpickled units resolve against. Without this, a pint.Unit pickled in
# one process (or by TimeFReader) comes back bound to pint's default application registry and raises
# "Cannot operate with Unit and Unit of different registries" when compared, breaking round-trips and
# multiprocessing DataLoaders.
pint.set_application_registry(ureg)
