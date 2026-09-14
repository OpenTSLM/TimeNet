"""Time axes for the TimeF model.

The hierarchy objects a builder populates (``Record``, ``Source``, ``Signal``, ``Task``,
``Annotation``) live in :mod:`timenet.control_plane.model`, which imports the axes from here.
"""

from timenet.dataset.axis import AxisType, IrregularAxis, OrdinalAxis, RegularAxis, TimeAxis


__all__ = ["AxisType", "IrregularAxis", "OrdinalAxis", "RegularAxis", "TimeAxis"]
