"""AstroKit package for orbital mechanics and astrodynamics."""

from . import control
from . import models
from . import orbit_design
from . import simulation
from . import utils

from .control import *
from .models import *
from .orbit_design import *
from .simulation import *
from .utils import *

__all__ = [
    "control",
    "models",
    "orbit_design",
    "simulation",
    "utils",
]
_exported = []
for _pkg in (control, models, orbit_design, simulation, utils):
    for _name in getattr(_pkg, "__all__", []):
        if _name not in _exported:
            _exported.append(_name)
__all__.extend(_exported)

del _exported
