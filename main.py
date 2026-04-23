import numpy as np
import plotly.graph_objects as go

from astrokit.models import CR3BP
from astrokit.models.perturbing_body import create_solar_perturbation
from astrokit.orbit_design.differential_corrector import DifferentialCorrector
from astrokit.orbit_design.initial_guesses import (
    EARTH_MOON_L2_NORTHERN_HALO_GUESS,
    EARTH_MOON_L2_SOUTHERN_HALO_GUESS,
)
from astrokit.simulation import Propagator
from astrokit.utils.constants import EARTH_MOON_MU
from astrokit.utils.plotting import show_figure


