# General imports
import numpy as np

# Import Astrokit Tool
from astrokit import *

model = CR3BP(EARTH_MOON_MU)
propagator = Propagator(model)
corrector = DifferentialCorrector(propagator)

# reference = corrector.solve(NRHO_GUESS)

# print("Corrected initial state:")
# print(reference.initial_state)
# print("Approximate period:", reference.period)

state0 = np.array([
    1.0277926091,   # x
    0.0,            # y
   -0.1858044184,   # z
    0.0,            # vx
   -0.1154896637,   # vy
    0.0             # vz
])

result = propagator.propagate(state0, 10, n_eval=10000)

fig = plot_trajectory_3d(result.states)
show_figure(fig)

