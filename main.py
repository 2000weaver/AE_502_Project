
from astrokit.models.cr3bp import CR3BP
from astrokit.simulation.propagator import Propagator
from astrokit.utils.constants import EARTH_MOON_MU
from astrokit.utils.plotting import plot_trajectory_3d, plot_jacobi

# Create model
model = CR3BP(EARTH_MOON_MU)

# Create propagator
propagator = Propagator(model)

# Test state near an L2 halo orbit
state0 = [
    1.02,
    0.0,
    0.05,
    0.0,
    -0.10,
    0.0
]

# Propagate
result = propagator.propagate(state0, tf=1.0)

# Plot
plot_trajectory_3d(result.states)
plot_jacobi(result.t, result.jacobi)