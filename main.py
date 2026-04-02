
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
    1.0113254829162490E+0,
    -3.4655306799190984E-28,
    1.7343215557041181E-1,
    -6.8885128080822615E-13,
    -7.8717210400801721E-2,
    1.0116298133884205E-11
]

state0 = [
    1 - EARTH_MOON_MU,
    0.0455,
    0,
    -0.5,
    0.5,
    0.0001
]

# Propagate
result = propagator.propagate(state0, tf=2*10.0)

# Plot
plot_trajectory_3d(result.states, EARTH_MOON_MU)
# plot_jacobi(result.t, result.jacobi)