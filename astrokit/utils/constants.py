"""Physical and canonical constants for the Earth-Moon CR3BP/BCP models.

The dynamical models in :mod:`astrokit.models` are dimensionless. Positions are
scaled by the Earth-Moon distance and time is scaled by the inverse Earth-Moon
mean motion, so the rotating frame has unit angular rate.

The SI values below come from Table 1 of:
de Almeida Junior & de Almeida Prado, Sci Rep 12, 4148 (2022).
"""

import numpy as np

# Sun-Earth-Moon parameters taken from Table 1 of
# de Almeida Junior & de Almeida Prado, Sci Rep 12, 4148 (2022).
# These values are the closest match to the paper's perturbation plots.
SUN_GRAVITATIONAL_PARAMETER = 1.3237395128595653e20  # m^3 / s^2
EARTH_GRAVITATIONAL_PARAMETER = 3.975837768911438e14  # m^3 / s^2
MOON_GRAVITATIONAL_PARAMETER = 4.890329364450684e12  # m^3 / s^2

# Useful distances from the paper
EARTH_MOON_DISTANCE_M = 3.84405000e8  # m
EARTH_SUN_DISTANCE_M = 1.49460947424915e11  # m
EARTH_MOON_DISTANCE = EARTH_MOON_DISTANCE_M / 1_000.0  # km
EARTH_SUN_DISTANCE = EARTH_SUN_DISTANCE_M / 1_000.0  # km

# CR3BP mu constants
EARTH_MOON_MU = MOON_GRAVITATIONAL_PARAMETER / (
    EARTH_GRAVITATIONAL_PARAMETER + MOON_GRAVITATIONAL_PARAMETER
)
SUN_EARTH_MU = SUN_GRAVITATIONAL_PARAMETER / (
    SUN_GRAVITATIONAL_PARAMETER + EARTH_GRAVITATIONAL_PARAMETER
)

# Earth-Moon canonical time unit and frame rate from the paper values.
EARTH_MOON_MEAN_MOTION_SI = np.sqrt(
    (EARTH_GRAVITATIONAL_PARAMETER + MOON_GRAVITATIONAL_PARAMETER)
    / EARTH_MOON_DISTANCE_M**3
)
EARTH_MOON_TIME_UNIT_SECONDS = 1.0 / EARTH_MOON_MEAN_MOTION_SI
EARTH_MOON_TIME_UNIT_DAYS = EARTH_MOON_TIME_UNIT_SECONDS / 86400.0
EARTH_MOON_LENGTH_UNIT_M = EARTH_MOON_DISTANCE_M
EARTH_MOON_VELOCITY_UNIT_MPS = (
    EARTH_MOON_LENGTH_UNIT_M / EARTH_MOON_TIME_UNIT_SECONDS
)
EARTH_MOON_ACCELERATION_UNIT_MPS2 = (
    EARTH_MOON_LENGTH_UNIT_M / EARTH_MOON_TIME_UNIT_SECONDS**2
)

# In standard CR3BP canonical units, the Earth-Moon rotating frame spins at 1.
EARTH_MOON_FRAME_ANGULAR_RATE = 1.0

# Perturbation parameters for the Sun in Earth-Moon canonical units
SUN_MASS_IN_EM = SUN_GRAVITATIONAL_PARAMETER / (
    EARTH_GRAVITATIONAL_PARAMETER + MOON_GRAVITATIONAL_PARAMETER
)
SUN_DISTANCE_IN_EM = EARTH_SUN_DISTANCE_M / EARTH_MOON_DISTANCE_M
SUN_INERTIAL_ANGULAR_RATE_SI = 2.0 * np.pi / (365.25 * 86400.0)
SUN_INERTIAL_ANGULAR_RATE_IN_EM = (
    SUN_INERTIAL_ANGULAR_RATE_SI / EARTH_MOON_MEAN_MOTION_SI
)
SUN_ORBITAL_PERIOD_IN_EM = 2.0 * np.pi / SUN_INERTIAL_ANGULAR_RATE_IN_EM

# Paper-style relative angular rate in the Earth-Moon rotating frame:
# theta_s(t) = gamma + (omega_0 - omega)t
SUN_RELATIVE_ANGULAR_RATE_IN_EM = (
    SUN_INERTIAL_ANGULAR_RATE_IN_EM - EARTH_MOON_FRAME_ANGULAR_RATE
)

# For the perturbation force, the paper writes the Sun position in the rotating
# frame as R_s(cos(theta_2), sin(theta_2), 0) with theta_2 = omega_s t + gamma.
# Keep gamma configurable and default it to the paper's theta_2 = 0 baseline.
SUN_INITIAL_PHASE = 0.0
