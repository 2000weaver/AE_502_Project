import numpy as np

from astrokit.control.station_keeping import (
    build_reference_sampler,
    reference_error,
    reference_state_at_time,
)
from astrokit.models import CR3BP
from astrokit.orbit_design.differential_corrector import DifferentialCorrector
from astrokit.orbit_design.initial_guesses import EARTH_MOON_L2_NORTHERN_HALO_GUESS
from astrokit.simulation import Propagator
from astrokit.utils.constants import EARTH_MOON_MU


def test_reference_sampler_reproduces_initial_state():
    propagator = Propagator(CR3BP(mu=EARTH_MOON_MU))
    corrector = DifferentialCorrector(propagator)
    reference = corrector.solve(
        EARTH_MOON_L2_NORTHERN_HALO_GUESS,
        tf_propagation=8.0,
        orbit_tf=8.0,
    )

    sampler = build_reference_sampler(reference)
    sampled = reference_state_at_time(reference, sampler, 0.0)

    np.testing.assert_allclose(sampled, reference.initial_state)


def test_reference_error_is_zero_for_matching_states():
    state = np.array([1.0, 0.0, 0.1, 0.0, -0.1, 0.0])
    error = reference_error(state, state)
    np.testing.assert_allclose(error, np.zeros(6))
