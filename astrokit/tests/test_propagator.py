import numpy as np
from astrokit.models import CR3BP
from astrokit.models.perturbing_body import create_solar_perturbation
from astrokit.simulation import Propagator
from astrokit.utils.constants import EARTH_MOON_MU

MU = EARTH_MOON_MU


def test_propagation_runs(cr3bp_model, initial_state):
    """Test that propagation completes without errors."""
    propagator = Propagator(cr3bp_model)
    result = propagator.propagate(initial_state, tf=1.0)

    assert len(result.t) > 10
    assert result.states.shape[0] == 6
    assert result.states.shape[1] == len(result.t)

def test_jacobi_constant_conserved(cr3bp_model, initial_state):
    """Test that Jacobi constant remains conserved during propagation."""
    propagator = Propagator(cr3bp_model)
    result = propagator.propagate(initial_state, tf=5.0)

    jacobi_variation = np.max(result.jacobi) - np.min(result.jacobi)
    
    # Jacobi constant should vary minimally (only due to numerical errors)
    assert jacobi_variation < 1e-7

def test_equations_of_motion_known_value(cr3bp_model):
    """Test equations of motion with a known initial state."""
    state = np.array([
        1.0113254829162490E+0,
        -3.4655306799190984E-28,
        1.7343215557041181E-1,
        -6.8885128080822615E-13,
        -7.8717210400801721E-2,
        1.0116298133884205E-11
    ])
    deriv = cr3bp_model.equations_of_motion(0.0, state)

    # First three components should be velocities
    expected_velocity = np.array([
        -6.8885128080822615E-13,
        -7.8717210400801721E-2,
        1.0116298133884205E-11
    ])
    np.testing.assert_allclose(deriv[:3], expected_velocity, atol=1e-12)


def test_differential_corrector_period(cr3bp_model):
    from astrokit.orbit_design.differential_corrector import DifferentialCorrector
    from astrokit.orbit_design.initial_guesses import EARTH_MOON_L2_NORTHERN_HALO_GUESS

    propagator = Propagator(cr3bp_model)
    corrector = DifferentialCorrector(propagator)

    reference = corrector.solve(EARTH_MOON_L2_NORTHERN_HALO_GUESS)

    assert reference.period > 0
    assert reference.period < 10

    propagated = propagator.propagate(reference.initial_state, tf=reference.period, n_eval=1000)
    assert abs(propagated.states[1, -1]) < 1e-2


def test_perturbed_propagation_runs_and_returns_finite_states(initial_state):
    model = CR3BP(mu=MU, perturbing_body=create_solar_perturbation())
    propagator = Propagator(model)

    result = propagator.propagate(initial_state, tf=1.0, n_eval=500)

    assert result.states.shape == (6, len(result.t))
    assert np.all(np.isfinite(result.states))
    assert np.all(np.isfinite(result.jacobi))


def test_perturbed_and_unperturbed_models_diverge_from_same_initial_state(initial_state):
    unperturbed = Propagator(CR3BP(mu=MU))
    perturbed = Propagator(CR3BP(mu=MU, perturbing_body=create_solar_perturbation()))

    result_unperturbed = unperturbed.propagate(initial_state, tf=2.0, n_eval=1000)
    result_perturbed = perturbed.propagate(initial_state, tf=2.0, n_eval=1000)

    final_position_delta = np.linalg.norm(
        result_perturbed.states[:3, -1] - result_unperturbed.states[:3, -1]
    )
    assert final_position_delta > 1e-8
