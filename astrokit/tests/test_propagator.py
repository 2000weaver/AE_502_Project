import numpy as np
from astrokit.models import CR3BP
from astrokit.simulation import Propagator

MU = 0.0121505856


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