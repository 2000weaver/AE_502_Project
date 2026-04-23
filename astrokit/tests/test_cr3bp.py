import numpy as np
from astrokit.models import CR3BP
from astrokit.models.perturbing_body import create_solar_perturbation
from astrokit.utils.constants import (
    EARTH_MOON_ACCELERATION_UNIT_MPS2,
    EARTH_MOON_LENGTH_UNIT_M,
    EARTH_MOON_MU,
    EARTH_MOON_TIME_UNIT_SECONDS,
)

MU = EARTH_MOON_MU


def test_equations_of_motion_shape(cr3bp_model):
    """Test that equations_of_motion returns correct shape."""
    state = np.array([
        1.0113254829162490E+0,
        -3.4655306799190984E-28,
        1.7343215557041181E-1,
        -6.8885128080822615E-13,
        -7.8717210400801721E-2,
        1.0116298133884205E-11
    ])
    dstatedt = cr3bp_model.equations_of_motion(0.0, state)

    assert isinstance(dstatedt, np.ndarray)
    assert dstatedt.shape == (6,)


def test_jacobi_constant_returns_scalar(cr3bp_model):
    """Test that jacobi_constant returns a scalar finite value."""
    state = np.array([
        1.0113254829162490E+0,
        -3.4655306799190984E-28,
        1.7343215557041181E-1,
        -6.8885128080822615E-13,
        -7.8717210400801721E-2,
        1.0116298133884205E-11
    ])
    C = cr3bp_model.jacobi_constant(state)

    assert np.isscalar(C) or isinstance(C, (int, float))
    assert np.isfinite(C)


def test_solar_perturbation_is_zero_at_barycenter():
    """The paper states that the solar perturbation vanishes at the barycenter."""
    model = CR3BP(MU, perturbing_body=create_solar_perturbation())
    state = np.zeros(6)

    perturbation = model._perturbation_acceleration(0.0, state)

    assert np.allclose(perturbation, np.zeros(3), atol=1e-15)


def test_canonical_acceleration_unit_is_length_over_time_squared():
    """Keep the canonical unit scaling internally consistent."""
    expected = EARTH_MOON_LENGTH_UNIT_M / EARTH_MOON_TIME_UNIT_SECONDS**2
    assert np.isclose(EARTH_MOON_ACCELERATION_UNIT_MPS2, expected)


def test_solar_perturbation_matches_model_difference():
    """The explicit perturbation term should equal the model-minus-CR3BP acceleration."""
    state = np.array([1.02, 0.01, 0.12, 0.0, -0.08, 0.0])
    unperturbed = CR3BP(MU)
    perturbed = CR3BP(MU, perturbing_body=create_solar_perturbation())

    direct_term = perturbed._perturbation_acceleration(0.0, state)
    difference_term = (
        perturbed.equations_of_motion(0.0, state)[3:6]
        - unperturbed.equations_of_motion(0.0, state)[3:6]
    )

    np.testing.assert_allclose(direct_term, difference_term, atol=1e-14, rtol=0.0)


def test_solar_perturbation_varies_with_time():
    """The Sun-perturbed model should be explicitly time dependent."""
    state = np.array([1.02, 0.01, 0.12, 0.0, -0.08, 0.0])
    model = CR3BP(MU, perturbing_body=create_solar_perturbation())

    perturbation_t0 = model._perturbation_acceleration(0.0, state)
    perturbation_t1 = model._perturbation_acceleration(1.0, state)

    assert not np.allclose(perturbation_t0, perturbation_t1)
