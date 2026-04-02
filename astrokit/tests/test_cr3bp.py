import numpy as np
from astrokit.models import CR3BP

MU = 0.0121505856


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