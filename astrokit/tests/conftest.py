import pytest
import numpy as np
from astrokit.models import CR3BP
from astrokit.utils.constants import EARTH_MOON_MU

@pytest.fixture
def cr3bp_model():
    """Fixture providing a CR3BP model instance with Earth-Moon parameters."""
    return CR3BP(EARTH_MOON_MU)

@pytest.fixture
def initial_state():
    """Fixture providing a realistic initial state for L2 Halo orbit."""
    return np.array([
        1.0113254829162490E+0,
        -3.4655306799190984E-28,
        1.7343215557041181E-1,
        -6.8885128080822615E-13,
        -7.8717210400801721E-2,
        1.0116298133884205E-11
    ])
