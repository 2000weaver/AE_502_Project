import numpy as np
from .dynamical_model import DynamicalModel

class CR3BP(DynamicalModel):
    def __init__(self, mu: float):
        """
        Initialize the CR3BP model with the given mass parameter.

        Parameters:
        mu (float): The mass parameter of the system, defined as the ratio of the smaller primary's mass to the total mass.
        """
        self.mu = mu
        self.mu1 = 1.0 - mu
        self.mu2 = mu

    def equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
        x, y, z, vx, vy, vz = state

        r1 = np.sqrt((x + self.mu2)**2 + y**2 + z**2)
        r2 = np.sqrt((x - self.mu1)**2 + y**2 + z**2)

        ax = (
            2*vy 
            + x 
            - self.mu1 * (x + self.mu2) / r1**3 
            - self.mu2 * (x - self.mu1) / r2**3
        )

        ay = (
            -2*vx 
            + y 
            - self.mu1 * y / r1**3 
            - self.mu2 * y / r2**3
        )

        az = (
            - self.mu1 * z / r1**3 
            - self.mu2 * z / r2**3
        )

        return np.array([vx, vy, vz, ax, ay, az])
    
    def jacobi_constant(self, state):
        x, y, z, vx, vy, vz = state

        r1 = np.sqrt((x + self.mu)**2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + self.mu)**2 + y**2 + z**2)

        C = (
            x**2 + y**2
            + 2*((1 - self.mu)/r1 + self.mu/r2)
            - (vx**2 + vy**2 + vz**2)
        )

        return C