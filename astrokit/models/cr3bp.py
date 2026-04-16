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

    #---------------------------------------------------------------------------
    # Core Dynamics
    #---------------------------------------------------------------------------

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
    
    #---------------------------------------------------------------------------
    # Jacobian of the EOM
    #---------------------------------------------------------------------------

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian matrix. A = df/dx of the CR3BP equations of motion.
        """ 
        x, y, z = state[0], state[1], state[2]

        r1 = np.sqrt((x + self.mu2)**2 + y**2 + z**2)
        r2 = np.sqrt((x - self.mu1)**2 + y**2 + z**2)

        r1_3 = r1**3
        r1_5 = r1**5
        r2_3 = r2**3
        r2_5 = r2**5

        # Position differences
        x1 = x + self.mu2
        x2 = x - self.mu1

        # Second Partial Deterivatives
        Uxx = (
            1
            - self.mu1 / r1_3
            - self.mu2 / r2_3
            + 3 * self.mu1 * x1**2 / r1_5
            + 3 * self.mu2 * x2**2 / r2_5
        )

        Uxx = (
            1
            - self.mu1 / r1_3
            - self.mu2 / r2_3
            + 3 * self.mu1 * x1**2 / r1_5
            + 3 * self.mu2 * x2**2 / r2_5
        )
        Uyy = (
            1
            - self.mu1 / r1_3
            - self.mu2 / r2_3
            + 3 * self.mu1 * y**2 / r1_5
            + 3 * self.mu2 * y**2 / r2_5
        )
        Uzz = (
            -self.mu1 / r1_3
            - self.mu2 / r2_3
            + 3 * self.mu1 * z**2 / r1_5
            + 3 * self.mu2 * z**2 / r2_5
        )
        Uxy = (
            3 * self.mu1 * x1 * y / r1_5
            + 3 * self.mu2 * x2 * y / r2_5
        )
        Uxz = (
            3 * self.mu1 * x1 * z / r1_5
            + 3 * self.mu2 * x2 * z / r2_5
        )
        Uyz = (
            3 * self.mu1 * y * z / r1_5
            + 3 * self.mu2 * y * z / r2_5
        )
 
        # Build 6x6 Jacobian
        #        x    y    z   vx   vy   vz
        A = np.array([
            [  0,    0,    0,   1,   0,   0],   # dx/dt  = vx
            [  0,    0,    0,   0,   1,   0],   # dy/dt  = vy
            [  0,    0,    0,   0,   0,   1],   # dz/dt  = vz
            [Uxx, Uxy, Uxz,    0,   2,   0],   # dvx/dt
            [Uxy, Uyy, Uyz,   -2,   0,   0],   # dvy/dt
            [Uxz, Uyz, Uzz,    0,   0,   0],   # dvz/dt
        ], dtype=float)
 
        return A
 
    # ------------------------------------------------------------------
    # Variational equations for STM propagation
    # ------------------------------------------------------------------
 
    def equations_of_motion_stm(self, t: float, state_stm: np.ndarray) -> np.ndarray:
        """
        Augmented EOM integrating the state and STM simultaneously.
 
        The input vector is 42-dimensional:
            state_stm[:6]   — the 6-element state  [x, y, z, vx, vy, vz]
            state_stm[6:]   — the 36-element STM Φ flattened row-major
 
        Returns the 42-element derivative vector.
 
        Usage with Propagator:
            propagator.propagate_stm(state0, tf)
        """
        state = state_stm[:6]
        Phi   = state_stm[6:].reshape(6, 6)
 
        # State derivative
        dstate = self.equations_of_motion(t, state)
 
        # STM derivative:  dΦ/dt = A(t) Φ
        A    = self.jacobian(state)
        dPhi = A @ Phi
 
        return np.concatenate([dstate, dPhi.ravel()])