from typing import Optional

import numpy as np

from .dynamical_model import DynamicalModel

__all__ = ["CR3BP"]


class CR3BP(DynamicalModel):
    """Dimensionless Earth-Moon CR3BP dynamics with optional solar perturbation.

    All states are expressed in Earth-Moon canonical units:
    - position scaled by the Earth-Moon distance
    - time scaled by the inverse Earth-Moon mean motion
    - velocity scaled by distance / time

    When ``perturbing_body`` is supplied, its mass parameter, distance, and
    angular rate must use the same canonical scaling.
    """

    def __init__(
        self,
        mu: float,
        perturbing_body: Optional["PerturbingBody"] = None,
        perturbation_scale: float = 1.0,
    ):
        self.mu = mu
        self.mu1 = 1.0 - mu
        self.mu2 = mu
        self.perturbing_body = perturbing_body
        self.perturbation_scale = perturbation_scale

    def _perturbation_acceleration(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return the solar perturbation in Earth-Moon canonical acceleration units."""
        if self.perturbing_body is None:
            return np.zeros(3)

        r_sc = state[:3]
        r_sun = self.perturbing_body.get_position(t)
        mu_sun = self.perturbing_body.get_mass_parameter()

        R_s = np.linalg.norm(r_sun)

        # --- safety checks ---
        if R_s < 1e-12:
            return np.zeros(3)

        # ------------------------------------------------------------
        # 1) Direct gravitational attraction from the Sun
        # ------------------------------------------------------------
        r_rel = r_sun - r_sc
        dist_rel = np.linalg.norm(r_rel)

        if dist_rel < 1e-12:
            a_sun = np.zeros(3)
        else:
            a_sun = mu_sun * r_rel / dist_rel**3

        # ------------------------------------------------------------
        # 2) Acceleration of the rotating frame barycenter
        #    (this is the key correction term in the paper)
        # ------------------------------------------------------------
        a_frame = mu_sun * r_sun / R_s**3

        # ------------------------------------------------------------
        # 3) Net perturbation (note SIGN from Eq. (7)/(9))
        # ------------------------------------------------------------
        return self.perturbation_scale * (a_sun - a_frame)

    def equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
        x, y, z, vx, vy, vz = state

        r1 = np.sqrt((x + self.mu2) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - self.mu1) ** 2 + y**2 + z**2)

        ax = (
            2 * vy
            + x
            - self.mu1 * (x + self.mu2) / r1**3
            - self.mu2 * (x - self.mu1) / r2**3
        )
        ay = -2 * vx + y - self.mu1 * y / r1**3 - self.mu2 * y / r2**3
        az = -self.mu1 * z / r1**3 - self.mu2 * z / r2**3

        if self.perturbing_body is not None:
            ax_p, ay_p, az_p = self._perturbation_acceleration(t, state)
            ax += ax_p
            ay += ay_p
            az += az_p

        return np.array([vx, vy, vz, ax, ay, az], dtype=float)

    def jacobi_constant(self, state: np.ndarray) -> float:
        x, y, z, vx, vy, vz = state

        r1 = np.sqrt((x + self.mu) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - 1 + self.mu) ** 2 + y**2 + z**2)

        return (
            x**2
            + y**2
            + 2 * ((1 - self.mu) / r1 + self.mu / r2)
            - (vx**2 + vy**2 + vz**2)
        )

    def _perturbation_jacobian(self, t: float, state: np.ndarray) -> np.ndarray:
        if self.perturbing_body is None:
            return np.zeros((6, 6))

        r_sc = state[:3]
        r_body = self.perturbing_body.get_position(t)
        r_rel = r_body - r_sc

        dist_rel = np.linalg.norm(r_rel)
        mu_body = self.perturbing_body.get_mass_parameter()

        if dist_rel > 1e-10:
            grad = np.eye(3) / dist_rel**3 - 3.0 * np.outer(r_rel, r_rel) / dist_rel**5
        else:
            grad = np.zeros((3, 3))

        jac = np.zeros((6, 6))
        jac[3:6, 0:3] = -self.perturbation_scale * mu_body * grad
        return jac

    def jacobian(self, state: np.ndarray, t: float = 0.0) -> np.ndarray:
        x, y, z = state[:3]

        r1 = np.sqrt((x + self.mu2) ** 2 + y**2 + z**2)
        r2 = np.sqrt((x - self.mu1) ** 2 + y**2 + z**2)

        r1_3 = r1**3
        r1_5 = r1**5
        r2_3 = r2**3
        r2_5 = r2**5

        x1 = x + self.mu2
        x2 = x - self.mu1

        uxx = (
            1
            - self.mu1 / r1_3
            - self.mu2 / r2_3
            + 3 * self.mu1 * x1**2 / r1_5
            + 3 * self.mu2 * x2**2 / r2_5
        )
        uyy = (
            1
            - self.mu1 / r1_3
            - self.mu2 / r2_3
            + 3 * self.mu1 * y**2 / r1_5
            + 3 * self.mu2 * y**2 / r2_5
        )
        uzz = (
            -self.mu1 / r1_3
            - self.mu2 / r2_3
            + 3 * self.mu1 * z**2 / r1_5
            + 3 * self.mu2 * z**2 / r2_5
        )
        uxy = 3 * self.mu1 * x1 * y / r1_5 + 3 * self.mu2 * x2 * y / r2_5
        uxz = 3 * self.mu1 * x1 * z / r1_5 + 3 * self.mu2 * x2 * z / r2_5
        uyz = 3 * self.mu1 * y * z / r1_5 + 3 * self.mu2 * y * z / r2_5

        jac = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [uxx, uxy, uxz, 0, 2, 0],
                [uxy, uyy, uyz, -2, 0, 0],
                [uxz, uyz, uzz, 0, 0, 0],
            ],
            dtype=float,
        )

        if self.perturbing_body is not None:
            jac += self._perturbation_jacobian(t, state)

        return jac

    def equations_of_motion_stm(self, t: float, state_stm: np.ndarray) -> np.ndarray:
        state = state_stm[:6]
        phi = state_stm[6:].reshape(6, 6)

        dstate = self.equations_of_motion(t, state)
        dphi = self.jacobian(state, t) @ phi
        return np.concatenate([dstate, dphi.ravel()])
