"""
Perturbing body models for the Earth-Moon rotating CR3BP frame.

For the Sun, this module now follows the paper-style approach more directly:
the Sun position is prescribed in the Earth-Moon rotating frame and the force
added to the CR3BP is the differential solar acceleration.
"""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from ..utils.constants import (
    SUN_DISTANCE_IN_EM,
    SUN_INITIAL_PHASE,
    SUN_MASS_IN_EM,
    SUN_RELATIVE_ANGULAR_RATE_IN_EM,
)

__all__ = [
    "PerturbingBody",
    "CircularOrbitBody",
    "VariableDistanceBody",
    "SunPerturbation",
    "create_solar_perturbation",
]


class PerturbingBody(ABC):
    """Abstract perturbing body expressed in Earth-Moon canonical units."""

    def __init__(self, mass_parameter: float):
        self.mu_body = mass_parameter

    @abstractmethod
    def get_position(self, t: float) -> np.ndarray:
        pass

    @abstractmethod
    def get_distance_to_barycenter(self, t: float) -> float:
        pass

    def get_mass_parameter(self) -> float:
        return self.mu_body


class CircularOrbitBody(PerturbingBody):
    def __init__(
        self,
        mass_parameter: float,
        mean_distance: float,
        angular_velocity: float,
        initial_angle: float = 0.0,
    ):
        super().__init__(mass_parameter)
        self.d_mean = mean_distance
        self.omega = angular_velocity
        self.theta_0 = initial_angle

    def get_position(self, t: float) -> np.ndarray:
        theta = self.theta_0 + self.omega * t
        return np.array(
            [
                self.d_mean * np.cos(theta),
                self.d_mean * np.sin(theta),
                0.0,
            ],
            dtype=float,
        )

    def get_distance_to_barycenter(self, t: float) -> float:
        return self.d_mean


class VariableDistanceBody(PerturbingBody):
    def __init__(
        self,
        mass_parameter: float,
        distance_func: Callable[[float], float],
        angular_velocity: float,
        initial_angle: float = 0.0,
    ):
        super().__init__(mass_parameter)
        self.distance_func = distance_func
        self.omega = angular_velocity
        self.theta_0 = initial_angle

    def get_position(self, t: float) -> np.ndarray:
        d_t = self.distance_func(t)
        theta = self.theta_0 + self.omega * t
        return np.array(
            [d_t * np.cos(theta), d_t * np.sin(theta), 0.0],
            dtype=float,
        )

    def get_distance_to_barycenter(self, t: float) -> float:
        return self.distance_func(t)


class SunPerturbation(PerturbingBody):
    """
    Prescribed Sun motion in the Earth-Moon rotating frame.

    This follows the paper's rotating-frame prescription:
        R_s(t) = R_s [cos(theta_2), sin(theta_2), 0]
        theta_2(t) = gamma + omega_rel t

    with omega_rel expressed in Earth-Moon canonical units.
    """

    def __init__(
        self,
        mass_parameter: float,
        mean_distance: float,
        relative_angular_rate: float,
        initial_phase: float = SUN_INITIAL_PHASE,
    ):
        super().__init__(mass_parameter)
        self.d_mean = mean_distance
        self.omega_rel = relative_angular_rate
        self.gamma_0 = initial_phase

    def get_position(self, t: float) -> np.ndarray:
        theta = self.gamma_0 + self.omega_rel * t
        return np.array(
            [
                self.d_mean * np.cos(theta),
                self.d_mean * np.sin(theta),
                0.0,
            ],
            dtype=float,
        )

    def get_distance_to_barycenter(self, t: float) -> float:
        return self.d_mean


def create_solar_perturbation(
    sun_mass_em: float = SUN_MASS_IN_EM,
    sun_distance_em: float = SUN_DISTANCE_IN_EM,
    sun_relative_angular_rate_em: float = SUN_RELATIVE_ANGULAR_RATE_IN_EM,
    sun_initial_phase: float = SUN_INITIAL_PHASE,
) -> SunPerturbation:
    """
    Create the paper-style solar perturbation in the Earth-Moon rotating frame.
    """
    return SunPerturbation(
        mass_parameter=sun_mass_em,
        mean_distance=sun_distance_em,
        relative_angular_rate=sun_relative_angular_rate_em,
        initial_phase=sun_initial_phase,
    )
