from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d

from ..orbit_design.reference_trajectory import ReferenceTrajectory

__all__ = [
    "StationKeepingProblem",
    "CorrectionRecord",
    "StationKeepingHistory",
    "build_reference_sampler",
    "reference_state_at_time",
    "reference_error",
]


@dataclass
class StationKeepingProblem:
    """
    Minimal container for a station-keeping study.

    This keeps the project structure in place without committing you to a
    particular guidance or targeting law.
    """

    reference: ReferenceTrajectory
    correction_interval: float
    duration: float
    metadata: dict = field(default_factory=dict)


@dataclass
class CorrectionRecord:
    time: float
    pre_maneuver_state: np.ndarray
    post_maneuver_state: np.ndarray
    delta_v: np.ndarray
    reference_state: np.ndarray


@dataclass
class StationKeepingHistory:
    """
    Lightweight output structure for your future STM-based controller.
    """

    times: list[float] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=list)
    corrections: list[CorrectionRecord] = field(default_factory=list)


def build_reference_sampler(reference: ReferenceTrajectory):
    """
    Return a periodic interpolator Phi_ref(t mod T) for the reference orbit.

    The reference trajectory produced by the differential corrector is usually a
    single half- or full-period arc sampled at discrete times. This helper makes
    it easy to query the reference state at arbitrary phase times during your
    station-keeping loop.
    """
    phase_times = reference.t
    states = reference.states

    if phase_times[0] != 0.0:
        raise ValueError("Reference trajectory must start at t = 0.")

    if phase_times[-1] < reference.period:
        phase_times = np.append(phase_times, reference.period)
        states = np.column_stack([states, reference.initial_state])

    return interp1d(
        phase_times,
        states,
        kind="cubic",
        axis=1,
        fill_value="extrapolate",
    )


def reference_state_at_time(reference: ReferenceTrajectory, sampler, time: float) -> np.ndarray:
    phase = np.mod(time, reference.period)
    return np.asarray(sampler(phase), dtype=float)


def reference_error(current_state: np.ndarray, reference_state: np.ndarray) -> np.ndarray:
    """
    Return the 6D state error x - x_ref in canonical CR3BP coordinates.

    This is the quantity you will likely feed into your STM-based targeting or
    local linear feedback design.
    """
    return np.asarray(current_state, dtype=float) - np.asarray(reference_state, dtype=float)
