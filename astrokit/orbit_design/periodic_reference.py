from __future__ import annotations

import numpy as np

from .reference_trajectory import ReferenceTrajectory

__all__ = [
    "build_full_period_reference",
    "loop_reference_trajectory",
]


def build_full_period_reference(
    propagator,
    corrected_reference: ReferenceTrajectory,
    samples_per_period: int = 2000,
) -> ReferenceTrajectory:
    """
    Propagate a corrected periodic-orbit seed over one full period.

    This is useful when the differential corrector stores only a partial arc but
    you want one clean, full-period reference trajectory that can be reused
    without long-horizon instability from repeated numerical propagation.
    """
    if corrected_reference.period <= 0.0:
        raise ValueError("corrected_reference.period must be positive")
    if samples_per_period < 2:
        raise ValueError("samples_per_period must be at least 2")

    result = propagator.propagate(
        corrected_reference.initial_state,
        tf=corrected_reference.period,
        n_eval=samples_per_period,
    )

    t = np.asarray(result.t, dtype=float)
    states = np.asarray(result.states, dtype=float).copy()

    # Force the stored one-period reference to close exactly at the initial
    # state so tiled references do not show a visual jump at each period
    # boundary due to small period/crossing interpolation error.
    if t.size > 0:
        states[:, -1] = np.asarray(corrected_reference.initial_state, dtype=float)

    metadata = dict(corrected_reference.metadata)
    metadata["reference_source"] = "full_period_propagation"
    metadata["closure_enforced"] = True

    return ReferenceTrajectory(
        initial_state=np.asarray(corrected_reference.initial_state, dtype=float).copy(),
        period=float(corrected_reference.period),
        t=t,
        states=states,
        family_type=corrected_reference.family_type,
        jacobi_constant=float(result.jacobi[0]) if result.jacobi is not None else corrected_reference.jacobi_constant,
        metadata=metadata,
    )


def loop_reference_trajectory(
    full_period_reference: ReferenceTrajectory,
    num_periods: int,
) -> ReferenceTrajectory:
    """
    Repeat a full-period reference trajectory for as many periods as desired.

    The returned trajectory is assembled by tiling the already-closed reference
    period, so it avoids the instability growth you would see from propagating
    the nonlinear dynamics over a long time horizon.
    """
    if num_periods < 1:
        raise ValueError("num_periods must be at least 1")
    if full_period_reference.period <= 0.0:
        raise ValueError("full_period_reference.period must be positive")

    base_t = np.asarray(full_period_reference.t, dtype=float)
    base_states = np.asarray(full_period_reference.states, dtype=float)

    if base_t.ndim != 1:
        raise ValueError("full_period_reference.t must be a 1D array")
    if base_states.ndim != 2 or base_states.shape[1] != base_t.size:
        raise ValueError("full_period_reference.states must align with full_period_reference.t")
    if base_t[0] != 0.0:
        raise ValueError("full_period_reference must start at t = 0")
    if base_t[-1] + 1e-10 < full_period_reference.period:
        raise ValueError(
            "full_period_reference must span a full period. "
            "Build it with build_full_period_reference() first."
        )

    include_endpoint = np.isclose(base_t[-1], full_period_reference.period)
    segment_t = base_t[:-1] if include_endpoint else base_t
    segment_states = base_states[:, :-1] if include_endpoint else base_states

    tiled_times = []
    tiled_states = []
    for period_index in range(num_periods):
        tiled_times.append(segment_t + period_index * full_period_reference.period)
        tiled_states.append(segment_states)

    final_time = np.array([num_periods * full_period_reference.period], dtype=float)
    final_state = np.asarray(full_period_reference.initial_state, dtype=float).reshape(-1, 1)

    t = np.concatenate(tiled_times + [final_time])
    states = np.hstack(tiled_states + [final_state])

    metadata = dict(full_period_reference.metadata)
    metadata["looped_periods"] = num_periods

    return ReferenceTrajectory(
        initial_state=np.asarray(full_period_reference.initial_state, dtype=float).copy(),
        period=float(full_period_reference.period),
        t=t,
        states=states,
        family_type=full_period_reference.family_type,
        jacobi_constant=full_period_reference.jacobi_constant,
        metadata=metadata,
    )
