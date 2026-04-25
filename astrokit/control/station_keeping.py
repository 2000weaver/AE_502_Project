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
    "stm_based_position_correction",
    "perform_correction_step",
    "run_station_keeping",
    "sweep_correction_intervals",
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
    max_delta_v_fraction: float | None = None
    max_phi_rv_condition_number: float = 1.0e8
    metadata: dict = field(default_factory=dict)


@dataclass
class CorrectionRecord:
    time: float
    pre_maneuver_state: np.ndarray
    post_maneuver_state: np.ndarray
    delta_v: np.ndarray
    reference_state: np.ndarray
    phi_rv_condition_number: float | None = None
    used_pseudoinverse: bool = False


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


def stm_based_position_correction(
    stm: np.ndarray,
    delta_r0: np.ndarray,
    delta_v0: np.ndarray | None = None,
    max_delta_v_fraction: float | None = None,
    current_velocity: np.ndarray | None = None,
    max_phi_rv_condition_number: float = 1.0e8,
) -> tuple[np.ndarray, float, bool]:
    """
    Compute the velocity correction that cancels final position error using an STM.

    With the STM partitioned as:

        [delta_r(tf)]   [Phi_rr  Phi_rv] [delta_r0]
        [delta_v(tf)] = [Phi_vr  Phi_vv] [delta_v ]

    enforcing delta_r(tf) = 0 gives:

        delta_v = -Phi_rv^{-1} (Phi_rr delta_r0 + Phi_rv delta_v0)

    Parameters
    ----------
    stm : np.ndarray
        Full 6x6 state transition matrix.
    delta_r0 : np.ndarray
        Initial position error vector with shape (3,).
    delta_v0 : np.ndarray, optional
        Initial velocity error vector with shape (3,). If omitted, a zero
        velocity error is assumed.
    max_delta_v_fraction : float, optional
        If provided, cap the correction magnitude to this fraction of the
        current speed.
    current_velocity : np.ndarray, optional
        Current velocity vector with shape (3,). Required when
        ``max_delta_v_fraction`` is provided.

    Returns
    -------
    tuple[np.ndarray, float, bool]
        Velocity correction, condition number of Phi_rv, and whether a
        pseudo-inverse fallback was used.
    """
    stm = np.asarray(stm, dtype=float)
    delta_r0 = np.asarray(delta_r0, dtype=float)
    delta_v0 = np.zeros(3, dtype=float) if delta_v0 is None else np.asarray(delta_v0, dtype=float)

    if stm.shape != (6, 6):
        raise ValueError("stm must have shape (6, 6)")
    if delta_r0.shape != (3,):
        raise ValueError("delta_r0 must have shape (3,)")
    if delta_v0.shape != (3,):
        raise ValueError("delta_v0 must have shape (3,)")

    phi_rr = stm[:3, :3]
    phi_rv = stm[:3, 3:6]

    rhs = phi_rr @ delta_r0 + phi_rv @ delta_v0
    phi_rv_condition_number = float(np.linalg.cond(phi_rv))
    used_pseudoinverse = (
        np.linalg.matrix_rank(phi_rv) < 3
        or not np.isfinite(phi_rv_condition_number)
        or phi_rv_condition_number > max_phi_rv_condition_number
    )

    if used_pseudoinverse:
        delta_v = -np.linalg.pinv(phi_rv) @ rhs
    else:
        delta_v = -np.linalg.solve(phi_rv, rhs)

    if max_delta_v_fraction is not None:
        if max_delta_v_fraction < 0.0:
            raise ValueError("max_delta_v_fraction must be nonnegative")
        if current_velocity is None:
            raise ValueError("current_velocity must be provided when max_delta_v_fraction is set")
        current_velocity = np.asarray(current_velocity, dtype=float)
        if current_velocity.shape != (3,):
            raise ValueError("current_velocity must have shape (3,)")

        max_delta_v = max_delta_v_fraction * np.linalg.norm(current_velocity)
        delta_v_norm = np.linalg.norm(delta_v)
        if delta_v_norm > max_delta_v and delta_v_norm > 0.0:
            delta_v = delta_v * (max_delta_v / delta_v_norm)

    return delta_v, phi_rv_condition_number, used_pseudoinverse


def _interpolate_state(result, target_time: float) -> np.ndarray:
    interpolated = np.array(
        [np.interp(target_time, result.t, result.states[row]) for row in range(result.states.shape[0])],
        dtype=float,
    )
    return interpolated


def _append_segment(history: StationKeepingHistory, times: np.ndarray, states: np.ndarray) -> None:
    for idx, time in enumerate(np.asarray(times, dtype=float)):
        state = np.asarray(states[:, idx], dtype=float)
        if history.times and np.isclose(time, history.times[-1]):
            history.states[-1] = state
        else:
            history.times.append(float(time))
            history.states.append(state)


def perform_correction_step(
    propagator,
    reference: ReferenceTrajectory,
    sampler,
    current_time: float,
    current_state: np.ndarray,
    correction_interval: float,
    control_offset: float | None = None,
    max_delta_v_fraction: float | None = None,
    max_phi_rv_condition_number: float = 1.0e8,
):
    """
    Execute one STM-based station-keeping step.

    The flow is:
    1. Propagate from the current correction point to a control point.
    2. Compute the position error relative to the periodic reference there.
    3. Use the STM from control point to next correction point to compute a dv.
    4. Apply the dv at the control point.
    5. Propagate to the next correction point.

    Parameters
    ----------
    propagator : Propagator
        Propagator for the model you want to control.
    reference : ReferenceTrajectory
        Periodic reference trajectory.
    sampler :
        Interpolator returned by build_reference_sampler(reference).
    current_time : float
        Current correction time.
    current_state : np.ndarray
        Current 6D state at current_time.
    correction_interval : float
        Time between correction points.
    control_offset : float, optional
        Offset from the current correction point to the control point.
        Defaults to half the correction interval.
    max_delta_v_fraction : float, optional
        If provided, cap the correction magnitude to this fraction of the
        current speed at the control point.
    max_phi_rv_condition_number : float, optional
        Threshold above which the solve switches from a direct solve to a
        pseudo-inverse for robustness.

    Returns
    -------
    dict
        Contains propagation segments, updated state/time, and the correction record.
    """
    current_state = np.asarray(current_state, dtype=float)
    if current_state.shape != (6,):
        raise ValueError("current_state must have shape (6,)")
    if correction_interval <= 0.0:
        raise ValueError("correction_interval must be positive")

    if control_offset is None:
        control_offset = 0.5 * correction_interval
    if not (0.0 < control_offset <= correction_interval):
        raise ValueError("control_offset must satisfy 0 < control_offset <= correction_interval")

    control_time = current_time + control_offset
    next_time = current_time + correction_interval

    to_control = propagator.propagate(
        current_state,
        tf=control_offset,
        n_eval=max(50, int(500 * control_offset / correction_interval)),
    )
    control_state = np.asarray(to_control.states[:, -1], dtype=float)
    reference_state = reference_state_at_time(reference, sampler, control_time)
    delta_x0 = reference_error(control_state, reference_state)
    delta_r0 = delta_x0[:3]
    delta_v0 = delta_x0[3:6]

    horizon = next_time - control_time
    reference_to_next_stm = propagator.propagate_stm(
        reference_state,
        tf=horizon,
        n_eval=max(50, int(500 * horizon / correction_interval)),
    )
    delta_v, phi_rv_condition_number, used_pseudoinverse = stm_based_position_correction(
        reference_to_next_stm.monodromy,
        delta_r0,
        delta_v0=delta_v0,
        max_delta_v_fraction=max_delta_v_fraction,
        current_velocity=control_state[3:6],
        max_phi_rv_condition_number=max_phi_rv_condition_number,
    )

    post_maneuver_state = control_state.copy()
    post_maneuver_state[3:6] += delta_v

    corrected_to_next = propagator.propagate(
        post_maneuver_state,
        tf=horizon,
        n_eval=max(50, int(500 * horizon / correction_interval)),
    )

    correction_record = CorrectionRecord(
        time=float(control_time),
        pre_maneuver_state=control_state,
        post_maneuver_state=post_maneuver_state,
        delta_v=delta_v,
        reference_state=reference_state,
        phi_rv_condition_number=phi_rv_condition_number,
        used_pseudoinverse=used_pseudoinverse,
    )

    return {
        "to_control": to_control,
        "to_next": corrected_to_next,
        "control_time": float(control_time),
        "next_time": float(next_time),
        "control_state": control_state,
        "post_maneuver_state": post_maneuver_state,
        "next_state": np.asarray(corrected_to_next.states[:, -1], dtype=float),
        "correction": correction_record,
    }


def run_station_keeping(
    problem: StationKeepingProblem,
    propagator,
    initial_state: np.ndarray | None = None,
    control_offset: float | None = None,
) -> StationKeepingHistory:
    """
    Run a simple STM-based station-keeping campaign.

    Parameters
    ----------
    problem : StationKeepingProblem
        Defines the reference orbit, correction cadence, and total duration.
    propagator : Propagator
        Propagator for the controlled dynamics model.
    initial_state : np.ndarray, optional
        Initial state to start from. Defaults to the reference initial state.
    control_offset : float, optional
        Time from each correction point to the control point. Defaults to half
        the correction interval.

    Returns
    -------
    StationKeepingHistory
        Time/state history plus correction records.
    """
    if problem.correction_interval <= 0.0:
        raise ValueError("problem.correction_interval must be positive")
    if problem.duration <= 0.0:
        raise ValueError("problem.duration must be positive")

    sampler = build_reference_sampler(problem.reference)
    history = StationKeepingHistory()

    current_time = 0.0
    current_state = (
        np.asarray(problem.reference.initial_state, dtype=float).copy()
        if initial_state is None
        else np.asarray(initial_state, dtype=float).copy()
    )
    if current_state.shape != (6,):
        raise ValueError("initial_state must have shape (6,)")

    history.times.append(current_time)
    history.states.append(current_state.copy())

    while current_time + problem.correction_interval <= problem.duration + 1e-12:
        step = perform_correction_step(
            propagator=propagator,
            reference=problem.reference,
            sampler=sampler,
            current_time=current_time,
            current_state=current_state,
            correction_interval=problem.correction_interval,
            control_offset=control_offset,
            max_delta_v_fraction=problem.max_delta_v_fraction,
            max_phi_rv_condition_number=problem.max_phi_rv_condition_number,
        )

        _append_segment(
            history,
            current_time + np.asarray(step["to_control"].t, dtype=float),
            step["to_control"].states,
        )
        _append_segment(
            history,
            step["control_time"] + np.asarray(step["to_next"].t, dtype=float),
            step["to_next"].states,
        )
        history.corrections.append(step["correction"])

        current_time = step["next_time"]
        current_state = step["next_state"]

    return history


def sweep_correction_intervals(
    reference: ReferenceTrajectory,
    propagator,
    initial_state: np.ndarray,
    correction_intervals: np.ndarray,
    control_offset_fraction: float = 0.5,
    max_delta_v_fraction: float | None = None,
    duration: float | None = None,
    max_phi_rv_condition_number: float = 1.0e8,
) -> list[dict]:
    """
    Run a station-keeping trade space over multiple correction intervals.
    """
    correction_intervals = np.asarray(correction_intervals, dtype=float)
    results = []

    for interval in correction_intervals:
        problem = StationKeepingProblem(
            reference=reference,
            correction_interval=float(interval),
            duration=float(reference.t[-1] if duration is None else duration),
            max_delta_v_fraction=max_delta_v_fraction,
            max_phi_rv_condition_number=max_phi_rv_condition_number,
            metadata={"study": "correction_interval_sweep"},
        )

        history = run_station_keeping(
            problem=problem,
            propagator=propagator,
            initial_state=np.asarray(initial_state, dtype=float).copy(),
            control_offset=control_offset_fraction * float(interval),
        )

        final_state = np.asarray(history.states[-1], dtype=float)
        final_reference_state = reference_state_at_time(
            reference,
            build_reference_sampler(reference),
            history.times[-1],
        )
        final_position_error = np.linalg.norm(final_state[:3] - final_reference_state[:3])
        total_delta_v = float(np.sum([np.linalg.norm(record.delta_v) for record in history.corrections]))
        max_condition_number = float(
            max(
                [record.phi_rv_condition_number for record in history.corrections if record.phi_rv_condition_number is not None],
                default=0.0,
            )
        )

        results.append(
            {
                "correction_interval": float(interval),
                "history": history,
                "total_delta_v": total_delta_v,
                "final_position_error": final_position_error,
                "max_phi_rv_condition_number": max_condition_number,
                "used_pseudoinverse": any(record.used_pseudoinverse for record in history.corrections),
            }
        )

    return results
