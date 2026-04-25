import numpy as np

from astrokit.control.station_keeping import (
    StationKeepingProblem,
    build_reference_sampler,
    perform_correction_step,
    reference_error,
    reference_state_at_time,
    run_station_keeping,
    stm_based_position_correction,
)
from astrokit.models import CR3BP
from astrokit.orbit_design.differential_corrector import DifferentialCorrector
from astrokit.orbit_design.initial_guesses import EARTH_MOON_L2_NORTHERN_HALO_GUESS
from astrokit.orbit_design.periodic_reference import (
    build_full_period_reference,
    loop_reference_trajectory,
)
from astrokit.simulation import Propagator
from astrokit.utils.constants import EARTH_MOON_MU


def test_reference_sampler_reproduces_initial_state():
    propagator = Propagator(CR3BP(mu=EARTH_MOON_MU))
    corrector = DifferentialCorrector(propagator)
    reference = corrector.solve(
        EARTH_MOON_L2_NORTHERN_HALO_GUESS,
        tf_propagation=8.0,
        orbit_tf=8.0,
    )

    sampler = build_reference_sampler(reference)
    sampled = reference_state_at_time(reference, sampler, 0.0)

    np.testing.assert_allclose(sampled, reference.initial_state)


def test_reference_error_is_zero_for_matching_states():
    state = np.array([1.0, 0.0, 0.1, 0.0, -0.1, 0.0])
    error = reference_error(state, state)
    np.testing.assert_allclose(error, np.zeros(6))


def test_build_full_period_reference_spans_one_period():
    propagator = Propagator(CR3BP(mu=EARTH_MOON_MU))
    corrector = DifferentialCorrector(propagator)
    corrected = corrector.solve(
        EARTH_MOON_L2_NORTHERN_HALO_GUESS,
        tf_propagation=8.0,
        orbit_tf=8.0,
    )

    full_reference = build_full_period_reference(propagator, corrected, samples_per_period=400)

    assert np.isclose(full_reference.t[0], 0.0)
    assert np.isclose(full_reference.t[-1], corrected.period)
    np.testing.assert_allclose(full_reference.states[:, 0], corrected.initial_state)


def test_loop_reference_trajectory_repeats_closed_period():
    propagator = Propagator(CR3BP(mu=EARTH_MOON_MU))
    corrector = DifferentialCorrector(propagator)
    corrected = corrector.solve(
        EARTH_MOON_L2_NORTHERN_HALO_GUESS,
        tf_propagation=8.0,
        orbit_tf=8.0,
    )
    full_reference = build_full_period_reference(propagator, corrected, samples_per_period=300)
    looped = loop_reference_trajectory(full_reference, num_periods=3)

    assert np.isclose(looped.t[0], 0.0)
    assert np.isclose(looped.t[-1], 3.0 * corrected.period)
    np.testing.assert_allclose(looped.states[:, 0], corrected.initial_state)
    np.testing.assert_allclose(looped.states[:, -1], corrected.initial_state, atol=1e-9)


def test_stm_based_position_correction_cancels_final_position_error():
    phi_rr = np.array(
        [
            [1.2, 0.1, -0.2],
            [0.0, 0.9, 0.3],
            [0.2, -0.1, 1.1],
        ]
    )
    phi_rv = np.array(
        [
            [2.0, 0.1, 0.0],
            [0.0, 1.8, 0.2],
            [0.1, -0.2, 1.6],
        ]
    )
    stm = np.eye(6)
    stm[:3, :3] = phi_rr
    stm[:3, 3:6] = phi_rv

    delta_r0 = np.array([0.03, -0.02, 0.01])
    delta_v = stm_based_position_correction(stm, delta_r0)

    final_position_error = phi_rr @ delta_r0 + phi_rv @ delta_v
    np.testing.assert_allclose(final_position_error, np.zeros(3), atol=1e-12)


def test_stm_based_position_correction_rejects_singular_phi_rv():
    stm = np.eye(6)
    stm[:3, 3:6] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    try:
        stm_based_position_correction(stm, np.array([0.1, 0.0, -0.1]))
    except np.linalg.LinAlgError:
        pass
    else:
        raise AssertionError("Expected LinAlgError for singular Phi_rv")


def test_stm_based_position_correction_respects_delta_v_cap():
    stm = np.eye(6)
    delta_r0 = np.array([1.0, 0.0, 0.0])
    delta_v0 = np.zeros(3)
    current_velocity = np.array([10.0, 0.0, 0.0])

    delta_v = stm_based_position_correction(
        stm,
        delta_r0,
        delta_v0=delta_v0,
        max_delta_v_fraction=0.01,
        current_velocity=current_velocity,
    )

    assert np.isclose(np.linalg.norm(delta_v), 0.1)


def test_stm_based_position_correction_uses_pseudoinverse_for_ill_conditioned_phi_rv():
    stm = np.eye(6)
    stm[:3, 3:6] = np.diag([1.0, 1.0e-12, 1.0])

    delta_v, phi_rv_condition_number, used_pseudoinverse = stm_based_position_correction(
        stm,
        np.array([0.1, 0.0, -0.1]),
        max_phi_rv_condition_number=1.0e8,
    )

    assert used_pseudoinverse
    assert phi_rv_condition_number > 1.0e8
    assert delta_v.shape == (3,)


def test_perform_correction_step_returns_expected_fields():
    propagator = Propagator(CR3BP(mu=EARTH_MOON_MU))
    corrector = DifferentialCorrector(propagator)
    corrected = corrector.solve(
        EARTH_MOON_L2_NORTHERN_HALO_GUESS,
        tf_propagation=8.0,
        orbit_tf=8.0,
    )
    reference = build_full_period_reference(propagator, corrected, samples_per_period=300)
    sampler = build_reference_sampler(reference)

    disturbed_state = corrected.initial_state.copy()
    disturbed_state[0] += 1e-5

    step = perform_correction_step(
        propagator=propagator,
        reference=reference,
        sampler=sampler,
        current_time=0.0,
        current_state=disturbed_state,
        correction_interval=0.25 * reference.period,
    )

    assert step["control_state"].shape == (6,)
    assert step["post_maneuver_state"].shape == (6,)
    assert step["next_state"].shape == (6,)
    assert step["correction"].delta_v.shape == (3,)
    assert step["next_time"] > step["control_time"] > 0.0


def test_run_station_keeping_builds_history_and_corrections():
    propagator = Propagator(CR3BP(mu=EARTH_MOON_MU))
    corrector = DifferentialCorrector(propagator)
    corrected = corrector.solve(
        EARTH_MOON_L2_NORTHERN_HALO_GUESS,
        tf_propagation=8.0,
        orbit_tf=8.0,
    )
    reference = build_full_period_reference(propagator, corrected, samples_per_period=250)
    problem = StationKeepingProblem(
        reference=reference,
        correction_interval=0.25 * reference.period,
        duration=0.75 * reference.period,
    )

    disturbed_state = corrected.initial_state.copy()
    disturbed_state[0] += 1e-5

    history = run_station_keeping(
        problem=problem,
        propagator=propagator,
        initial_state=disturbed_state,
    )

    assert len(history.corrections) == 3
    assert len(history.times) == len(history.states)
    assert np.isclose(history.times[0], 0.0)
