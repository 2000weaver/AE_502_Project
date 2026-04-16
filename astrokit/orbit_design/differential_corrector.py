import numpy as np
from scipy.optimize import root

from .reference_trajectory import ReferenceTrajectory
from ..utils.crossing import find_y_crossing

class DifferentialCorrector:
    def __init__(self, propagator):
        self.propagator = propagator

    def constraint_function(self, free_vars, base_state, tf=10.0):
        """
        free_vars = [x0, vy0]
        base_state already contains z0 and other fixed entries
        """
        from scipy.interpolate import interp1d

        state0 = base_state.copy()
        state0[0] = free_vars[0]
        state0[4] = free_vars[1]

        result = self.propagator.propagate(state0, tf=tf, n_eval=10000)
        
        idx, t_cross = find_y_crossing(result.t, result.states)
        if idx is None:
            raise RuntimeError("No y = 0 crossing found in constraint function")

        # Interpolate to exact crossing time instead of using grid point
        interp = interp1d(result.t, result.states, kind='cubic', axis=1)
        crossing_state = interp(t_cross)

        return np.array([crossing_state[3], crossing_state[5]])  # vx, vz

    def solve(self, initial_guess):
        base_state = initial_guess.copy()

        free_vars0 = np.array([
            base_state[0],
            base_state[4]
        ])

        try:
            solution = root(
                fun=lambda vars: self.constraint_function(vars, base_state),
                x0=free_vars0,
                method="hybr",
                tol=1e-11,
            )
        except Exception as e:
            print(f"DC root() success={solution.success}")
            print(f"DC final residual={solution.fun}")
            print(f"DC message={solution.message}")
            raise RuntimeError(f"Differential correction failed with error: {e}")

        if not solution.success:
            raise RuntimeError("Differential correction failed")

        corrected_state = base_state.copy()
        corrected_state[0] = solution.x[0]
        corrected_state[4] = solution.x[1]

        # propagate and determine period from y-plane crossing
        orbit_tf = 20.0
        result = self.propagator.propagate(corrected_state, tf=orbit_tf)

        crossing_index, crossing_time = find_y_crossing(result.t, result.states)
        if crossing_index is None:
            raise RuntimeError("No y = 0 crossing found after correction")

        period = 2 * crossing_time

        return ReferenceTrajectory(
            initial_state=corrected_state,
            period=period,
            t=result.t[:crossing_index+1],
            states=result.states[:, :crossing_index+1]
        )