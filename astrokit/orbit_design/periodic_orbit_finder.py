import numpy as np
from scipy.optimize import root
from .reference_trajectory import ReferenceTrajectory


class PeriodicOrbitFinder:
    """
    Generalized periodic orbit finder for various orbit families in the CR3BP.
    Supports different constraint types and crossing planes.
    """

    def __init__(self, propagator):
        self.propagator = propagator

    def _find_crossing(self, t, y_component, crossing_type='positive_to_negative'):
        """
        Find zero crossing in a state component.

        Parameters:
        y_component: 1D array of a state component
        crossing_type: 'positive_to_negative', 'negative_to_positive', or 'zero_crossing'
        """
        for i in range(1, len(y_component)):
            if crossing_type == 'positive_to_negative':
                if y_component[i-1] > 0 and y_component[i] <= 0:
                    alpha = y_component[i-1] / (y_component[i-1] - y_component[i])
                    return i, t[i-1] + alpha * (t[i] - t[i-1])
            elif crossing_type == 'negative_to_positive':
                if y_component[i-1] < 0 and y_component[i] >= 0:
                    alpha = -y_component[i-1] / (y_component[i] - y_component[i-1])
                    return i, t[i-1] + alpha * (t[i] - t[i-1])
            elif crossing_type == 'zero_crossing':
                if y_component[i-1] * y_component[i] < 0:
                    alpha = abs(y_component[i-1]) / (abs(y_component[i-1]) + abs(y_component[i]))
                    return i, t[i-1] + alpha * (t[i] - t[i-1])
        return None, None

    def solve_halo(self, initial_guess, tf_search=20.0):
        """
        Solve for L2 halo orbit family (y-plane crossing, symmetric).
        Free variables: [x0, vy0], Fixed: y0=0, z0, vx0=0, vz0=0
        Constraints: vx_cross = 0, vz_cross = 0
        """
        base_state = initial_guess.copy()

        free_vars0 = np.array([
            base_state[0],
            base_state[4]
        ])

        def constraint_func(free_vars):
            state0 = base_state.copy()
            state0[0] = free_vars[0]
            state0[4] = free_vars[1]

            result = self.propagator.propagate(state0, tf=tf_search)
            y = result.states[1]

            crossing_index, _ = self._find_crossing(result.t, y, 'positive_to_negative')
            if crossing_index is None:
                raise RuntimeError("No y = 0 crossing found")

            crossing_state = result.states[:, crossing_index]
            return np.array([crossing_state[3], crossing_state[5]])  # [vx, vz]

        solution = root(fun=constraint_func, x0=free_vars0, method="hybr")

        if not solution.success:
            raise RuntimeError(f"Halo orbit correction failed: {solution.message}")

        corrected_state = base_state.copy()
        corrected_state[0] = solution.x[0]
        corrected_state[4] = solution.x[1]

        # Find period
        result = self.propagator.propagate(corrected_state, tf=tf_search)
        crossing_index, crossing_time = self._find_crossing(result.t, result.states[1], 'positive_to_negative')
        if crossing_index is None:
            raise RuntimeError("No y = 0 crossing found after correction")

        period = crossing_time

        return ReferenceTrajectory(
            initial_state=corrected_state,
            period=period,
            t=result.t[:crossing_index+1],
            states=result.states[:, :crossing_index+1],
            family_type='halo'
        )




    def solve_generic(self, initial_guess, family_type='halo', **kwargs):
        """
        Generic interface to solve different orbit families.

        Parameters:
        initial_guess: Starting state vector
        family_type: Currently only 'halo' is supported
        **kwargs: Additional arguments passed to specific solver
        """
        if family_type.lower() == 'halo':
            return self.solve_halo(initial_guess, **kwargs)
        else:
            raise ValueError(f"Unknown family type: {family_type}. Only 'halo' is supported.")
