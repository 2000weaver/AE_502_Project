import numpy as np
from scipy.optimize import root
from typing import List, Optional

from .reference_trajectory import ReferenceTrajectory
from ..utils.crossing import find_y_crossing


class DifferentialCorrector:
    """
    Differential corrector for finding periodic orbits in the CR3BP.
    
    Solves for periodic orbits by correcting initial conditions to satisfy 
    periodicity constraints. Supports configurable free variables and constraints.
    
    Parameters
    ----------
    propagator : Propagator
        Propagator object with propagate() method.
    free_var_indices : list[int], optional
        Indices of free variables in state vector [x, y, z, vx, vy, vz].
        Default: [0, 4] for halo orbits (x0, vy0).
    constraint_indices : list[int], optional
        Indices of constraints to satisfy at y=0 crossing.
        Default: [3, 5] for halo orbits (vx=0, vz=0).
    
    Examples
    --------
    Halo orbit (x0, vy0 free; vx=0, vz=0 at crossing):
        corrector = DifferentialCorrector(propagator)
    
    NRHO orbit (x0, z0, vy0 free; y=0, vy=0, vx=0 at crossing):
        corrector = DifferentialCorrector(
            propagator,
            free_var_indices=[0, 2, 4],
            constraint_indices=[1, 4, 3]
        )
    
    Custom orbit:
        corrector = DifferentialCorrector(
            propagator,
            free_var_indices=[0, 1, 4],
            constraint_indices=[3, 5]
        )
    """
    
    # State vector indices: [x, y, z, vx, vy, vz] = [0, 1, 2, 3, 4, 5]
    
    def __init__(
        self,
        propagator,
        free_var_indices: Optional[List[int]] = None,
        constraint_indices: Optional[List[int]] = None,
    ):
        """
        Initialize the differential corrector.
        
        Parameters
        ----------
        propagator : Propagator
            Orbital propagator.
        free_var_indices : list[int], optional
            Indices of free variables. Default: [0, 4] (x0, vy0).
        constraint_indices : list[int], optional
            Indices of constraints at y=0 crossing. Default: [3, 5] (vx, vz).
        """
        self.propagator = propagator
        
        # Default to halo orbit configuration
        self.free_var_indices = free_var_indices if free_var_indices is not None else [0, 4]
        self.constraint_indices = constraint_indices if constraint_indices is not None else [3, 5]
        
        # Validate
        if len(self.free_var_indices) != len(self.constraint_indices):
            raise ValueError(
                f"Number of free variables ({len(self.free_var_indices)}) "
                f"must equal number of constraints ({len(self.constraint_indices)})"
            )
        
        if len(self.free_var_indices) == 0:
            raise ValueError("Must have at least one free variable")
        
        # Check for valid state indices (0-5 for 6D state)
        all_indices = self.free_var_indices + self.constraint_indices
        if any(i < 0 or i > 5 for i in all_indices):
            raise ValueError("State indices must be in range [0, 5]")

    def set_free_variables(self, indices: List[int]):
        """
        Update the free variable indices.
        
        Parameters
        ----------
        indices : list[int]
            New free variable indices.
        """
        if len(indices) == 0:
            raise ValueError("Must have at least one free variable")
        self.free_var_indices = indices

    def set_constraints(self, indices: List[int]):
        """
        Update the constraint indices.
        
        Parameters
        ----------
        indices : list[int]
            New constraint indices.
        """
        if len(indices) == 0:
            raise ValueError("Must have at least one constraint")
        self.constraint_indices = indices

    def _starts_on_symmetry_plane(self, state: np.ndarray) -> bool:
        """Return True when the state matches the usual halo-orbit symmetry seed."""
        return (
            np.isclose(state[1], 0.0, atol=1e-10)
            and np.isclose(state[3], 0.0, atol=1e-10)
            and np.isclose(state[5], 0.0, atol=1e-10)
        )

    def _estimate_period(
        self,
        initial_state: np.ndarray,
        result,
        period_hint: float | None = None,
    ) -> tuple[float, str]:
        """
        Estimate the full period from a propagated trajectory.

        For symmetry-plane seeds we retain the classic halo shortcut:
        half-period = first positive-to-negative y=0 crossing.

        For arbitrary-phase seeds we estimate the period from the first
        meaningful return of the full 6D state to the initial condition.
        """
        if period_hint is not None and period_hint <= 0.0:
            raise ValueError("period_hint must be positive when provided")

        if self._starts_on_symmetry_plane(initial_state) and period_hint is None:
            _, crossing_time = find_y_crossing(result.t, result.states)
            if crossing_time is None:
                raise RuntimeError("No y = 0 crossing found after correction")
            return 2.0 * crossing_time, "symmetry_plane_crossing"

        times = np.asarray(result.t, dtype=float)
        states = np.asarray(result.states, dtype=float)
        state_error_norm = np.linalg.norm(
            states - np.asarray(initial_state, dtype=float).reshape(-1, 1),
            axis=0,
        )

        if times.size < 3:
            raise RuntimeError("Not enough propagated samples to estimate orbit period")

        dt = times[1] - times[0]
        t_min = max(0.5, 5.0 * dt)
        local_min_mask = (
            (state_error_norm[1:-1] <= state_error_norm[:-2])
            & (state_error_norm[1:-1] <= state_error_norm[2:])
        )
        candidate_indices = np.where(local_min_mask)[0] + 1
        candidate_indices = candidate_indices[times[candidate_indices] >= t_min]

        if candidate_indices.size == 0:
            raise RuntimeError(
                "Unable to estimate full orbit period from the propagated return map"
            )

        min_candidate_error = float(np.min(state_error_norm[candidate_indices]))
        eligible_indices = candidate_indices[
            state_error_norm[candidate_indices]
            <= max(10.0 * min_candidate_error, 1.0e-8)
        ]

        if period_hint is not None:
            hint_idx = int(np.argmin(np.abs(times[eligible_indices] - float(period_hint))))
            period_index = int(eligible_indices[hint_idx])
            return float(times[period_index]), "state_recurrence_period_hint"

        period_index = int(eligible_indices[0])
        return float(times[period_index]), "state_recurrence"

    def constraint_function(self, free_vars, base_state, tf=10.0):
        """
        Evaluate constraints at y=0 crossing.
        
        Parameters
        ----------
        free_vars : np.ndarray
            Values for free variables.
        base_state : np.ndarray
            Base state with fixed variables.
        tf : float
            Integration time.
            
        Returns
        -------
        np.ndarray
            Constraint residuals (should be zero for periodic orbit).
        """
        from scipy.interpolate import interp1d

        state0 = base_state.copy()
        
        # Inject free variables into state
        for i, var_idx in enumerate(self.free_var_indices):
            state0[var_idx] = free_vars[i]

        # Propagate
        result = self.propagator.propagate(state0, tf=tf, n_eval=10000)
        
        # Find y=0 crossing
        idx, t_cross = find_y_crossing(result.t, result.states)
        if idx is None:
            raise RuntimeError("No y = 0 crossing found in constraint function")
        
        # Interpolate to exact crossing time
        interp = interp1d(result.t, result.states, kind='cubic', axis=1)
        crossing_state = interp(t_cross)
        
        # Extract constraint values
        constraints = np.array([crossing_state[c_idx] for c_idx in self.constraint_indices])
        return constraints

    def solve(self, initial_guess, tf_propagation=10.0, orbit_tf=20.0, period_hint=None):
        """
        Solve for periodic orbit using differential correction.
        
        Parameters
        ----------
        initial_guess : np.ndarray or ReferenceTrajectory
            Initial state vector [x, y, z, vx, vy, vz], or a reference orbit
            whose initial state should be corrected.
        tf_propagation : float, optional
            Integration time for constraint evaluation. Default: 10.0.
        orbit_tf : float, optional
            Integration time for final orbit. Default: 20.0.
        period_hint : float, optional
            Guide for the corrected-orbit period estimate. Useful when the
            initial state comes from a known periodic orbit at arbitrary phase.
            
        Returns
        -------
        ReferenceTrajectory
            Corrected periodic orbit.
        """
        inferred_period_hint = period_hint
        if isinstance(initial_guess, ReferenceTrajectory):
            base_state = np.asarray(initial_guess.initial_state, dtype=float).copy()
            if inferred_period_hint is None and initial_guess.period > 0.0:
                inferred_period_hint = float(initial_guess.period)
        else:
            base_state = np.asarray(initial_guess, dtype=float).copy()
        
        # Extract initial free variables
        free_vars0 = np.array([base_state[i] for i in self.free_var_indices])
        
        try:
            solution = root(
                fun=lambda vars: self.constraint_function(vars, base_state, tf=tf_propagation),
                x0=free_vars0,
                method="hybr",
                tol=1e-11,
            )
        except Exception as e:
            raise RuntimeError(f"Differential correction failed with error: {e}")
        
        if not solution.success:
            raise RuntimeError(f"Differential correction failed: {solution.message}")
        
        # Update state with corrected free variables
        corrected_state = base_state.copy()
        for i, var_idx in enumerate(self.free_var_indices):
            corrected_state[var_idx] = solution.x[i]
        
        # Propagate to estimate the full period of the corrected orbit.
        result = self.propagator.propagate(corrected_state, tf=orbit_tf)

        period, period_estimation_method = self._estimate_period(
            corrected_state,
            result,
            period_hint=inferred_period_hint,
        )
        if period >= result.t[-1]:
            raise RuntimeError(
                "Estimated period reaches the propagation horizon. "
                "Increase orbit_tf so the corrected orbit spans at least one full period."
            )

        end_index = int(np.searchsorted(result.t, period, side="left"))
        t_history = np.asarray(result.t[: end_index + 1], dtype=float).copy()
        state_history = np.asarray(result.states[:, : end_index + 1], dtype=float).copy()
        t_history[-1] = period
        state_history[:, -1] = corrected_state

        return ReferenceTrajectory(
            initial_state=corrected_state,
            period=period,
            t=t_history,
            states=state_history,
            metadata={"period_estimation_method": period_estimation_method},
        )
