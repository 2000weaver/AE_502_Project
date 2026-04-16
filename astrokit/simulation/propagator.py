import numpy as np
from scipy.integrate import solve_ivp
from .results import PropagationResult


class Propagator:
    def __init__(self, model, method="DOP853", rtol=1e-12, atol=1e-12):
        self.model  = model
        self.method = method
        self.rtol   = rtol
        self.atol   = atol

    # ------------------------------------------------------------------
    # Standard propagation (unchanged)
    # ------------------------------------------------------------------

    def propagate(self, state0: np.ndarray, tf: float, n_eval: int = 5000) -> PropagationResult:
        """Integrate the CR3BP EOM from state0 to tf."""
        t_eval = np.linspace(0, tf, n_eval)

        sol = solve_ivp(
            fun=self.model.equations_of_motion,
            t_span=(0, tf),
            y0=state0,
            method=self.method,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
        )

        jacobi = np.array([
            self.model.jacobi_constant(sol.y[:, i])
            for i in range(sol.y.shape[1])
        ])

        return PropagationResult(t=sol.t, states=sol.y, jacobi=jacobi)

    # ------------------------------------------------------------------
    # STM-aware propagation
    # ------------------------------------------------------------------

    def propagate_stm(
        self,
        state0: np.ndarray,
        tf: float,
        n_eval: int = 5000,
    ) -> PropagationResult:
        """
        Integrate the CR3BP EOM *and* the State Transition Matrix simultaneously.

        The STM Φ(t, t0) satisfies:
            dΦ/dt = A(t) Φ,   Φ(t0) = I

        Returns a PropagationResult with an additional attribute:
            result.stm   — ndarray of shape (6, 6, n_eval)
                           stm[:, :, k] is Φ(t_k, t0)
            result.monodromy — ndarray (6, 6)
                           Φ(T, t0), i.e. the monodromy matrix at the final time

        Parameters
        ----------
        state0 : array-like, shape (6,)
            Initial state [x, y, z, vx, vy, vz] in CR3BP non-dimensional units.
        tf : float
            Final integration time (non-dimensional).
        n_eval : int
            Number of output time steps.
        """
        # Build 42-element initial vector: state + identity STM flattened
        Phi0      = np.eye(6).ravel()
        state_stm0 = np.concatenate([state0, Phi0])

        t_eval = np.linspace(0, tf, n_eval)

        sol = solve_ivp(
            fun=self.model.equations_of_motion_stm,
            t_span=(0, tf),
            y0=state_stm0,
            method=self.method,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
        )

        # Split solution back into state and STM histories
        states = sol.y[:6, :]                          # (6, n_eval)
        stm_flat = sol.y[6:, :]                        # (36, n_eval)
        stm    = stm_flat.reshape(6, 6, -1)            # (6, 6, n_eval)

        jacobi = np.array([
            self.model.jacobi_constant(states[:, i])
            for i in range(states.shape[1])
        ])

        result            = PropagationResult(t=sol.t, states=states, jacobi=jacobi)
        result.stm        = stm                        # full STM history
        result.monodromy  = stm[:, :, -1]             # Φ(T, t0)

        return result