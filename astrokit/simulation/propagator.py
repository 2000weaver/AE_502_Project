import numpy as np
from scipy.integrate import solve_ivp
from .results import PropagationResult

class Propagator:
    def __init__(self, model, method="DOP853", rtol=1e-12, atol=1e-12):
        self.model = model
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def propagate(self, state0, tf, n_eval=1000):
        t_eval = np.linspace(0, tf, n_eval)

        sol = solve_ivp(
            fun=self.model.equations_of_motion,
            t_span=(0, tf),
            y0=state0,
            method=self.method,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol
        )

        jacobi = np.array([
            self.model.jacobi_constant(sol.y[:,i])
            for i in range(sol.y.shape[1])
        ])

        return PropagationResult(
            t=sol.t,
            states=sol.y,
            jacobi=jacobi
        )