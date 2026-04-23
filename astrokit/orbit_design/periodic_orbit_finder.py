from .differential_corrector import DifferentialCorrector

__all__ = ["PeriodicOrbitFinder"]


class PeriodicOrbitFinder:
    """
    Thin compatibility wrapper around DifferentialCorrector.

    This keeps the previous public API but routes halo-orbit solving through a
    single correction implementation so the codebase is easier to maintain.
    """

    def __init__(self, propagator):
        self.propagator = propagator

    def solve_halo(self, initial_guess, tf_search=20.0):
        corrector = DifferentialCorrector(self.propagator)
        return corrector.solve(
            initial_guess,
            tf_propagation=tf_search,
            orbit_tf=tf_search,
        )

    def solve_generic(self, initial_guess, family_type="halo", **kwargs):
        if family_type.lower() == "halo":
            return self.solve_halo(initial_guess, **kwargs)
        raise ValueError(f"Unknown family type: {family_type}. Only 'halo' is supported.")
