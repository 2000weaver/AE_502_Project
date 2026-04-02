from dataclasses import dataclass
import numpy as np 

@dataclass
class PropagationResult:

    t: np.ndarray
    states: np.ndarray
    jacobi: np.ndarray | None = None