from abc import ABC, abstractmethod
import numpy as np

class DynamicalModel(ABC):
    @abstractmethod
    def equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
        pass