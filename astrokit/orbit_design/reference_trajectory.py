from dataclasses import dataclass, field
import numpy as np

@dataclass 
class ReferenceTrajectory:
    initial_state: np.ndarray
    period: float
    t: np.ndarray
    states: np.ndarray
    family_type: str = 'halo'
    jacobi_constant: float = field(default_factory=lambda: 0.0)
    metadata: dict = field(default_factory=dict)

    def save(self, filename):
        np.savez(
            filename, 
            initial_state=self.initial_state, 
            period=self.period, 
            t=self.t, 
            states=self.states,
            family_type=self.family_type,
            jacobi_constant=self.jacobi_constant
        )

    @classmethod
    def load(cls, filename):
        data = np.load(filename)
        
        family_type = str(data['family_type']) if 'family_type' in data else 'halo'
        jacobi_constant = float(data['jacobi_constant']) if 'jacobi_constant' in data else 0.0
        
        return cls(
            initial_state=data['initial_state'],
            period=float(data['period']),
            t=data['t'],
            states=data['states'],
            family_type=family_type,
            jacobi_constant=jacobi_constant
        )