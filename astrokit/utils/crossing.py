# astrokit/utils/crossing.py

import numpy as np

def find_y_crossing(t, states, t_min=0.5):
    """
    Find the first positive-to-negative y=0 crossing after t_min.
    Returns (index, interpolated_time) or (None, None).
    """
    y = states[1] if states.ndim == 2 else states
    for i in range(1, len(y)):
        if t[i] < t_min:
            continue
        if y[i-1] > 0 and y[i] <= 0:
            alpha = y[i-1] / (y[i-1] - y[i])
            return i, t[i-1] + alpha * (t[i] - t[i-1])
    return None, None