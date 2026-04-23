import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

from ..utils.constants import EARTH_MOON_MU

__all__ = [
    "plot_trajectory_3d",
    "plot_jacobi",
    "show_figure",
    "plot_multiple_trajectories_3d",
    "plot_position_error_3d",
]


def _build_periodic_reference_positions(reference_trajectory):
    reference_times = np.asarray(reference_trajectory.t, dtype=float)
    reference_positions = np.asarray(reference_trajectory.states[:3], dtype=float)

    if reference_times.ndim != 1:
        raise ValueError("reference_trajectory.t must be a 1D array")
    if reference_positions.shape[1] != reference_times.size:
        raise ValueError("Reference trajectory state history must align with reference times")
    if reference_times[0] != 0.0:
        raise ValueError("Reference trajectory must start at t = 0.")
    if reference_trajectory.period <= 0.0:
        raise ValueError("Reference trajectory period must be positive")
    if reference_times[-1] + 1e-10 < reference_trajectory.period:
        raise ValueError(
            "Reference history must span a full period. "
            "Propagate the corrected initial state over one full period and pass "
            "that full history into plot_position_error_3d()."
        )

    return reference_times, reference_positions


def _sample_reference_positions(reference_trajectory, sample_times):
    reference_times, reference_positions = _build_periodic_reference_positions(reference_trajectory)
    wrapped_times = np.mod(np.asarray(sample_times, dtype=float), reference_trajectory.period)

    sampled_positions = np.vstack(
        [
            np.interp(wrapped_times, reference_times, reference_positions[axis])
            for axis in range(3)
        ]
    )
    return sampled_positions

def plot_trajectory_3d(states, mu = EARTH_MOON_MU, fig=None, format_dict=None):
    
    # Default formatting
    default_format = {
        'trajectory': {
            'name': 'CR3BP Orbit',
            'mode': 'lines',
            'line': {'width': 4, 'color': 'black'}
        },
        'primary': {
            'name': 'Primary',
            'mode': 'markers',
            'marker': {'size': 8, 'color': 'red', 'symbol': 'diamond'}
        },
        'secondary': {
            'name': 'Secondary', 
            'mode': 'markers',
            'marker': {'size': 6, 'color': 'blue', 'symbol': 'diamond'}
        },
        'layout': {
            'scene': {
                'xaxis': {'title': 'X Axis'},
                'yaxis': {'title': 'Y Axis'},
                'zaxis': {'title': 'Z Axis'},
                'aspectmode': 'cube',
                'aspectratio': {'x': 1, 'y': 1, 'z': 1}
            }
        }
    }
    
    # Update defaults with user format_dict
    if format_dict:
        for key in format_dict:
            if key in default_format:
                default_format[key].update(format_dict[key])
    
    # Create figure if not provided
    if fig is None:
        fig = go.Figure()
    
    # Add trajectory trace
    fig.add_trace(
        go.Scatter3d(
            x = states[0],
            y = states[1],
            z = states[2],
            **default_format['trajectory']
        )
    )

    # Add primary body
    fig.add_trace(
        go.Scatter3d(
            x = [-mu],
            y = [0],
            z = [0],
            **default_format['primary']
        )
    )

    # Add secondary body
    fig.add_trace(
        go.Scatter3d(
            x = [1 - mu],
            y = [0], 
            z = [0],
            **default_format['secondary']
        )
    )

    # Update layout only if this is a new figure
    if len(fig.data) == 3:  # Only update if we just added the three traces
        fig.update_layout(**default_format['layout'])

    return fig


def plot_jacobi(t, jacobi, fig=None, format_dict=None):
    
    # Default formatting
    default_format = {
        'jacobi': {
            'name': 'Jacobi Constant',
            'mode': 'lines'
        },
        'layout': {
            'xaxis': {'title': 'Time Units'},
            'yaxis': {
                'title': 'Jacobi Constant [m^2 / s^2]',
                'range': [min(jacobi) - 1e-5, max(jacobi) + 1e-5]
            }
        }
    }
    
    # Update defaults with user format_dict
    if format_dict:
        for key in format_dict:
            if key in default_format:
                default_format[key].update(format_dict[key])
    
    # Create figure if not provided
    if fig is None:
        fig = go.Figure()
    
    # Add jacobi trace
    fig.add_trace(
        go.Scatter(
            x = t, 
            y = jacobi,
            **default_format['jacobi']
        )
    )

    # Update layout only if this is a new figure
    if len(fig.data) == 1:  # Only update if we just added the first trace
        fig.update_layout(**default_format['layout'])

    return fig


def show_figure(fig):
    """Display a plotly figure."""
    fig.show()


def plot_multiple_trajectories_3d(trajectory_list, mu=EARTH_MOON_MU, fig=None, format_dicts=None):
    """
    Plot multiple trajectories on the same 3D figure.
    
    Parameters:
    trajectory_list: List of state arrays to plot
    mu: Mass parameter
    fig: Existing figure to add to (optional)
    format_dicts: List of format dictionaries, one per trajectory (optional)
    """
    if fig is None:
        fig = go.Figure()
    
    if format_dicts is None:
        format_dicts = [None] * len(trajectory_list)
    
    for i, (states, fmt_dict) in enumerate(zip(trajectory_list, format_dicts)):
        # Default name for multiple trajectories
        default_name = f'CR3BP Orbit {i+1}'
        if fmt_dict and 'trajectory' in fmt_dict and 'name' not in fmt_dict['trajectory']:
            fmt_dict['trajectory'] = fmt_dict.get('trajectory', {})
            fmt_dict['trajectory']['name'] = default_name
        elif fmt_dict is None:
            fmt_dict = {'trajectory': {'name': default_name}}
        
        fig = plot_trajectory_3d(states, mu=mu, fig=fig, format_dict=fmt_dict)
    
    return fig


def plot_position_error_3d(reference_trajectory, trajectory_dict, fig=None, format_dict=None):
    """
    Plot 3D position error histories against a periodic reference trajectory.

    Parameters
    ----------
    reference_trajectory : ReferenceTrajectory
        Periodic reference trajectory used to compute position error.
    trajectory_dict : dict[str, PropagationResult]
        Mapping of legend label to propagated trajectory result. Each result must
        contain `t` and `states`, where `states[:3]` are the propagated position
        components in the same frame as the reference.
    fig : plotly.graph_objects.Figure, optional
        Existing figure to add traces to.
    format_dict : dict, optional
        Optional formatting overrides for the error traces and layout.
    """
    default_format = {
        "error": {
            "mode": "lines",
            "line": {"width": 4},
        },
        "origin": {
            "name": "Zero Error",
            "mode": "markers",
            "marker": {"size": 5, "color": "black", "symbol": "x"},
        },
        "layout": {
            "scene": {
                "xaxis": {"title": "x Error"},
                "yaxis": {"title": "y Error"},
                "zaxis": {"title": "z Error"},
                "aspectmode": "cube",
                "aspectratio": {"x": 1, "y": 1, "z": 1},
            },
            "title": "3D Position Error Relative to Reference Trajectory",
        },
    }

    if format_dict:
        for key in format_dict:
            if key in default_format and isinstance(default_format[key], dict):
                default_format[key].update(format_dict[key])

    if fig is None:
        fig = go.Figure()

    for label, trajectory_result in trajectory_dict.items():
        sample_times = np.asarray(trajectory_result.t, dtype=float)
        propagated_positions = np.asarray(trajectory_result.states[:3], dtype=float)

        if propagated_positions.shape[1] != sample_times.size:
            raise ValueError(
                f"Trajectory '{label}' has mismatched time and state history lengths"
            )

        reference_positions = _sample_reference_positions(reference_trajectory, sample_times)
        position_error = propagated_positions - reference_positions

        trace_format = {
            key: (value.copy() if isinstance(value, dict) else value)
            for key, value in default_format["error"].items()
        }
        trace_format["name"] = label

        fig.add_trace(
            go.Scatter3d(
                x=position_error[0],
                y=position_error[1],
                z=position_error[2],
                **trace_format,
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            **default_format["origin"],
        )
    )

    fig.update_layout(**default_format["layout"])
    return fig
