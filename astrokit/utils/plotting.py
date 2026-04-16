import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

from ..utils.constants import EARTH_MOON_MU

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