import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def plot_trajectory_3d(states, mu):
    
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x = states[0],
            y = states[1],
            z = states[2],
            name = "CR3PB Orbit",
            mode = "lines",
            line = dict(
                width = 4,
                color = "black"
            )
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x = [-mu],
            y = [0],
            z = [0],
            name = "Primary",
            mode = "markers",
            marker = dict(
                size = 8,
                color = "red",
                symbol = "diamond"
            )
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x = [1 - mu],
            y = [0], 
            z = [0],
            name = "Secondary",
            mode = "markers",
            marker = dict(
                size = 6,
                color = "blue",
                symbol = "diamond"
            )
        )
    )

    fig.update_layout(
        scene = dict(
            xaxis = dict(
                title = "X Axis"
            ),
            yaxis = dict(
                title = "Y Axis"
            ),
            zaxis = dict(
                title = "Z Axis"
            ),
            aspectmode = "cube",
            aspectratio = dict(x=1, y=1, z=1)
        )
    )

    fig.show()


def plot_jacobi(t, jacobi):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x = t, 
            y = jacobi,
            name = "Jacobi Constant",
            mode = "lines"
        )
    )

    fig.update_layout(
        xaxis = dict(
            title = "Time Units"
        ),
        yaxis = dict(
            title = "Jacobi Constant [m^2 / s^2]",
            range = [min(jacobi) - 1e-5, max(jacobi) + 1e-5]
        )
    )

    fig.show()