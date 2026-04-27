import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.figure import Figure as MatplotlibFigure
from pathlib import Path
import re
import csv

from astrokit.models.cr3bp import CR3BP
from astrokit.models.perturbing_body import create_solar_perturbation
from astrokit.orbit_design.differential_corrector import DifferentialCorrector
from astrokit.orbit_design.jpl_api_client import query_earth_moon_halo
from astrokit.orbit_design.periodic_reference import (
    build_full_period_reference,
    loop_reference_trajectory,
)
from astrokit.simulation import PropagationResult, Propagator
from astrokit.control.station_keeping import (
    StationKeepingProblem,
    build_reference_sampler,
    reference_state_at_time,
    run_station_keeping,
    sweep_correction_intervals,
)
from astrokit.utils.constants import (
    EARTH_MOON_LENGTH_UNIT_M,
    EARTH_MOON_MU,
    EARTH_MOON_VELOCITY_UNIT_MPS,
)
from astrokit.utils.plotting import (
    plot_jacobi,
    plot_phase_error,
    plot_position_error_magnitude,
    show_figure,
)


BRANCHES = [
    {"code": "N", "name": "Northern", "color": "#1f77b4"},
    {"code": "S", "name": "Southern", "color": "#d62728"},
]
MODEL_STYLES = {
    "CR3BP": {"color": "#1f77b4", "dash": "solid"},
    "BR4BP": {"color": "#c0392b", "dash": "solid"},
    "Reference": {"color": "#111111", "dash": "dash"},
    "StationKept": {"color": "#2e8b57", "dash": "solid"},
    "Uncontrolled": {"color": "#7f7f7f", "dash": "dot"},
}

EARTH_MOON_LENGTH_UNIT_KM = EARTH_MOON_LENGTH_UNIT_M / 1000.0

REFERENCE_PERIODS = 10
STATION_KEEPING_PERIODS = 8
SAMPLES_PER_PERIOD_REFERENCE = 6000
SAMPLES_PER_PERIOD_PROPAGATION = 3200
SWEEP_INTERVAL_COUNT = 50
SWEEP_MIN_INTERVAL = 0.05
SWEEP_MAX_INTERVAL = 0.95
MAX_DELTA_V_FRACTION = 0.075
CONTROL_OFFSET_FRACTION = 0.5
MAX_PHI_RV_CONDITION_NUMBER = 1.0e7
RNG_SEED = 12
FIGURE_OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_figures"
TABLE_OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_tables"

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"


def apply_report_layout(fig, title, legend_title=None):
    is_3d = "scene" in fig.layout
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center"},
        template="plotly_white",
        font={
            "family": "STIX Two Text, Times New Roman, serif",
            "size": 15 if is_3d else 17,
            "color": "#111111",
        },
        title_font={"size": 22 if is_3d else 22},
        legend={
            "title": {"text": legend_title} if legend_title else None,
            "orientation": "v" if is_3d else "h",
            "yanchor": "top" if is_3d else "bottom",
            "y": 0.98 if is_3d else 1.02,
            "xanchor": "left" if is_3d else "right",
            "x": 1.02 if is_3d else 1.0,
            "bgcolor": "rgba(255,255,255,0.90)",
            "bordercolor": "rgba(0,0,0,0.15)",
            "borderwidth": 1,
            "font": {"size": 14 if is_3d else 13},
        },
        margin={"l": 60, "r": 170 if is_3d else 40, "t": 85 if is_3d else 90, "b": 60},
    )

    if "xaxis" in fig.layout:
        fig.update_xaxes(
            showline=True,
            linewidth=1.2,
            linecolor="black",
            mirror=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            ticks="outside",
        )
    if "yaxis" in fig.layout:
        fig.update_yaxes(
            showline=True,
            linewidth=1.2,
            linecolor="black",
            mirror=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            ticks="outside",
        )
    if "scene" in fig.layout:
        fig.update_layout(
            scene={
                **fig.layout.scene.to_plotly_json(),
                "aspectmode": "manual",
                "aspectratio": {"x": 1.15, "y": 1.0, "z": 0.95},
                "camera": {
                    "eye": {"x": 1.05, "y": 0.98, "z": 0.72},
                    "up": {"x": 0.0, "y": 0.0, "z": 1.0},
                },
                "bgcolor": "white",
                "xaxis": {
                    **fig.layout.scene.xaxis.to_plotly_json(),
                    "title": {"font": {"size": 14}},
                    "tickfont": {"size": 12},
                    "backgroundcolor": "white",
                    "gridcolor": "rgba(0,0,0,0.08)",
                    "showline": True,
                    "linecolor": "black",
                },
                "yaxis": {
                    **fig.layout.scene.yaxis.to_plotly_json(),
                    "title": {"font": {"size": 14}},
                    "tickfont": {"size": 12},
                    "backgroundcolor": "white",
                    "gridcolor": "rgba(0,0,0,0.08)",
                    "showline": True,
                    "linecolor": "black",
                },
                "zaxis": {
                    **fig.layout.scene.zaxis.to_plotly_json(),
                    "title": {"font": {"size": 14}},
                    "tickfont": {"size": 12},
                    "backgroundcolor": "white",
                    "gridcolor": "rgba(0,0,0,0.08)",
                    "showline": True,
                    "linecolor": "black",
                },
            }
        )
        fig.update_layout(width=1280, height=820)
    return fig


def combine_2d_figures_side_by_side(figures, title, subplot_titles=None):
    if not figures:
        raise ValueError("figures must contain at least one 2D figure")
    if len(figures) > 2:
        raise ValueError("combine_2d_figures_side_by_side supports at most two figures")

    subplot_titles = subplot_titles or [""] * len(figures)
    specs = []
    for fig in figures:
        has_secondary_y = any(getattr(trace, "yaxis", None) == "y2" for trace in fig.data)
        specs.append({"secondary_y": has_secondary_y})
    if len(specs) == 1:
        specs.append({"secondary_y": False})
        subplot_titles = list(subplot_titles) + [""]

    combined = make_subplots(
        rows=1,
        cols=2,
        specs=[specs],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.12,
    )

    for col_idx, fig in enumerate(figures, start=1):
        has_secondary_y = any(getattr(trace, "yaxis", None) == "y2" for trace in fig.data)
        for trace in fig.data:
            trace_copy = go.Figure(data=[trace]).data[0]
            combined.add_trace(
                trace_copy,
                row=1,
                col=col_idx,
                secondary_y=(getattr(trace, "yaxis", None) == "y2"),
            )

        xaxis_title = fig.layout.xaxis.title.text if fig.layout.xaxis.title.text else ""
        yaxis_title = fig.layout.yaxis.title.text if fig.layout.yaxis.title.text else ""
        combined.update_xaxes(title_text=xaxis_title, row=1, col=col_idx)
        combined.update_yaxes(title_text=yaxis_title, row=1, col=col_idx, secondary_y=False)
        if has_secondary_y:
            secondary_title = fig.layout.yaxis2.title.text if fig.layout.yaxis2.title.text else ""
            combined.update_yaxes(title_text=secondary_title, row=1, col=col_idx, secondary_y=True)

    combined.update_layout(showlegend=True)
    apply_report_layout(combined, title)
    return combined


def _plotly_dash_to_matplotlib(dash_style):
    return {
        "solid": "-",
        "dash": "--",
        "dot": ":",
        "dashdot": "-.",
    }.get(dash_style, "-")


def _plotly_marker_to_matplotlib(symbol):
    return {
        "circle": "o",
        "square": "s",
        "diamond": "D",
        "x": "x",
        "cross": "+",
        "triangle-up": "^",
        "triangle-down": "v",
    }.get(symbol, "o")


def _get_plotly_axis_title(axis):
    if axis is None or getattr(axis, "title", None) is None:
        return ""
    return axis.title.text or ""


def _style_matplotlib_axis(axis):
    axis.grid(True, color="0.88", linewidth=0.8)
    axis.set_facecolor("white")
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    axis.tick_params(direction="out", width=1.0, colors="black", labelsize=10)


def _plot_plotly_trace_on_axis(axis, trace):
    trace_dict = trace.to_plotly_json()
    x_data = np.asarray(trace_dict.get("x", []), dtype=float)
    y_data = np.asarray(trace_dict.get("y", []), dtype=float)
    mode = trace_dict.get("mode", "lines")
    line_dict = trace_dict.get("line", {})
    marker_dict = trace_dict.get("marker", {})

    linestyle = _plotly_dash_to_matplotlib(line_dict.get("dash", "solid"))
    linewidth = line_dict.get("width", 2.0)
    color = line_dict.get("color")
    marker = _plotly_marker_to_matplotlib(marker_dict.get("symbol", "circle"))
    markersize = max(4.0, min(float(marker_dict.get("size", 6.0)), 8.0))

    if "lines" not in mode:
        linestyle = "None"
    if "markers" not in mode:
        marker = None

    axis.plot(
        x_data,
        y_data,
        label=trace_dict.get("name"),
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markersize=markersize,
    )


def _render_plotly_figure_on_axes(plotly_figure, axis, secondary_axis=None):
    for trace in plotly_figure.data:
        target_axis = secondary_axis if getattr(trace, "yaxis", None) == "y2" and secondary_axis is not None else axis
        _plot_plotly_trace_on_axis(target_axis, trace)

    axis.set_xlabel(_get_plotly_axis_title(getattr(plotly_figure.layout, "xaxis", None)))
    axis.set_ylabel(_get_plotly_axis_title(getattr(plotly_figure.layout, "yaxis", None)))
    _style_matplotlib_axis(axis)

    if secondary_axis is not None:
        secondary_axis.set_ylabel(_get_plotly_axis_title(getattr(plotly_figure.layout, "yaxis2", None)))
        _style_matplotlib_axis(secondary_axis)
        secondary_axis.grid(False)


def build_matplotlib_two_panel_figure(plotly_figures, title, panel_titles):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=14, fontweight="semibold", y=0.98)

    for idx, axis in enumerate(axes):
        if idx >= len(plotly_figures):
            axis.axis("off")
            continue

        source_figure = plotly_figures[idx]
        has_secondary = any(getattr(trace, "yaxis", None) == "y2" for trace in source_figure.data)
        secondary_axis = axis.twinx() if has_secondary else None
        _render_plotly_figure_on_axes(source_figure, axis, secondary_axis)
        axis.set_title(panel_titles[idx], fontsize=11, pad=10)

        handles, labels = axis.get_legend_handles_labels()
        if secondary_axis is not None:
            secondary_handles, secondary_labels = secondary_axis.get_legend_handles_labels()
            handles += secondary_handles
            labels += secondary_labels
        if handles:
            axis.legend(
                handles,
                labels,
                fontsize=8.3,
                frameon=True,
                facecolor=(1.0, 1.0, 1.0, 0.78),
                edgecolor=(0.0, 0.0, 0.0, 0.18),
                loc="lower right",
            )

    fig.subplots_adjust(top=0.78, wspace=0.28)
    return fig


def build_matplotlib_single_panel_figure(plotly_figure, title):
    fig, axis = plt.subplots(1, 1, figsize=(7.2, 5.0), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=14, fontweight="semibold", y=0.98)

    has_secondary = any(getattr(trace, "yaxis", None) == "y2" for trace in plotly_figure.data)
    secondary_axis = axis.twinx() if has_secondary else None
    _render_plotly_figure_on_axes(plotly_figure, axis, secondary_axis)

    handles, labels = axis.get_legend_handles_labels()
    if secondary_axis is not None:
        secondary_handles, secondary_labels = secondary_axis.get_legend_handles_labels()
        handles += secondary_handles
        labels += secondary_labels
    if handles:
        axis.legend(
            handles,
            labels,
            fontsize=8.3,
            frameon=True,
            facecolor=(1.0, 1.0, 1.0, 0.78),
            edgecolor=(0.0, 0.0, 0.0, 0.18),
            loc="lower right",
        )

    fig.subplots_adjust(top=0.82)
    return fig


def build_trajectory_projection_figure(trajectory_specs, title):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=14, fontweight="semibold", y=0.98)

    xz_axis, yz_axis = axes
    all_x = []
    all_y = []
    all_z = []

    for spec in trajectory_specs:
        states = np.asarray(spec["states"], dtype=float)
        x_values = states[0, :]
        y_values = states[1, :]
        z_values = states[2, :]
        all_x.append(x_values)
        all_y.append(y_values)
        all_z.append(z_values)

        xz_axis.plot(
            x_values,
            z_values,
            label=spec["label"],
            color=spec["color"],
            linestyle=spec.get("linestyle", "-"),
            linewidth=spec.get("linewidth", 2.5),
        )
        yz_axis.plot(
            y_values,
            z_values,
            label=spec["label"],
            color=spec["color"],
            linestyle=spec.get("linestyle", "-"),
            linewidth=spec.get("linewidth", 2.5),
        )

    x_limits = np.concatenate(all_x)
    y_limits = np.concatenate(all_y)
    z_limits = np.concatenate(all_z)

    x_pad = 0.05 * max(np.ptp(x_limits), 1.0e-6)
    y_pad = 0.05 * max(np.ptp(y_limits), 1.0e-6)
    z_pad = 0.08 * max(np.ptp(z_limits), 1.0e-6)

    xz_axis.set_title("x-z Projection", fontsize=11, pad=10)
    xz_axis.set_xlabel("x [LU]")
    xz_axis.set_ylabel("z [LU]")
    xz_axis.set_xlim(float(np.min(x_limits) - x_pad), float(np.max(x_limits) + x_pad))
    xz_axis.set_ylim(float(np.min(z_limits) - z_pad), float(np.max(z_limits) + z_pad))
    _style_matplotlib_axis(xz_axis)

    yz_axis.set_title("y-z Projection", fontsize=11, pad=10)
    yz_axis.set_xlabel("y [LU]")
    yz_axis.set_ylabel("z [LU]")
    yz_axis.set_xlim(float(np.min(y_limits) - y_pad), float(np.max(y_limits) + y_pad))
    yz_axis.set_ylim(float(np.min(z_limits) - z_pad), float(np.max(z_limits) + z_pad))
    _style_matplotlib_axis(yz_axis)

    legend = yz_axis.legend(
        fontsize=8.3,
        frameon=True,
        facecolor=(1.0, 1.0, 1.0, 0.78),
        edgecolor=(0.0, 0.0, 0.0, 0.18),
        loc="lower right",
    )
    legend.set_zorder(10)
    fig.subplots_adjust(top=0.78, wspace=0.26)
    return fig


def _slugify_filename(label):
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", label.strip().lower()).strip("_")
    return cleaned or "figure"


def save_figure_png(figure, label, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_slugify_filename(label)}.png"

    if isinstance(figure, MatplotlibFigure):
        figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        return output_path

    figure.write_image(str(output_path), format="png", scale=2)
    return output_path


def write_csv_table(output_path, fieldnames, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_trajectory_style(name, color, dash="solid", width=3):
    return {
        "trajectory": {
            "name": name,
            "mode": "lines",
            "line": {"color": color, "dash": dash, "width": width},
        }
    }


def build_error_style(color, dash="solid", width=3):
    return {
        "error": {
            "mode": "lines",
            "line": {"color": color, "dash": dash, "width": width},
        }
    }


def build_jacobi_style(color, dash="solid", width=3):
    return {
        "jacobi": {
            "mode": "lines",
            "line": {"color": color, "dash": dash, "width": width},
        }
    }


def station_history_to_result(history):
    return PropagationResult(
        t=np.asarray(history.times, dtype=float),
        states=np.column_stack(history.states),
    )


def build_initial_disturbance(reference_state, rng):
    disturbed_state = np.asarray(reference_state, dtype=float).copy()
    disturbed_state[:3] += rng.normal(0.0, 2.0e-5, size=3)
    disturbed_state[3:6] += rng.normal(0.0, 5.0e-6, size=3)
    return disturbed_state


def compute_position_error_stats(trajectory_result, reference):
    sample_times = np.asarray(trajectory_result.t, dtype=float)
    propagated_positions = np.asarray(trajectory_result.states[:3], dtype=float)
    sampler = build_reference_sampler(reference)
    reference_positions = np.column_stack(
        [reference_state_at_time(reference, sampler, time)[:3] for time in sample_times]
    ).T
    position_error = propagated_positions.T - reference_positions
    error_norm = np.linalg.norm(position_error, axis=1)
    return {
        "final_position_error": float(error_norm[-1]),
        "max_position_error": float(np.max(error_norm)),
        "rms_position_error": float(np.sqrt(np.mean(error_norm**2))),
    }


def build_combined_position_phase_figure(
    reference_trajectory,
    trajectory_dict,
    normalization_period,
    position_format_dict=None,
    phase_colors=None,
    title="",
):
    fig = plot_position_error_magnitude(
        reference_trajectory,
        trajectory_dict,
        normalization_period=normalization_period,
        format_dict=position_format_dict,
        length_scale=EARTH_MOON_LENGTH_UNIT_KM,
        length_unit_label="km",
    )

    phase_fig = plot_phase_error(
        reference_trajectory,
        trajectory_dict,
        normalization_period=normalization_period,
    )

    phase_colors = phase_colors or {}
    for trace in phase_fig.data:
        trace_dict = trace.to_plotly_json()
        base_name = trace_dict.get("name", "Phase")
        trace_dict["name"] = f"{base_name} Phase"
        trace_dict["yaxis"] = "y2"
        if "line" not in trace_dict:
            trace_dict["line"] = {}
        trace_dict["line"]["dash"] = "dot"
        trace_dict["line"]["width"] = 2
        if base_name in phase_colors:
            trace_dict["line"]["color"] = phase_colors[base_name]
        fig.add_trace(go.Scatter(**trace_dict))

    fig.update_layout(
        title=title,
        yaxis={"title": "Position Error Magnitude [km]"},
        yaxis2={
            "title": "Phase Error [TU]",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
    )
    return fig


def choose_stationkeeping_solution(trade_space, uncontrolled_stats, initial_position_error):
    acceptable_final_error = min(
        0.25 * uncontrolled_stats["final_position_error"],
        2.0 * initial_position_error,
    )
    acceptable_max_error = min(
        0.25 * uncontrolled_stats["max_position_error"],
        5.0 * initial_position_error,
    )

    feasible_entries = [
        entry
        for entry in trade_space
        if entry["final_position_error"] <= acceptable_final_error
        and entry["max_position_error"] <= acceptable_max_error
    ]

    if feasible_entries:
        selected_entry = min(feasible_entries, key=lambda entry: entry["total_delta_v"])
    else:
        selected_entry = min(
            trade_space,
            key=lambda entry: (
                entry["final_position_error"] / max(acceptable_final_error, 1.0e-12),
                entry["max_position_error"] / max(acceptable_max_error, 1.0e-12),
                entry["total_delta_v"],
            ),
        )

    return selected_entry, acceptable_final_error, acceptable_max_error


def evaluate_trade_space(reference, propagator, initial_state, station_reference, station_duration, full_period):
    correction_intervals = full_period * np.linspace(
        SWEEP_MIN_INTERVAL,
        SWEEP_MAX_INTERVAL,
        SWEEP_INTERVAL_COUNT,
    )
    trade_space = sweep_correction_intervals(
        reference=reference,
        propagator=propagator,
        initial_state=initial_state,
        correction_intervals=correction_intervals,
        control_offset_fraction=CONTROL_OFFSET_FRACTION,
        max_delta_v_fraction=MAX_DELTA_V_FRACTION,
        duration=station_duration,
        max_phi_rv_condition_number=MAX_PHI_RV_CONDITION_NUMBER,
    )

    for entry in trade_space:
        history_result = station_history_to_result(entry["history"])
        entry["result"] = history_result
        entry.update(compute_position_error_stats(history_result, station_reference))

    return trade_space


def load_branch_case(branch, three_body_prop, four_body_prop):
    api_call = query_earth_moon_halo(
        libration_point=2,
        branch=branch["code"],
        stabmin=5,
        stabmax=6,
    )
    api_orbit = api_call.get_orbit_by_index(0)

    diff_corr = DifferentialCorrector(propagator=three_body_prop)
    corrected_orbit = diff_corr.solve(
        api_orbit,
        tf_propagation=api_orbit.period * REFERENCE_PERIODS,
        orbit_tf=api_orbit.period * REFERENCE_PERIODS,
    )

    full_reference = build_full_period_reference(
        propagator=three_body_prop,
        corrected_reference=corrected_orbit,
        samples_per_period=SAMPLES_PER_PERIOD_REFERENCE,
    )
    looped_reference = loop_reference_trajectory(
        full_period_reference=full_reference,
        num_periods=REFERENCE_PERIODS,
    )

    reference_duration = full_reference.period * REFERENCE_PERIODS
    n_eval_reference = SAMPLES_PER_PERIOD_PROPAGATION * REFERENCE_PERIODS

    corrected_cr3bp = three_body_prop.propagate(
        corrected_orbit.initial_state,
        reference_duration,
        n_eval=n_eval_reference,
    )
    corrected_br4bp = four_body_prop.propagate(
        corrected_orbit.initial_state,
        reference_duration,
        n_eval=n_eval_reference,
    )

    return {
        "branch": branch,
        "api_orbit": api_orbit,
        "corrected_orbit": corrected_orbit,
        "full_reference": full_reference,
        "looped_reference": looped_reference,
        "corrected_cr3bp": corrected_cr3bp,
        "corrected_br4bp": corrected_br4bp,
    }


def build_baseline_figures(case):
    branch_name = case["branch"]["name"]
    figures = []

    fig_traj = build_trajectory_projection_figure(
        [
            {
                "states": case["looped_reference"].states,
                "label": f"{branch_name} L2 Reference Halo",
                "color": MODEL_STYLES["Reference"]["color"],
                "linestyle": "--",
                "linewidth": 3.0,
            },
            {
                "states": case["corrected_cr3bp"].states,
                "label": "Corrected State in CR3BP",
                "color": MODEL_STYLES["CR3BP"]["color"],
                "linestyle": "-",
                "linewidth": 2.5,
            },
            {
                "states": case["corrected_br4bp"].states,
                "label": "Corrected State in BR4BP",
                "color": MODEL_STYLES["BR4BP"]["color"],
                "linestyle": "-",
                "linewidth": 2.5,
            },
        ],
        f"{branch_name} L2 Halo Orbit Reference, CR3BP, and BR4BP Projections",
    )
    figures.append(
        (
            f"{branch_name} L2 Halo Orbit Reference CR3BP and BR4BP Projections",
            fig_traj,
        )
    )

    fig_error = build_combined_position_phase_figure(
        case["looped_reference"],
        {
            "Corrected State in CR3BP": case["corrected_cr3bp"],
            "Corrected State in BR4BP": case["corrected_br4bp"],
        },
        normalization_period=case["full_reference"].period,
        position_format_dict=[
            build_error_style(MODEL_STYLES["CR3BP"]["color"], width=3),
            build_error_style(MODEL_STYLES["BR4BP"]["color"], width=3),
        ],
        phase_colors={
            "Corrected State in CR3BP": MODEL_STYLES["CR3BP"]["color"],
            "Corrected State in BR4BP": MODEL_STYLES["BR4BP"]["color"],
        },
    )
    apply_report_layout(
        fig_error,
        f"{branch_name} L2 Halo Orbit Position and Phase Error Relative to the CR3BP Reference",
    )
    figs_2d = [fig_error]

    fig_jacobi = plot_jacobi(
        {
            "Corrected State in CR3BP": case["corrected_cr3bp"],
            "Corrected State in BR4BP": case["corrected_br4bp"],
        },
        normalization_period=case["full_reference"].period,
        format_dict=[
            build_jacobi_style(MODEL_STYLES["CR3BP"]["color"], width=3),
            build_jacobi_style(MODEL_STYLES["BR4BP"]["color"], width=3),
        ],
    )
    apply_report_layout(
        fig_jacobi,
        f"{branch_name} L2 Halo Orbit Jacobi Constant Evolution",
    )
    figs_2d.append(fig_jacobi)

    combined_2d = build_matplotlib_two_panel_figure(
        figs_2d,
        f"{branch_name} L2 Halo Orbit Error and Jacobi Summary",
        ["Position and Phase Error", "Jacobi Constant"],
    )
    figures.append((f"{branch_name} L2 Halo Orbit Error and Jacobi Summary", combined_2d))
    return figures


def analyze_stationkeeping(case, three_body_prop, four_body_prop, rng):
    full_reference = case["full_reference"]
    looped_reference = case["looped_reference"]
    station_duration = full_reference.period * STATION_KEEPING_PERIODS
    station_reference = loop_reference_trajectory(
        full_period_reference=full_reference,
        num_periods=STATION_KEEPING_PERIODS,
    )
    disturbed_initial_state = build_initial_disturbance(case["corrected_orbit"].initial_state, rng)
    initial_position_error = float(
        np.linalg.norm(disturbed_initial_state[:3] - case["corrected_orbit"].initial_state[:3])
    )

    uncontrolled_cr3bp = three_body_prop.propagate(
        disturbed_initial_state,
        station_duration,
        n_eval=SAMPLES_PER_PERIOD_PROPAGATION * STATION_KEEPING_PERIODS,
    )
    uncontrolled_br4bp = four_body_prop.propagate(
        disturbed_initial_state,
        station_duration,
        n_eval=SAMPLES_PER_PERIOD_PROPAGATION * STATION_KEEPING_PERIODS,
    )

    trade_cr3bp = evaluate_trade_space(
        reference=looped_reference,
        propagator=three_body_prop,
        initial_state=disturbed_initial_state.copy(),
        station_reference=station_reference,
        station_duration=station_duration,
        full_period=full_reference.period,
    )
    trade_br4bp = evaluate_trade_space(
        reference=looped_reference,
        propagator=four_body_prop,
        initial_state=disturbed_initial_state.copy(),
        station_reference=station_reference,
        station_duration=station_duration,
        full_period=full_reference.period,
    )

    uncontrolled_stats_cr3bp = compute_position_error_stats(uncontrolled_cr3bp, station_reference)
    uncontrolled_stats_br4bp = compute_position_error_stats(uncontrolled_br4bp, station_reference)

    selected_cr3bp, allowable_final_cr3bp, allowable_max_cr3bp = choose_stationkeeping_solution(
        trade_cr3bp,
        uncontrolled_stats_cr3bp,
        initial_position_error,
    )
    selected_br4bp, allowable_final_br4bp, allowable_max_br4bp = choose_stationkeeping_solution(
        trade_br4bp,
        uncontrolled_stats_br4bp,
        initial_position_error,
    )

    station_problem_br4bp = StationKeepingProblem(
        reference=looped_reference,
        correction_interval=float(selected_br4bp["correction_interval"]),
        duration=station_duration,
        max_delta_v_fraction=MAX_DELTA_V_FRACTION,
        max_phi_rv_condition_number=MAX_PHI_RV_CONDITION_NUMBER,
        metadata={"study": f"{case['branch']['name']}_BR4BP_stationkeeping"},
    )
    station_history_br4bp = run_station_keeping(
        problem=station_problem_br4bp,
        propagator=four_body_prop,
        initial_state=disturbed_initial_state.copy(),
        control_offset=CONTROL_OFFSET_FRACTION * float(selected_br4bp["correction_interval"]),
    )
    station_kept_br4bp = station_history_to_result(station_history_br4bp)
    station_kept_stats_br4bp = compute_position_error_stats(station_kept_br4bp, station_reference)

    return {
        "station_reference": station_reference,
        "disturbed_initial_state": disturbed_initial_state,
        "uncontrolled_cr3bp": uncontrolled_cr3bp,
        "uncontrolled_br4bp": uncontrolled_br4bp,
        "uncontrolled_stats_cr3bp": uncontrolled_stats_cr3bp,
        "uncontrolled_stats_br4bp": uncontrolled_stats_br4bp,
        "trade_cr3bp": trade_cr3bp,
        "trade_br4bp": trade_br4bp,
        "selected_cr3bp": selected_cr3bp,
        "selected_br4bp": selected_br4bp,
        "allowable_final_cr3bp": allowable_final_cr3bp,
        "allowable_final_br4bp": allowable_final_br4bp,
        "allowable_max_cr3bp": allowable_max_cr3bp,
        "allowable_max_br4bp": allowable_max_br4bp,
        "station_history_br4bp": station_history_br4bp,
        "station_kept_br4bp": station_kept_br4bp,
        "station_kept_stats_br4bp": station_kept_stats_br4bp,
    }


def build_trade_summary_figure(branch_cases, stationkeeping_results):
    fig, axis = plt.subplots(1, 1, figsize=(7.8, 5.4), constrained_layout=False)
    fig.patch.set_facecolor("white")

    for branch in BRANCHES:
        branch_key = branch["code"]
        case = branch_cases[branch_key]
        sk = stationkeeping_results[branch_key]
        period = case["full_reference"].period
        periods_per_run = sk["station_reference"].t[-1] / period

        interval_cr3bp = np.array(
            [entry["correction_interval"] / period for entry in sk["trade_cr3bp"]],
            dtype=float,
        )
        interval_br4bp = np.array(
            [entry["correction_interval"] / period for entry in sk["trade_br4bp"]],
            dtype=float,
        )
        dv_cr3bp = np.array(
            [(entry["total_delta_v"] / periods_per_run) * EARTH_MOON_VELOCITY_UNIT_MPS for entry in sk["trade_cr3bp"]],
            dtype=float,
        )
        dv_br4bp = np.array(
            [(entry["total_delta_v"] / periods_per_run) * EARTH_MOON_VELOCITY_UNIT_MPS for entry in sk["trade_br4bp"]],
            dtype=float,
        )

        axis.plot(
            interval_cr3bp,
            dv_cr3bp,
            label=f"{branch['name']} CR3BP",
            color=branch["color"],
            linewidth=2.5,
            linestyle="-",
            marker="o",
            markersize=5,
        )
        axis.plot(
            interval_br4bp,
            dv_br4bp,
            label=f"{branch['name']} BR4BP",
            color=branch["color"],
            linewidth=2.5,
            linestyle="--",
            marker="s",
            markersize=5,
        )
        axis.scatter(
            [sk["selected_cr3bp"]["correction_interval"] / period],
            [(sk["selected_cr3bp"]["total_delta_v"] / periods_per_run) * EARTH_MOON_VELOCITY_UNIT_MPS],
            color="black",
            marker="D",
            s=42,
            zorder=5,
        )
        axis.scatter(
            [sk["selected_br4bp"]["correction_interval"] / period],
            [(sk["selected_br4bp"]["total_delta_v"] / periods_per_run) * EARTH_MOON_VELOCITY_UNIT_MPS],
            color="black",
            marker="x",
            s=52,
            linewidths=1.8,
            zorder=5,
        )

    axis.set_xlabel("Correction Interval / Reference Period [-]")
    axis.set_ylabel("Average |Δv| per Period [m/s]")
    axis.set_title("Station-Keeping Cost vs. Correction Interval", fontsize=12, pad=10)
    _style_matplotlib_axis(axis)
    axis.legend(
        fontsize=8.3,
        frameon=True,
        facecolor=(1.0, 1.0, 1.0, 0.78),
        edgecolor=(0.0, 0.0, 0.0, 0.18),
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    fig.suptitle(
        "Station-Keeping Cost Comparison for Northern and Southern L2 Halo Orbits",
        fontsize=14,
        fontweight="semibold",
        y=0.98,
    )
    fig.subplots_adjust(top=0.82, bottom=0.24)
    return fig


def build_tracking_summary_figure(branch_cases, stationkeeping_results):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0), constrained_layout=True, sharey=True)
    fig.patch.set_facecolor("white")

    for axis, branch in zip(axes, BRANCHES):
        branch_key = branch["code"]
        case = branch_cases[branch_key]
        sk = stationkeeping_results[branch_key]
        period = case["full_reference"].period

        interval_cr3bp = np.array(
            [entry["correction_interval"] / period for entry in sk["trade_cr3bp"]],
            dtype=float,
        )
        interval_br4bp = np.array(
            [entry["correction_interval"] / period for entry in sk["trade_br4bp"]],
            dtype=float,
        )
        final_error_cr3bp = np.array(
            [entry["final_position_error"] * EARTH_MOON_LENGTH_UNIT_KM for entry in sk["trade_cr3bp"]],
            dtype=float,
        )
        final_error_br4bp = np.array(
            [entry["final_position_error"] * EARTH_MOON_LENGTH_UNIT_KM for entry in sk["trade_br4bp"]],
            dtype=float,
        )

        axis.plot(
            interval_cr3bp,
            final_error_cr3bp,
            label="CR3BP",
            color=branch["color"],
            linewidth=2.5,
            linestyle="-",
            marker="o",
            markersize=5,
        )
        axis.plot(
            interval_br4bp,
            final_error_br4bp,
            label="BR4BP",
            color="#222222",
            linewidth=2.5,
            linestyle="--",
            marker="s",
            markersize=5,
        )
        axis.axhline(
            sk["allowable_final_br4bp"] * EARTH_MOON_LENGTH_UNIT_KM,
            color="#555555",
            linewidth=1.6,
            linestyle=":",
            label="Acceptance Threshold",
        )
        axis.set_title(f"{branch['name']} Branch", fontsize=11, pad=10)
        axis.set_xlabel("Correction Interval / Reference Period [-]")
        if axis is axes[0]:
            axis.set_ylabel("Final Position Error [km]")
        _style_matplotlib_axis(axis)
        axis.legend(
            fontsize=8.5,
            frameon=True,
            facecolor=(1.0, 1.0, 1.0, 0.78),
            edgecolor=(0.0, 0.0, 0.0, 0.18),
            loc="upper left",
        )

    fig.suptitle(
        "Final Position Error vs. Correction Interval",
        fontsize=14,
        fontweight="semibold",
        y=1.02,
    )
    return fig


def build_model_comparison_bar_figure(branch_cases, stationkeeping_results):
    labels = []
    cr3bp_costs = []
    br4bp_costs = []

    for branch in BRANCHES:
        branch_key = branch["code"]
        case = branch_cases[branch_key]
        sk = stationkeeping_results[branch_key]
        periods_per_run = sk["station_reference"].t[-1] / case["full_reference"].period
        labels.append(branch["name"])
        cr3bp_costs.append(
            (sk["selected_cr3bp"]["total_delta_v"] / periods_per_run) * EARTH_MOON_VELOCITY_UNIT_MPS
        )
        br4bp_costs.append(
            (sk["selected_br4bp"]["total_delta_v"] / periods_per_run) * EARTH_MOON_VELOCITY_UNIT_MPS
        )

    x_positions = np.arange(len(labels), dtype=float)
    width = 0.36

    fig, axis = plt.subplots(1, 1, figsize=(7.0, 4.8), constrained_layout=False)
    fig.patch.set_facecolor("white")
    axis.bar(
        x_positions - width / 2.0,
        cr3bp_costs,
        width=width,
        label="CR3BP",
        color=MODEL_STYLES["CR3BP"]["color"],
    )
    axis.bar(
        x_positions + width / 2.0,
        br4bp_costs,
        width=width,
        label="BR4BP",
        color=MODEL_STYLES["BR4BP"]["color"],
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels)
    axis.set_xlabel("Halo Orbit Branch")
    axis.set_ylabel("Selected Average |Δv| per Period [m/s]")
    _style_matplotlib_axis(axis)
    axis.legend(
        fontsize=8.5,
        frameon=True,
        facecolor=(1.0, 1.0, 1.0, 0.78),
        edgecolor=(0.0, 0.0, 0.0, 0.18),
        loc="upper left",
    )
    fig.suptitle(
        "Station-Keeping Cost Comparison Summary",
        fontsize=14,
        fontweight="semibold",
        y=0.97,
    )
    fig.subplots_adjust(top=0.82)
    return fig


def build_stationkeeping_result_figures(branch_case, sk_result):
    branch_name = branch_case["branch"]["name"]

    figures = []
    fig_traj = build_trajectory_projection_figure(
        [
            {
                "states": sk_result["station_reference"].states,
                "label": f"{branch_name} Reference Halo",
                "color": MODEL_STYLES["Reference"]["color"],
                "linestyle": "--",
                "linewidth": 3.0,
            },
            {
                "states": sk_result["uncontrolled_br4bp"].states,
                "label": "Uncontrolled BR4BP",
                "color": MODEL_STYLES["Uncontrolled"]["color"],
                "linestyle": ":",
                "linewidth": 2.5,
            },
            {
                "states": sk_result["station_kept_br4bp"].states,
                "label": "Station-Kept BR4BP",
                "color": MODEL_STYLES["StationKept"]["color"],
                "linestyle": "-",
                "linewidth": 3.0,
            },
        ],
        f"{branch_name} L2 Halo Orbit Station-Keeping Projections in BR4BP",
    )
    figures.append((f"{branch_name} L2 Halo Orbit Station Keeping Projections in BR4BP", fig_traj))

    fig_error = build_combined_position_phase_figure(
        sk_result["station_reference"],
        {
            "Uncontrolled BR4BP": sk_result["uncontrolled_br4bp"],
            "Station-Kept BR4BP": sk_result["station_kept_br4bp"],
        },
        normalization_period=branch_case["full_reference"].period,
        position_format_dict=[
            build_error_style(MODEL_STYLES["Uncontrolled"]["color"], dash="dot", width=3),
            build_error_style(MODEL_STYLES["StationKept"]["color"], width=4),
        ],
        phase_colors={
            "Uncontrolled BR4BP": MODEL_STYLES["Uncontrolled"]["color"],
            "Station-Kept BR4BP": MODEL_STYLES["StationKept"]["color"],
        },
    )
    apply_report_layout(
        fig_error,
        f"{branch_name} L2 Halo Orbit Station-Keeping Position and Phase Error in BR4BP",
    )
    figs_2d = [fig_error]

    correction_times_periods = np.array(
        [record.time / branch_case["full_reference"].period for record in sk_result["station_history_br4bp"].corrections],
        dtype=float,
    )
    cumulative_delta_v_mps = np.cumsum(
        [
            np.linalg.norm(record.delta_v) * EARTH_MOON_VELOCITY_UNIT_MPS
            for record in sk_result["station_history_br4bp"].corrections
        ]
    )
    fig_cumulative = go.Figure()
    fig_cumulative.add_trace(
        go.Scatter(
            x=correction_times_periods,
            y=cumulative_delta_v_mps,
            mode="lines+markers",
            name="Cumulative |Δv|",
            line={"color": MODEL_STYLES["StationKept"]["color"], "width": 3},
            marker={"size": 7, "color": MODEL_STYLES["StationKept"]["color"]},
        )
    )
    fig_cumulative.update_layout(
        xaxis={"title": "Time / Reference Period [-]"},
        yaxis={"title": "Cumulative |Δv| [m/s]"},
    )
    apply_report_layout(
        fig_cumulative,
        f"{branch_name} L2 Halo Orbit Cumulative Station-Keeping Cost in BR4BP",
    )
    figs_2d.append(fig_cumulative)
    combined_2d = build_matplotlib_two_panel_figure(
        figs_2d,
        f"{branch_name} L2 Halo Orbit Station-Keeping Performance Summary",
        ["Position and Phase Error", "Cumulative Station-Keeping Cost"],
    )
    figures.append((f"{branch_name} L2 Halo Orbit Station Keeping Performance Summary", combined_2d))
    return figures


def export_analysis_tables(branch_cases, stationkeeping_results, output_dir):
    summary_rows = []
    trade_rows = []

    for branch in BRANCHES:
        branch_key = branch["code"]
        case = branch_cases[branch_key]
        sk = stationkeeping_results[branch_key]
        full_period = float(case["full_reference"].period)
        run_periods = float(sk["station_reference"].t[-1] / full_period)

        summary_rows.append(
            {
                "branch": branch["name"],
                "branch_code": branch["code"],
                "api_period_tu": float(case["api_orbit"].period),
                "corrected_period_tu": full_period,
                "initial_position_error_km": float(
                    np.linalg.norm(sk["disturbed_initial_state"][:3] - case["corrected_orbit"].initial_state[:3])
                    * EARTH_MOON_LENGTH_UNIT_KM
                ),
                "uncontrolled_cr3bp_final_error_km": sk["uncontrolled_stats_cr3bp"]["final_position_error"]
                * EARTH_MOON_LENGTH_UNIT_KM,
                "uncontrolled_br4bp_final_error_km": sk["uncontrolled_stats_br4bp"]["final_position_error"]
                * EARTH_MOON_LENGTH_UNIT_KM,
                "selected_cr3bp_interval_fraction": float(sk["selected_cr3bp"]["correction_interval"] / full_period),
                "selected_br4bp_interval_fraction": float(sk["selected_br4bp"]["correction_interval"] / full_period),
                "selected_cr3bp_avg_dv_per_period_mps": float(
                    (sk["selected_cr3bp"]["total_delta_v"] / run_periods) * EARTH_MOON_VELOCITY_UNIT_MPS
                ),
                "selected_br4bp_avg_dv_per_period_mps": float(
                    (sk["selected_br4bp"]["total_delta_v"] / run_periods) * EARTH_MOON_VELOCITY_UNIT_MPS
                ),
                "selected_cr3bp_final_error_km": float(
                    sk["selected_cr3bp"]["final_position_error"] * EARTH_MOON_LENGTH_UNIT_KM
                ),
                "selected_br4bp_final_error_km": float(
                    sk["selected_br4bp"]["final_position_error"] * EARTH_MOON_LENGTH_UNIT_KM
                ),
                "station_kept_br4bp_final_error_km": float(
                    sk["station_kept_stats_br4bp"]["final_position_error"] * EARTH_MOON_LENGTH_UNIT_KM
                ),
                "station_kept_br4bp_max_error_km": float(
                    sk["station_kept_stats_br4bp"]["max_position_error"] * EARTH_MOON_LENGTH_UNIT_KM
                ),
                "station_kept_br4bp_rms_error_km": float(
                    sk["station_kept_stats_br4bp"]["rms_position_error"] * EARTH_MOON_LENGTH_UNIT_KM
                ),
                "allowable_br4bp_final_error_km": float(sk["allowable_final_br4bp"] * EARTH_MOON_LENGTH_UNIT_KM),
            }
        )

        for model_name, trade_entries, selected_entry in (
            ("CR3BP", sk["trade_cr3bp"], sk["selected_cr3bp"]),
            ("BR4BP", sk["trade_br4bp"], sk["selected_br4bp"]),
        ):
            for entry in trade_entries:
                trade_rows.append(
                    {
                        "branch": branch["name"],
                        "branch_code": branch["code"],
                        "model": model_name,
                        "correction_interval_tu": float(entry["correction_interval"]),
                        "correction_interval_fraction": float(entry["correction_interval"] / full_period),
                        "total_delta_v_lu_per_tu": float(entry["total_delta_v"]),
                        "average_delta_v_per_period_mps": float(
                            (entry["total_delta_v"] / run_periods) * EARTH_MOON_VELOCITY_UNIT_MPS
                        ),
                        "final_position_error_km": float(
                            entry["final_position_error"] * EARTH_MOON_LENGTH_UNIT_KM
                        ),
                        "max_position_error_km": float(
                            entry["max_position_error"] * EARTH_MOON_LENGTH_UNIT_KM
                        ),
                        "rms_position_error_km": float(
                            entry["rms_position_error"] * EARTH_MOON_LENGTH_UNIT_KM
                        ),
                        "selected": bool(entry is selected_entry),
                    }
                )

        correction_rows = []
        for correction_index, record in enumerate(sk["station_history_br4bp"].corrections, start=1):
            delta_v = np.asarray(record.delta_v, dtype=float)
            correction_rows.append(
                {
                    "correction_index": correction_index,
                    "time_tu": float(record.time),
                    "time_periods": float(record.time / full_period),
                    "delta_vx_mps": float(delta_v[0] * EARTH_MOON_VELOCITY_UNIT_MPS),
                    "delta_vy_mps": float(delta_v[1] * EARTH_MOON_VELOCITY_UNIT_MPS),
                    "delta_vz_mps": float(delta_v[2] * EARTH_MOON_VELOCITY_UNIT_MPS),
                    "delta_v_norm_mps": float(np.linalg.norm(delta_v) * EARTH_MOON_VELOCITY_UNIT_MPS),
                }
            )

        write_csv_table(
            output_dir / f"{branch['code'].lower()}_br4bp_stationkeeping_corrections.csv",
            [
                "correction_index",
                "time_tu",
                "time_periods",
                "delta_vx_mps",
                "delta_vy_mps",
                "delta_vz_mps",
                "delta_v_norm_mps",
            ],
            correction_rows,
        )

    write_csv_table(
        output_dir / "halo_stationkeeping_summary.csv",
        [
            "branch",
            "branch_code",
            "api_period_tu",
            "corrected_period_tu",
            "initial_position_error_km",
            "uncontrolled_cr3bp_final_error_km",
            "uncontrolled_br4bp_final_error_km",
            "selected_cr3bp_interval_fraction",
            "selected_br4bp_interval_fraction",
            "selected_cr3bp_avg_dv_per_period_mps",
            "selected_br4bp_avg_dv_per_period_mps",
            "selected_cr3bp_final_error_km",
            "selected_br4bp_final_error_km",
            "station_kept_br4bp_final_error_km",
            "station_kept_br4bp_max_error_km",
            "station_kept_br4bp_rms_error_km",
            "allowable_br4bp_final_error_km",
        ],
        summary_rows,
    )

    write_csv_table(
        output_dir / "halo_stationkeeping_trade_study.csv",
        [
            "branch",
            "branch_code",
            "model",
            "correction_interval_tu",
            "correction_interval_fraction",
            "total_delta_v_lu_per_tu",
            "average_delta_v_per_period_mps",
            "final_position_error_km",
            "max_position_error_km",
            "rms_position_error_km",
            "selected",
        ],
        trade_rows,
    )


def main():
    rng = np.random.default_rng(RNG_SEED)

    three_body_model = CR3BP(mu=EARTH_MOON_MU)
    sun = create_solar_perturbation()
    four_body_model = CR3BP(mu=EARTH_MOON_MU, perturbing_body=sun)

    three_body_prop = Propagator(model=three_body_model)
    four_body_prop = Propagator(model=four_body_model)

    branch_cases = {}
    stationkeeping_results = {}
    figures = []

    for branch in BRANCHES:
        branch_case = load_branch_case(branch, three_body_prop, four_body_prop)
        branch_cases[branch["code"]] = branch_case
        figures.extend(build_baseline_figures(branch_case))

        branch_rng = np.random.default_rng(RNG_SEED + (0 if branch["code"] == "N" else 100))
        stationkeeping_result = analyze_stationkeeping(
            branch_case,
            three_body_prop,
            four_body_prop,
            branch_rng,
        )
        stationkeeping_results[branch["code"]] = stationkeeping_result
        figures.extend(build_stationkeeping_result_figures(branch_case, stationkeeping_result))

    summary_trade = build_trade_summary_figure(branch_cases, stationkeeping_results)
    summary_tracking = build_tracking_summary_figure(branch_cases, stationkeeping_results)
    summary_bar = build_model_comparison_bar_figure(branch_cases, stationkeeping_results)
    figures.append(
        (
            "Station Keeping Cost Comparison for Northern and Southern L2 Halo Orbits",
            summary_trade,
        )
    )
    figures.append(
        (
            "Final Position Error Versus Correction Interval for Northern and Southern L2 Halo Orbits",
            summary_tracking,
        )
    )
    figures.append(
        (
            "Station Keeping Cost Comparison Summary",
            summary_bar,
        )
    )

    export_analysis_tables(branch_cases, stationkeeping_results, TABLE_OUTPUT_DIR)

    for label, figure in figures:
        save_figure_png(figure, label, FIGURE_OUTPUT_DIR)
        if isinstance(figure, MatplotlibFigure):
            figure.show()
        else:
            show_figure(figure)


if __name__ == "__main__":
    main()
