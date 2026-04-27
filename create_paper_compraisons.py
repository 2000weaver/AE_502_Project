"""
Reproduce the page 6-8 perturbation plots from
"Comparisons between the circular.pdf" using the current perturbed CR3BP model.

This script generates three figures:
1. Solar perturbation magnitude in the x-y plane (z = 0)
2. Solar perturbation magnitude in the x-z plane (y = 0)
3. Ratio |p_s| / |p_m| in the x-y plane (z = 0)

The solar perturbation vector p_s is obtained from the current implementation by
subtracting the unperturbed CR3BP acceleration from the perturbed CR3BP
acceleration at each grid point. That makes this script a direct validation tool
for the dynamics currently implemented in the package.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib.patheffects as path_effects

from astrokit.models import CR3BP
from astrokit.models.perturbing_body import create_solar_perturbation
from astrokit.utils.constants import (
    EARTH_MOON_ACCELERATION_UNIT_MPS2,
    EARTH_MOON_DISTANCE_M,
    EARTH_MOON_LENGTH_UNIT_M,
    EARTH_MOON_MU,
    EARTH_MOON_TIME_UNIT_SECONDS,
)


DU_M = EARTH_MOON_LENGTH_UNIT_M
TU_S = EARTH_MOON_TIME_UNIT_SECONDS
ACCEL_SCALE = EARTH_MOON_ACCELERATION_UNIT_MPS2

EARTH_RADIUS_M = 6_378_136.3
MOON_RADIUS_M = 1_737_400.0

EARTH_CENTER_M = np.array([-EARTH_MOON_MU * EARTH_MOON_DISTANCE_M, 0.0])
MOON_CENTER_M = np.array([(1.0 - EARTH_MOON_MU) * EARTH_MOON_DISTANCE_M, 0.0])

PHASES_DEG = (0, 45, 90, 135, 180, 270)
PAPER_RATIO_TICKS = np.array([0.03, 0.06, 0.07, 0.08, 0.09, 0.10, 0.13, 0.14, 0.20, 0.30, 0.40])
FIGURE4_RATIO_MAX = 0.5

rcParams["font.family"] = "Times New Roman"
rcParams["axes.titlesize"] = 14
rcParams["axes.labelsize"] = 12
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10
rcParams["legend.fontsize"] = 10
rcParams["figure.titlesize"] = 18


@dataclass(frozen=True)
class PlaneField:
    x_m: np.ndarray
    y_m: np.ndarray
    u: np.ndarray
    v: np.ndarray
    magnitude: np.ndarray


def build_models(theta_deg: float) -> tuple[CR3BP, CR3BP]:
    theta_rad = np.deg2rad(theta_deg)
    sun = create_solar_perturbation(sun_initial_phase=theta_rad)
    perturbed = CR3BP(mu=EARTH_MOON_MU, perturbing_body=sun)
    unperturbed = CR3BP(mu=EARTH_MOON_MU, perturbing_body=None)
    return perturbed, unperturbed


def canonical_to_si_accel(accel: np.ndarray) -> np.ndarray:
    return accel * ACCEL_SCALE


def perturbation_acceleration_si(
    perturbed: CR3BP,
    unperturbed: CR3BP,
    position_canonical: np.ndarray,
) -> np.ndarray:
    state = np.array(
        [position_canonical[0], position_canonical[1], position_canonical[2], 0.0, 0.0, 0.0],
        dtype=float,
    )
    a_pert = perturbed.equations_of_motion(0.0, state)[3:6]
    a_base = unperturbed.equations_of_motion(0.0, state)[3:6]
    return canonical_to_si_accel(a_pert - a_base)


def moon_gravity_acceleration_si(position_canonical: np.ndarray) -> np.ndarray:
    moon_position = np.array([1.0 - EARTH_MOON_MU, 0.0, 0.0], dtype=float)
    rel = moon_position - position_canonical
    distance = np.linalg.norm(rel)
    if distance < 1e-12:
        return np.full(3, np.nan)
    accel = EARTH_MOON_MU * rel / distance**3
    return canonical_to_si_accel(accel)


def compute_xy_plane(theta_deg: float, x_bounds_m: tuple[float, float], y_bounds_m: tuple[float, float], n: int) -> PlaneField:
    perturbed, unperturbed = build_models(theta_deg)
    x_m = np.linspace(*x_bounds_m, n)
    y_m = np.linspace(*y_bounds_m, n)
    xx_m, yy_m = np.meshgrid(x_m, y_m)

    u = np.empty_like(xx_m)
    v = np.empty_like(xx_m)
    mag = np.empty_like(xx_m)

    for idx in np.ndindex(xx_m.shape):
        position_canonical = np.array([xx_m[idx] / DU_M, yy_m[idx] / DU_M, 0.0], dtype=float)
        accel = perturbation_acceleration_si(perturbed, unperturbed, position_canonical)
        u[idx] = accel[0]
        v[idx] = accel[1]
        mag[idx] = np.linalg.norm(accel)

    return PlaneField(x_m=xx_m, y_m=yy_m, u=u, v=v, magnitude=mag)


def compute_xz_plane(theta_deg: float, x_bounds_m: tuple[float, float], z_bounds_m: tuple[float, float], n: int) -> PlaneField:
    perturbed, unperturbed = build_models(theta_deg)
    x_m = np.linspace(*x_bounds_m, n)
    z_m = np.linspace(*z_bounds_m, n)
    xx_m, zz_m = np.meshgrid(x_m, z_m)

    u = np.empty_like(xx_m)
    v = np.empty_like(xx_m)
    mag = np.empty_like(xx_m)

    for idx in np.ndindex(xx_m.shape):
        position_canonical = np.array([xx_m[idx] / DU_M, 0.0, zz_m[idx] / DU_M], dtype=float)
        accel = perturbation_acceleration_si(perturbed, unperturbed, position_canonical)
        u[idx] = accel[0]
        v[idx] = accel[2]
        mag[idx] = np.linalg.norm(accel)

    return PlaneField(x_m=xx_m, y_m=zz_m, u=u, v=v, magnitude=mag)


def compute_ratio_xy(theta_deg: float, x_bounds_m: tuple[float, float], y_bounds_m: tuple[float, float], n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    perturbed, unperturbed = build_models(theta_deg)
    x_m = np.linspace(*x_bounds_m, n)
    y_m = np.linspace(*y_bounds_m, n)
    xx_m, yy_m = np.meshgrid(x_m, y_m)
    ratio = np.empty_like(xx_m)

    for idx in np.ndindex(xx_m.shape):
        position_canonical = np.array([xx_m[idx] / DU_M, yy_m[idx] / DU_M, 0.0], dtype=float)
        p_s = perturbation_acceleration_si(perturbed, unperturbed, position_canonical)
        p_m = moon_gravity_acceleration_si(position_canonical)
        denom = np.linalg.norm(p_m)
        ratio[idx] = np.linalg.norm(p_s) / denom if denom > 0.0 else np.nan

    earth_mask = (xx_m - EARTH_CENTER_M[0]) ** 2 + yy_m**2 <= EARTH_RADIUS_M**2
    moon_mask = (xx_m - MOON_CENTER_M[0]) ** 2 + yy_m**2 <= MOON_RADIUS_M**2
    ratio[earth_mask | moon_mask] = np.nan

    return xx_m, yy_m, ratio


def draw_body(ax: plt.Axes, center_m: np.ndarray, radius_m: float, color: str, label: str) -> None:
    circle = plt.Circle((center_m[0] / 1e6, center_m[1] / 1e6), radius_m / 1e6, color=color, zorder=5)
    ax.add_patch(circle)
    ax.text(center_m[0] / 1e6, center_m[1] / 1e6 - 6, label, color="white", fontsize=10, weight="bold", ha="center", va="top", zorder=6)


def add_sun_arrow(ax: plt.Axes, theta_deg: float, length_m: float) -> None:
    start = np.array([0.0, 0.0], dtype=float)
    direction = np.array([np.cos(np.deg2rad(theta_deg)), np.sin(np.deg2rad(theta_deg))], dtype=float)
    end = start + direction * (length_m / 1e6)
    text_pos = start + direction * (0.68 * length_m / 1e6)
    text_alignment = "left" if direction[0] >= 0.0 else "right"
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.8},
        zorder=6,
    )
    text = ax.text(
        text_pos[0],
        text_pos[1],
        "To the Sun",
        color="white",
        fontsize=9,
        fontweight="bold",
        ha=text_alignment,
        va="bottom" if direction[1] >= 0.0 else "top",
        zorder=6,
    )
    text.set_path_effects(
        [
            path_effects.Stroke(linewidth=2.4, foreground="black"),
            path_effects.Normal(),
        ]
    )


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.tick_params(direction="out", length=4, width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")


def style_colorbar(colorbar, label: str) -> None:
    colorbar.set_label(label)
    colorbar.ax.tick_params(direction="out", length=3, width=0.8)
    colorbar.outline.set_linewidth(0.8)


def apply_report_figure_style(fig: plt.Figure, title: str) -> None:
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontweight="bold")


def build_ratio_levels(ratio: np.ndarray) -> np.ndarray:
    max_ratio = min(float(np.nanmax(ratio)), FIGURE4_RATIO_MAX)
    levels = [0.0]
    levels.extend(value for value in PAPER_RATIO_TICKS if value < max_ratio)

    if max_ratio > PAPER_RATIO_TICKS[-1]:
        extra = np.linspace(PAPER_RATIO_TICKS[-1], max_ratio, 6)[1:]
        levels.extend(extra.tolist())

    return np.unique(np.round(levels + [max_ratio], 6))


def plot_xy_magnitude(output_path: str | None, resolution: int) -> None:
    x_bounds_m = (-40e6, 40e6)
    y_bounds_m = (-38e6, 38e6)
    fig, axes = plt.subplots(3, 2, figsize=(8.6, 11.8), constrained_layout=True)

    for ax, theta_deg in zip(axes.flat, PHASES_DEG):
        field = compute_xy_plane(theta_deg, x_bounds_m, y_bounds_m, resolution)
        contour = ax.contourf(
            field.x_m / 1e6,
            field.y_m / 1e6,
            field.magnitude / 1e-6,
            levels=12,
            cmap="inferno",
        )
        ax.streamplot(
            field.x_m[0, :] / 1e6,
            field.y_m[:, 0] / 1e6,
            field.u,
            field.v,
            density=1.0,
            color="#3d6fb6",
            linewidth=0.8,
            arrowsize=0.7,
        )
        draw_body(ax, EARTH_CENTER_M, EARTH_RADIUS_M, "blue", "Earth")
        add_sun_arrow(ax, theta_deg, length_m=45e6)
        ax.set_title(rf"$\theta_2$={theta_deg}$^\circ$")
        ax.set_xlabel(r"$x$ ($10^6$ m)")
        ax.set_ylabel(r"$y$ ($10^6$ m)")
        ax.set_aspect("equal")
        ax.set_xlim(np.array(x_bounds_m) / 1e6)
        ax.set_ylim(np.array(y_bounds_m) / 1e6)
        style_axes(ax)
        cbar = fig.colorbar(contour, ax=ax)
        style_colorbar(cbar, r"$|p_s|$ ($10^{-6}$ m/s$^2$)")

    apply_report_figure_style(fig, "Replication of Paper Page 6: Solar Perturbation in the x-y Plane")
    if output_path:
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.show()


def plot_xz_magnitude(output_path: str | None, resolution: int) -> None:
    x_bounds_m = (-40e6, 40e6)
    z_bounds_m = (-40e6, 40e6)
    fig, axes = plt.subplots(3, 2, figsize=(8.6, 11.8), constrained_layout=True)

    for ax, theta_deg in zip(axes.flat, PHASES_DEG):
        field = compute_xz_plane(theta_deg, x_bounds_m, z_bounds_m, resolution)
        contour = ax.contourf(
            field.x_m / 1e6,
            field.y_m / 1e6,
            field.magnitude / 1e-6,
            levels=12,
            cmap="inferno",
        )
        ax.streamplot(
            field.x_m[0, :] / 1e6,
            field.y_m[:, 0] / 1e6,
            field.u,
            field.v,
            density=1.0,
            color="#3d6fb6",
            linewidth=0.8,
            arrowsize=0.7,
        )
        draw_body(ax, EARTH_CENTER_M, EARTH_RADIUS_M, "blue", "Earth")
        ax.set_title(rf"$\theta_2$={theta_deg}$^\circ$")
        ax.set_xlabel(r"$x$ ($10^6$ m)")
        ax.set_ylabel(r"$z$ ($10^6$ m)")
        ax.set_aspect("equal")
        ax.set_xlim(np.array(x_bounds_m) / 1e6)
        ax.set_ylim(np.array(z_bounds_m) / 1e6)
        style_axes(ax)
        cbar = fig.colorbar(contour, ax=ax)
        style_colorbar(cbar, r"$|p_s|$ ($10^{-6}$ m/s$^2$)")

    apply_report_figure_style(fig, "Replication of Paper Page 7: Solar Perturbation in the x-z Plane")
    if output_path:
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.show()


def plot_ratio_xy(output_path: str | None, resolution: int) -> None:
    x_bounds_m = (-150e6, 600e6)
    y_bounds_m = (-300e6, 300e6)
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 8.6), constrained_layout=True)

    for ax, theta_deg in zip(axes.flat, PHASES_DEG):
        xx_m, yy_m, ratio = compute_ratio_xy(theta_deg, x_bounds_m, y_bounds_m, resolution)
        ratio_for_plot = np.minimum(ratio, FIGURE4_RATIO_MAX)
        levels = build_ratio_levels(ratio)
        contour = ax.contourf(
            xx_m / 1e6,
            yy_m / 1e6,
            ratio_for_plot,
            levels=levels,
            cmap="inferno",
            extend="max",
        )
        ax.contour(
            xx_m / 1e6,
            yy_m / 1e6,
            ratio_for_plot,
            levels=levels,
            colors="#444444",
            linewidths=0.35,
            alpha=0.7,
        )
        draw_body(ax, EARTH_CENTER_M, EARTH_RADIUS_M, "blue", "Earth")
        draw_body(ax, MOON_CENTER_M, MOON_RADIUS_M, "black", "Moon")
        ax.plot(EARTH_CENTER_M[0] / 1e6, 0.0, "o", color="blue", markersize=4, zorder=6)
        ax.plot(MOON_CENTER_M[0] / 1e6, 0.0, "o", color="gray", markersize=2, zorder=6)
        ax.set_title(rf"$\theta_2$={theta_deg}$^\circ$")
        ax.set_xlabel(r"$x$ ($10^6$ m)")
        ax.set_ylabel(r"$y$ ($10^6$ m)")
        ax.set_aspect("equal")
        ax.set_xlim(np.array(x_bounds_m) / 1e6)
        ax.set_ylim(np.array(y_bounds_m) / 1e6)
        style_axes(ax)
        cbar = fig.colorbar(contour, ax=ax, ticks=levels)
        style_colorbar(cbar, r"$|p_s| / |p_m|$")

    apply_report_figure_style(fig, "Perturbation-to-Lunar-Gravity Ratio")
    if output_path:
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replicate page 6-8 perturbation plots using the current perturbed CR3BP model.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=121,
        help="Grid resolution in each direction for each subplot.",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="If provided, save PNG files using this prefix instead of only displaying them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.save_prefix
    plot_xy_magnitude(None if prefix is None else f"{prefix}_page6_xy.png", args.resolution)
    plot_xz_magnitude(None if prefix is None else f"{prefix}_page7_xz.png", args.resolution)
    plot_ratio_xy(None if prefix is None else f"{prefix}_page8_ratio.png", args.resolution)


if __name__ == "__main__":
    main()
