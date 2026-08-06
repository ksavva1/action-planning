"""Figures for the three experiments."""

import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from experiment_config import (
    BATTERY_ASPECTS,
    BATTERY_DISTANCES,
    BATTERY_ROTATIONS,
    INFORMATIVE_BAND,
    PROFILE_COLOURS,
    SENSITIVITY_METRICS,
)

PLOT_METRICS = [
    ("success_rate", "Success rate", "#2196F3", (0, 1.05)),
    ("mean_efficiency", "Path efficiency", "#E76C1F", (0, 1.05)),
    ("mean_movement_onset", "Movement onset (ts)", "#7C0DC2", None),
    ("mean_gaze_switches", "Gaze switches", "#2E7D32", None),
]


def _style():
    """Apply the shared white-background, spineless matplotlib style."""

    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


# Experiment 1
def plot_success_dotplot(cell_summaries, profile_names, colours=None):
    """Per-task success rate for each profile, as a dot plot with the profile mean.

    Args:
        cell_summaries: list of per-(profile, task) summary dicts.
        profile_names: ordered profile labels to plot.
        colours: mapping from profile name to colour; defaults to
            PROFILE_COLOURS.

    Returns:
        The matplotlib Figure.
    """

    _style()
    colours = colours or PROFILE_COLOURS
    by_profile = {name: [] for name in profile_names}
    for row in cell_summaries:
        by_profile[row["stage"]].append(row["success_rate"])

    figure, axis = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(7)
    for position, name in enumerate(profile_names):
        values = np.array(by_profile[name])
        jitter = rng.uniform(-0.12, 0.12, len(values))
        axis.scatter(
            position + jitter,
            values,
            s=42,
            alpha=0.75,
            color=colours[name],
            edgecolors="none",
            zorder=3,
        )
        axis.hlines(
            values.mean(),
            position - 0.28,
            position + 0.28,
            color="black",
            lw=2.2,
            zorder=5,
        )
        axis.hlines(
            np.median(values),
            position - 0.28,
            position + 0.28,
            color="grey",
            lw=1.4,
            ls="--",
            zorder=4,
        )
        # Proportion of tasks at each bound, which is what distinguishes a
        # bimodal profile from a broadly distributed one.
        axis.text(
            position,
            1.09,
            f"{np.mean(values <= 0.05):.0%} at floor",
            ha="center",
            fontsize=8,
            color="grey",
        )
        axis.text(
            position,
            1.03,
            f"{np.mean(values >= 0.95):.0%} at ceiling",
            ha="center",
            fontsize=8,
            color="grey",
        )

    axis.set_xticks(range(len(profile_names)))
    axis.set_xticklabels(profile_names, fontsize=12)
    axis.set_xlabel("Developmental profile", fontsize=11)
    axis.set_ylabel("Per-task success rate", fontsize=11)
    axis.set_ylim(-0.05, 1.16)
    axis.grid(axis="y", alpha=0.3)
    axis.set_title(
        "Per-task success rate across the battery\n"
        "(one point per task; solid line = mean, dashed = median)",
        fontsize=12,
    )
    plt.tight_layout()
    return figure


def plot_task_heatmap(cell_summaries, profile_names, task_meta):
    """Task-by-profile heatmap of success rate, ordered by task difficulty.

    Args:
        cell_summaries: list of per-(profile, task) summary dicts.
        profile_names: ordered profile labels for the rows.
        task_meta: mapping from task name to its metadata dict.

    Returns:
        The matplotlib Figure.
    """

    _style()
    tasks = sorted(
        task_meta,
        key=lambda name: (
            task_meta[name]["rot_idx"],
            task_meta[name]["dist_idx"],
            task_meta[name]["aspect_idx"],
        ),
    )
    lookup = {
        (row["stage"], row["task"]): row["success_rate"] for row in cell_summaries
    }
    grid = np.array(
        [[lookup[(profile, task)] for task in tasks] for profile in profile_names]
    )

    figure, axis = plt.subplots(figsize=(13, 2.6))
    image = axis.imshow(grid, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    axis.set_yticks(range(len(profile_names)))
    axis.set_yticklabels(profile_names)
    axis.set_xticks(range(len(tasks)))
    axis.set_xticklabels(
        [
            f"{math.degrees(task_meta[t]['rot']):.0f}°\n{task_meta[t]['dist']:.2f}\n{task_meta[t]['shape'][:2]}"
            for t in tasks
        ],
        fontsize=6.5,
    )
    axis.set_xlabel("Task (rotation / distance / shape)", fontsize=9)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            value = grid[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=5.8,
                color="white" if value > 0.55 else "black",
            )
    figure.colorbar(image, ax=axis, shrink=0.9, label="Success rate")
    axis.set_title("Success rate by task and profile", fontsize=11)
    plt.tight_layout()
    return figure


def plot_marginal_effects(marginals, profile_names, colours=None):
    """Marginal success rate along each task dimension, with Wilson intervals.

    Intervals are computed on the pooled trials at each level rather than on an
    average of per-task proportions, so they reflect the number of trials the
    point actually rests on.

    Args:
        marginals: mapping from factor name to per-profile level cells, as
            returned by :func:`analysis.marginal_by_factor`.
        profile_names: ordered profile labels to plot.
        colours: mapping from profile name to colour; defaults to
            PROFILE_COLOURS.

    Returns:
        The matplotlib Figure.
    """

    _style()
    colours = colours or PROFILE_COLOURS
    dimensions = [
        ("dist", "Distance (units)", [f"{d:.2f}" for d in BATTERY_DISTANCES]),
        ("rot", "Rotation", [f"{math.degrees(r):.0f}°" for r in BATTERY_ROTATIONS]),
        (
            "aspect",
            "Aspect ratio (w/h)",
            [
                f"{a}\n({'tall' if a < 1 else 'square' if a == 1 else 'wide'})"
                for a in BATTERY_ASPECTS
            ],
        ),
    ]

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for axis, (factor, label, ticks) in zip(axes, dimensions):
        for profile in profile_names:
            levels = sorted(marginals[factor][profile])
            rates = [
                marginals[factor][profile][level]["success_rate"] for level in levels
            ]
            # Clipped at zero: a Wilson bound can land a floating-point step
            # the wrong side of an estimate that sits exactly at 0 or 1.
            low = np.clip(
                [
                    rate - marginals[factor][profile][level]["ci_low"]
                    for rate, level in zip(rates, levels)
                ],
                0,
                None,
            )
            high = np.clip(
                [
                    marginals[factor][profile][level]["ci_high"] - rate
                    for rate, level in zip(rates, levels)
                ],
                0,
                None,
            )
            axis.errorbar(
                range(len(levels)),
                rates,
                yerr=[low, high],
                fmt="o-",
                color=colours[profile],
                lw=2.0,
                markersize=7,
                capsize=3,
                label=f"Profile {profile}",
            )
        axis.set_xticks(range(len(ticks)))
        axis.set_xticklabels(ticks, fontsize=9)
        axis.set_xlabel(label, fontsize=11)
        axis.grid(alpha=0.3)
        axis.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Success rate (marginal)", fontsize=11)
    axes[-1].legend(fontsize=9)
    figure.suptitle(
        "Marginal effects of task geometry, with 95% Wilson intervals",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    return figure


def plot_convergence(curves, profile_names, colours=None, target_half_width=0.05):
    """Monte Carlo convergence: estimate and interval half-width against trials per cell.

    Args:
        curves: mapping from "profile_<name>" to a convergence curve, as
            returned by :func:`analysis.convergence_curve`.
        profile_names: ordered profile labels to plot.
        colours: mapping from profile name to colour; defaults to
            PROFILE_COLOURS.
        target_half_width: half-width threshold marked on the right panel.

    Returns:
        The matplotlib Figure.
    """

    _style()
    colours = colours or PROFILE_COLOURS
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    for profile in profile_names:
        curve = curves[f"profile_{profile}"]
        n = [point["n"] for point in curve]
        estimate = [point["estimate"] for point in curve]
        low = [point["ci_low"] for point in curve]
        high = [point["ci_high"] for point in curve]
        axes[0].plot(
            n, estimate, "o-", color=colours[profile], lw=2, label=f"Profile {profile}"
        )
        axes[0].fill_between(n, low, high, color=colours[profile], alpha=0.15)
        axes[1].plot(
            n,
            [point["half_width"] for point in curve],
            "o-",
            color=colours[profile],
            lw=2,
            label=f"Profile {profile}",
        )

    axes[0].axvline(20, color="grey", ls="--", lw=1.2)
    axes[0].text(20, 1.02, " original design", fontsize=8, color="grey")
    axes[0].set_xlabel("Trials per cell")
    axes[0].set_ylabel("Profile success rate")
    axes[0].set_ylim(-0.03, 1.08)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].axhline(target_half_width, color="grey", ls="--", lw=1.2)
    axes[1].text(
        axes[1].get_xlim()[1],
        target_half_width,
        f" {target_half_width:.0%}",
        fontsize=8,
        color="grey",
        va="bottom",
        ha="right",
    )
    axes[1].set_xlabel("Trials per cell")
    axes[1].set_ylabel("95% interval half-width")
    axes[1].grid(alpha=0.3)

    figure.suptitle(
        "Monte Carlo convergence of the profile-level success estimate", fontsize=12
    )
    plt.tight_layout()
    return figure


def plot_seed_batches(profile_summary, profile_names, colours=None):
    """Success rate in each independent seed batch, against the pooled estimate.

    Args:
        profile_summary: mapping from profile name to its summary dict,
            including ``batch_success_rates`` and ``success_rate``.
        profile_names: ordered profile labels to plot.
        colours: mapping from profile name to colour; defaults to
            PROFILE_COLOURS.

    Returns:
        The matplotlib Figure.
    """

    _style()
    colours = colours or PROFILE_COLOURS
    figure, axis = plt.subplots(figsize=(7, 4.4))
    for position, profile in enumerate(profile_names):
        rates = profile_summary[profile]["batch_success_rates"]
        axis.scatter(
            [position] * len(rates),
            rates,
            s=48,
            alpha=0.8,
            color=colours[profile],
            zorder=3,
        )
        pooled = profile_summary[profile]["success_rate"]
        axis.hlines(
            pooled, position - 0.25, position + 0.25, color="black", lw=2, zorder=4
        )
        axis.text(
            position + 0.30,
            pooled,
            f"range {max(rates) - min(rates):.3f}",
            fontsize=8,
            va="center",
            color="grey",
        )
    axis.set_xticks(range(len(profile_names)))
    axis.set_xticklabels(profile_names)
    axis.set_xlabel("Developmental profile")
    axis.set_ylabel("Success rate")
    axis.set_ylim(-0.05, 1.12)
    axis.grid(axis="y", alpha=0.3)
    axis.set_title(
        "Success rate in each independent seed batch\n(black line = pooled estimate)",
        fontsize=11,
    )
    plt.tight_layout()
    return figure


def plot_radar(profile_summary, profile_names, colours=None):
    """Spider chart: one polygon per profile, six normalised performance axes.

    Args:
        profile_summary: mapping from profile name to its summary dict.
        profile_names: ordered profile labels to plot.
        colours: mapping from profile name to colour; defaults to
            PROFILE_COLOURS.

    Returns:
        The matplotlib Figure.
    """

    _style()
    colours = colours or PROFILE_COLOURS
    axes_spec = [
        ("Success\nRate", "success_rate", True),
        ("Path\nEfficiency", "mean_efficiency", True),
        ("Target\nGaze %", "mean_target_fixation", True),
        ("Speed\n(inv. steps)", "mean_timesteps", False),
        ("Onset\n(inv. delay)", "mean_movement_onset", False),
        ("Pos.\nAccuracy", "mean_pos_error", False),
    ]
    labels = [spec[0] for spec in axes_spec]
    count = len(labels)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False).tolist()
    angles += angles[:1]

    raw = {
        name: [profile_summary[name][metric] for _, metric, _ in axes_spec]
        for name in profile_names
    }
    minima = [min(raw[name][index] for name in profile_names) for index in range(count)]
    maxima = [max(raw[name][index] for name in profile_names) for index in range(count)]

    def normalise(value, low, high, higher_better):
        span = high - low
        if span == 0:
            return 0.5
        return (value - low) / span if higher_better else 1 - (value - low) / span

    figure, axis = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)
    axis.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10.5)
    axis.set_ylim(0, 1)
    axis.set_yticks([0.25, 0.5, 0.75, 1.0])
    axis.set_yticklabels(["25 %", "50 %", "75 %", "100 %"], fontsize=7, color="grey")
    axis.grid(alpha=0.3)

    for name in profile_names:
        values = [
            normalise(
                raw[name][index], minima[index], maxima[index], axes_spec[index][2]
            )
            for index in range(count)
        ]
        values += values[:1]
        axis.plot(
            angles,
            values,
            "o-",
            color=colours[name],
            lw=2.2,
            markersize=7,
            label=f"Profile {name}",
        )
        axis.fill(angles, values, alpha=0.12, color=colours[name])

    axis.legend(loc="upper right", bbox_to_anchor=(1.40, 1.20), fontsize=11)
    axis.set_title(
        "Performance profile across the battery\n(outer edge = best)",
        pad=24,
        fontsize=12,
    )
    plt.tight_layout()
    return figure


# Experiment 2
def plot_sensitivity_heatmap(index, sweep_summaries, metrics=None):
    """Relative sensitivity index per parameter and metric.

    Args:
        index: output of :func:`analysis.relative_sensitivity_index`.
        sweep_summaries: mapping from parameter to {value: summary dict}.
        metrics: list of (key, label) pairs; defaults to SENSITIVITY_METRICS.

    Returns:
        The matplotlib Figure.
    """

    _style()
    metrics = metrics or SENSITIVITY_METRICS
    order = index["order"]
    grid = np.array(
        [
            [index["normalised"][parameter][key] for key, _ in metrics]
            for parameter in order
        ]
    )

    censored_flags = np.zeros_like(grid, dtype=bool)
    censor_keys = {
        "mean_movement_onset": "censored_movement_onset",
        "mean_efficiency": "censored_efficiency",
    }
    for row, parameter in enumerate(order):
        for column, (key, _) in enumerate(metrics):
            censor_key = censor_keys.get(key)
            if not censor_key:
                continue
            censored_flags[row, column] = any(
                summary.get(censor_key, 0.0) > 0.10
                for summary in sweep_summaries[parameter].values()
            )

    figure, axis = plt.subplots(
        figsize=(len(metrics) * 1.5 + 2.5, len(order) * 0.45 + 1.4),
        constrained_layout=True,
    )
    image = axis.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    axis.set_xticks(range(len(metrics)))
    axis.set_xticklabels([label for _, label in metrics], fontsize=9)
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels(
        [parameter.replace("_", " ") for parameter in order], fontsize=9
    )

    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            value = grid[row, column]
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if value > 0.65 else "black",
            )
            if censored_flags[row, column]:
                axis.add_patch(
                    plt.Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#1565C0",
                        lw=1.8,
                    )
                )

    figure.colorbar(
        image,
        ax=axis,
        shrink=0.6,
        label="Relative sensitivity index (0 = no effect, 1 = largest observed range)",
    )
    axis.set_title(
        "Parameter sensitivity — normalised metric range across each sweep\n"
        "(blue outline: measure censored at some sweep value)",
        fontsize=11,
        pad=8,
    )
    return figure


def plot_group_sweep(group_name, parameter_names, sweep_summaries, baseline):
    """Each parameter in a group plotted against the four headline metrics.

    Args:
        group_name: label for the parameter group, used in the title.
        parameter_names: parameters to plot, one row per parameter.
        sweep_summaries: mapping from parameter to {value: summary dict}.
        baseline: DevelopmentalParams giving each parameter's baseline value.
    Returns:
        The matplotlib Figure.
    """

    _style()
    rows, columns = len(parameter_names), len(PLOT_METRICS)
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.5 * columns, 2.6 * rows), constrained_layout=True
    )
    if rows == 1:
        axes = axes[np.newaxis, :]

    for row, parameter in enumerate(parameter_names):
        values = sorted(sweep_summaries[parameter])
        baseline_value = getattr(baseline, parameter)
        for column, (metric, title, colour, ylim) in enumerate(PLOT_METRICS):
            axis = axes[row, column]
            series = [sweep_summaries[parameter][value][metric] for value in values]
            axis.plot(values, series, "o-", color=colour, lw=2, markersize=5, zorder=3)

            # Mark settings at which the metric was censored
            censor_key = {
                "mean_movement_onset": "censored_movement_onset",
                "mean_efficiency": "censored_efficiency",
            }.get(metric)
            if censor_key:
                censored = [
                    value
                    for value in values
                    if sweep_summaries[parameter][value].get(censor_key, 0.0) > 0.10
                ]
                for value in censored:
                    axis.axvspan(
                        value - 1e-9, value + 1e-9, color="#1565C0", alpha=0.25, lw=6
                    )

            axis.axvline(baseline_value, color="#bbbbbb", lw=1.5, ls="--", zorder=2)
            if ylim:
                axis.set_ylim(*ylim)
            if row == 0:
                axis.set_title(title, fontsize=10, color=colour)
            if column == 0:
                axis.set_ylabel(
                    parameter.replace("_", "\n"),
                    fontsize=8,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=4,
                )
            if row == rows - 1:
                axis.set_xlabel("parameter value", fontsize=7)
            axis.tick_params(labelsize=7)
            axis.grid(alpha=0.25)
            axis.xaxis.set_major_locator(mticker.MaxNLocator(5))

    figure.suptitle(
        f"{group_name} — effect of each parameter in isolation",
        fontsize=13,
        fontweight="bold",
    )
    return figure


def plot_rank_stability(grid, stability, top_n=8):
    """Where each parameter ranks under every baseline-by-task combination.

    Args:
        grid: mapping from design label to its sensitivity results.
        stability: output of :func:`analysis.rank_stability`.
        top_n: number of most-influential parameters to display.

    Returns:
        The matplotlib Figure.
    """

    _style()
    designs = list(grid)
    ranks = stability["ranks"]
    parameters = sorted(
        stability["per_parameter"],
        key=lambda name: stability["per_parameter"][name]["mean_rank"],
    )[:top_n]

    figure, axis = plt.subplots(figsize=(9, 0.42 * len(parameters) + 2.2))
    for row, parameter in enumerate(parameters):
        positions = [ranks[design][parameter] for design in designs]
        axis.plot(
            positions,
            [row] * len(positions),
            "o",
            color="#0584CD",
            alpha=0.6,
            markersize=7,
        )
        axis.hlines(
            row, min(positions), max(positions), color="#0584CD", alpha=0.35, lw=2
        )
        axis.plot(
            stability["per_parameter"][parameter]["mean_rank"],
            row,
            "D",
            color="black",
            markersize=6,
        )
    axis.set_yticks(range(len(parameters)))
    axis.set_yticklabels(
        [parameter.replace("_", " ") for parameter in parameters], fontsize=9
    )
    axis.invert_yaxis()
    axis.set_xlabel("Rank by relative sensitivity index (1 = most influential)")
    axis.grid(axis="x", alpha=0.3)
    taus = [pair["kendall_tau"] for pair in stability["pairwise"].values()]
    axis.set_title(
        f"Rank of each parameter across {len(designs)} baseline x task combinations\n"
        f"(pairwise Kendall tau: median {np.median(taus):.2f}, range {min(taus):.2f}-{max(taus):.2f})",
        fontsize=11,
    )
    plt.tight_layout()
    return figure


# Experiment 3
def plot_matrix_by_profile_task(
    cells, profile_names, task_names, variants, band=INFORMATIVE_BAND
):
    """Success by matrix variant, one panel per profile, with Wilson intervals.

    Args:
        cells: nested dict ``{profile: {task: {variant: summary}}}``.
        profile_names: ordered profile labels, one panel per profile.
        task_names: ordered task labels for the x-axis.
        variants: iterable of matrix variant names to plot.
        band: (low, high) success-rate bounds defining "informative".

    Returns:
        The matplotlib Figure.
    """

    _style()
    low, high = band
    figure, axes = plt.subplots(
        1, len(profile_names), figsize=(4.0 * len(profile_names), 4.6), sharey=True
    )
    width = 0.8 / len(variants)
    palette = plt.get_cmap("tab10")

    for axis, profile in zip(axes, profile_names):
        for index, variant in enumerate(variants):
            rates, lows, highs = [], [], []
            for task in task_names:
                summary = cells[profile][task][variant]
                rate = summary["success_rate"]
                rates.append(rate)
                lows.append(max(0.0, rate - summary["success_ci_low"]))
                highs.append(max(0.0, summary["success_ci_high"] - rate))
            positions = np.arange(len(task_names)) + index * width - 0.4 + width / 2
            axis.bar(
                positions,
                rates,
                width=width,
                color=palette(index),
                label=variant.replace("_", " "),
                alpha=0.9,
            )
            axis.errorbar(
                positions,
                rates,
                yerr=[lows, highs],
                fmt="none",
                ecolor="black",
                elinewidth=0.9,
                capsize=1.8,
                alpha=0.7,
            )

        for position, task in enumerate(task_names):
            baseline_rate = cells[profile][task]["baseline"]["success_rate"]
            if not (low <= baseline_rate <= high):
                axis.axvspan(
                    position - 0.45, position + 0.45, color="grey", alpha=0.16, zorder=0
                )

        axis.set_xticks(range(len(task_names)))
        axis.set_xticklabels(
            [name.replace("_", "\n") for name in task_names], fontsize=7.5
        )
        axis.set_title(f"Profile {profile}", fontsize=11)
        axis.set_ylim(0, 1.08)
        axis.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Success rate", fontsize=11)
    axes[-1].legend(fontsize=7.5, loc="upper right")
    figure.suptitle(
        "Affordance matrix variants across a difficulty-graded task set\n"
        "(shaded tasks are at floor or ceiling and cannot express a matrix effect)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    return figure


def plot_affordance_matrices(variants, feature_labels, action_labels):
    """Heatmaps of the five affordance weight matrices.

    Args:
        variants: mapping from matrix variant name to its weight matrix.
        feature_labels: row labels, one per working-memory feature.
        action_labels: column labels, one per affordance.

    Returns:
        The matplotlib Figure.
    """

    _style()
    figure, axes = plt.subplots(1, len(variants), figsize=(2.7 * len(variants), 5.2))
    for axis, (name, matrix) in zip(np.atleast_1d(axes), variants.items()):
        image = axis.imshow(matrix, cmap="Purples", vmin=0, vmax=1, aspect="auto")
        axis.set_title(name.replace("_", "\n"), fontsize=10)
        axis.set_xticks(range(len(action_labels)))
        axis.set_xticklabels(action_labels, rotation=90, fontsize=7)
        axis.set_yticks(range(len(feature_labels)))
        axis.set_yticklabels(
            feature_labels if axis is np.atleast_1d(axes)[0] else [], fontsize=7
        )
    figure.colorbar(image, ax=axes, shrink=0.7, label="Weight")
    return figure
