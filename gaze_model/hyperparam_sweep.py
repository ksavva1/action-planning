"""Run hyperparameter sweeps and save CSV/plot outputs."""

import argparse
import csv
import itertools
from dataclasses import fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

try:
    from .planning_cascade_model import DevelopmentalParams, TASKS, run_trial
except ImportError:
    from planning_cascade_model import DevelopmentalParams, TASKS, run_trial

# The default grid varies one mechanism from each major part of the model:
# gaze sampling, habit strength, anticipatory planning, and working-memory
# capacity.
DEFAULT_GRID = {
    "gaze_switch_rate": [0.10, 0.25, 0.40, 0.55, 0.70],
    "habit_strength": [0.15, 0.40, 0.65, 0.85],
    "planning_horizon": [1, 3, 5],
    "wm_capacity": [1, 3, 5],
}

BASE_PARAMS = DevelopmentalParams(
    name="sweep",
    gaze_switch_rate=0.35,
    fixation_duration_mean=3.0,
    target_bias=0.45,
    simultaneous_rate=0.20,
    sampling_rate=0.50,
    perceptual_noise=0.25,
    location_acuity=0.85,
    orientation_acuity=0.45,
    relation_acuity=0.20,
    wm_capacity=3,
    wm_decay=0.12,
    wm_unfixated_decay=0.25,
    affordance_coupling=0.45,
    affordance_noise=0.22,
    planning_horizon=3,
    motor_noise=0.22,
    habit_strength=0.45,
    goal_directed_strength=0.55,
    correction_rate=0.15,
    correction_delay=1,
    initiation_threshold=0.35,
)


def make_params_with_overrides(overrides: dict) -> DevelopmentalParams:
    """
    Build a sweep parameter set by applying overrides to BASE_PARAMS.

    If habit_strength is swept, goal_directed_strength is adjusted to keep the
    habit/goal tradeoff complementary, matching the developmental presets.
    """

    # Start from the same mid-range profile for every combination so each row
    # isolates only the parameters explicitly listed in the grid.
    params = {field.name: getattr(BASE_PARAMS, field.name) for field in fields(DevelopmentalParams)}
    params.update(overrides)
    if "habit_strength" in overrides:
        # Habit and goal-directed control are treated as competing sources of
        # motor influence, so increasing one decreases the other.
        params["goal_directed_strength"] = 1.0 - params["habit_strength"]
    params["name"] = "_".join(f"{key}={value}" for key, value in sorted(overrides.items()))
    return DevelopmentalParams(**params)


def run_sweep(grid: dict, task_names: list[str], n_trials: int = 15, seed: int = 42, verbose: bool = True) -> list[dict]:
    """
    Run every parameter combination and collect one row per trial.

    Args:
        grid: mapping from DevelopmentalParams field name to values to test.
        task_names: task keys from TASKS.
        n_trials: number of repeated trials per combination/task pair.
        seed: base seed; trial i uses seed + i.
        verbose: whether to print progress.

    Returns:
        List of CSV-ready dictionaries containing swept params and outcomes.
    """

    # Sorting gives reproducible combination order.
    param_names = sorted(grid)
    combos = list(itertools.product(*(grid[name] for name in param_names)))
    total = len(combos) * len(task_names) * n_trials
    if verbose:
        print(f"Sweep: {len(param_names)} params x {len(combos)} combos x {len(task_names)} tasks x {n_trials} trials = {total} runs")

    rows = []
    completed = 0
    for combo in combos:
        # One DevelopmentalParams object is reused for all tasks/trials in this
        # parameter combination; stochasticity comes from trial seeds.
        overrides = dict(zip(param_names, combo))
        params = make_params_with_overrides(overrides)
        for task_name in task_names:
            for trial_index in range(n_trials):
                result = run_trial(params, TASKS[task_name], seed=seed + trial_index, trial_id=trial_index)
                rows.append({
                    **overrides,
                    "task": task_name,
                    "trial": trial_index,
                    "success": int(result.success),
                    "timesteps": result.timesteps_used,
                    "pos_error": round(result.final_pos_error, 4),
                    "angle_error": round(result.final_angle_error, 4),
                    "efficiency": round(result.efficiency, 4),
                    "gaze_switches": result.total_gaze_switches,
                    "obj_fix_pct": round(result.object_fixation_pct, 4),
                    "tgt_fix_pct": round(result.target_fixation_pct, 4),
                    "movement_onset": result.movement_onset,
                })
                completed += 1
                if verbose and completed % 200 == 0:
                    print(f"  {completed}/{total} ({100 * completed / total:.0f}%)")

    if verbose:
        print(f"  {completed}/{total} complete.")
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    """
    Aggregate per-trial sweep rows by parameter combination and task.

    Returns one row per unique combination with mean success, timing, error, and
    gaze metrics.
    """

    # Any column that is not an outcome metric is a swept parameter column.
    metric_columns = {
        "task", "trial", "success", "timesteps", "pos_error", "angle_error",
        "efficiency", "gaze_switches", "obj_fix_pct", "tgt_fix_pct", "movement_onset",
    }
    param_columns = [column for column in rows[0] if column not in metric_columns]
    groups = {}
    for row in rows:
        # Group by all swept parameter values plus the task name.
        key = tuple(row[column] for column in param_columns) + (row["task"],)
        groups.setdefault(key, []).append(row)

    summary = []
    for trials in groups.values():
        # Preserve the parameter values from the first row in the group; all rows
        # in that group share them by construction.
        summary.append({
            **{column: trials[0][column] for column in param_columns},
            "task": trials[0]["task"],
            "n_trials": len(trials),
            "success_rate": np.mean([trial["success"] for trial in trials]),
            "mean_timesteps": np.mean([trial["timesteps"] for trial in trials]),
            "mean_pos_error": np.mean([trial["pos_error"] for trial in trials]),
            "mean_angle_error": np.mean([trial["angle_error"] for trial in trials]),
            "mean_efficiency": np.mean([trial["efficiency"] for trial in trials]),
            "mean_gaze_switches": np.mean([trial["gaze_switches"] for trial in trials]),
            "mean_movement_onset": np.mean([trial["movement_onset"] for trial in trials]),
        })
    return summary


def save_csv(rows: list[dict], output_path: str | Path) -> None:
    """Write rows to CSV using the first row's key order as the header."""

    if not rows:
        return
    # DictWriter keeps output columns aligned with the row dictionaries generated
    # by run_sweep() or summarise().
    with open(output_path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {output_path}  ({len(rows)} rows)")


def style_axis(ax, xlabel: str, ylabel: str) -> None:
    """Apply the shared dark plot styling used by sweep figures."""

    # Keeping axis styling central prevents the 1D and heatmap plots from
    # drifting visually as plotting code changes.
    ax.set_facecolor("#1a1e28")
    for spine in ax.spines.values():
        spine.set_color("#2a3148")
    ax.tick_params(colors="#636b83", labelsize=7)
    ax.set_xlabel(xlabel, fontsize=8, color="#9aa1b9")
    ax.set_ylabel(ylabel, fontsize=8, color="#9aa1b9")


def plot_sweep(summary: list[dict], grid: dict, output_path: str | Path) -> None:
    """
    Plot success, timing, and efficiency against each swept parameter.

    Lines are split by task so parameter effects can be read as general or
    task-specific.
    """

    # One row per swept mechanism, one column per behavioral outcome.
    param_names = sorted(grid)
    fig, axes = plt.subplots(len(param_names), 3, figsize=(16, 3.8 * len(param_names)), facecolor="#0c0e12")
    fig.subplots_adjust(hspace=.45, wspace=.3, left=.07, right=.96, top=.94, bottom=.06)
    fig.text(
        .5, .98, "Hyperparameter Sweep Results",
        ha="center", va="top", fontsize=16, fontweight="bold", color="#6ee7b7",
        fontfamily="serif", path_effects=[patheffects.withStroke(linewidth=2, foreground="#0c0e12")],
    )

    metrics = [
        ("success_rate", "Success Rate"),
        ("mean_timesteps", "Mean Timesteps"),
        ("mean_efficiency", "Efficiency"),
    ]
    task_colours = {
        "rotate_insert": "#fb7185",
        "translate_only": "#fbbf24",
        "rotate_only": "#60a5fa",
        "complex_manipulation": "#a78bfa",
    }

    for row, param_name in enumerate(param_names):
        values = sorted(set(grid[param_name]))
        for col, (metric_key, metric_title) in enumerate(metrics):
            ax = axes[row, col] if len(param_names) > 1 else axes[col]
            style_axis(ax, param_name.replace("_", " "), metric_title)
            for task_name, colour in task_colours.items():
                # Average across all other swept parameters so each line shows
                # the marginal effect of the current parameter for one task.
                points = [
                    (value, np.mean([
                        entry[metric_key] for entry in summary
                        if entry.get(param_name) == value and entry["task"] == task_name
                    ]))
                    for value in values
                    if any(entry.get(param_name) == value and entry["task"] == task_name for entry in summary)
                ]
                if points:
                    ax.plot(*zip(*points), "o-", color=colour, markersize=4, lw=1.5, alpha=.8,
                            label=task_name.replace("_", " "))
            if row == 0 and col == 0:
                ax.legend(fontsize=6, facecolor="#13161d", edgecolor="#2a3148",
                          labelcolor="#e4e8f1", loc="best")

    plt.savefig(output_path, facecolor="#0c0e12", dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_heatmaps(summary: list[dict], grid: dict, output_path: str | Path) -> None:
    """
    Plot success-rate heatmaps for exactly two swept parameters.

    The heatmap shows interactions between mechanisms, for example whether
    stronger planning only helps when working memory is also high enough.
    """

    param_names = sorted(grid)
    if len(param_names) != 2:
        # Heatmaps need two axes; larger grids are handled by plot_sweep().
        return

    x_name, y_name = param_names
    x_values = sorted(set(grid[x_name]))
    y_values = sorted(set(grid[y_name]))
    task_names = sorted({entry["task"] for entry in summary})
    fig, axes = plt.subplots(1, len(task_names), figsize=(5 * len(task_names), 4.5), facecolor="#0c0e12")
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(wspace=.35, top=.88, bottom=.15, left=.08, right=.95)
    fig.text(.5, .96, f"Success Rate Heatmap: {x_name} vs {y_name}",
             ha="center", fontsize=14, fontweight="bold", color="#6ee7b7")

    for ax, task_name in zip(axes, task_names):
        ax.set_facecolor("#1a1e28")
        matrix = np.zeros((len(y_values), len(x_values)))
        for entry in summary:
            # Rows/columns map parameter values to cells, and each task gets its
            # own panel to keep task difficulty from hiding parameter effects.
            if entry["task"] == task_name:
                matrix[y_values.index(entry[y_name]), x_values.index(entry[x_name])] = entry["success_rate"]

        image = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels(x_values, fontsize=7, color="#9aa1b9")
        ax.set_yticks(range(len(y_values)))
        ax.set_yticklabels(y_values, fontsize=7, color="#9aa1b9")
        ax.set_xlabel(x_name.replace("_", " "), fontsize=8, color="#9aa1b9")
        ax.set_ylabel(y_name.replace("_", " "), fontsize=8, color="#9aa1b9")
        ax.set_title(task_name.replace("_", " ").title(), fontsize=10, color="#e4e8f1")

        for row in range(len(y_values)):
            for col in range(len(x_values)):
                value = matrix[row, col]
                ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black" if value > .5 else "white")
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)

    plt.savefig(output_path, facecolor="#0c0e12", dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def parse_value(value: str):
    """Parse CLI values as int, then float, then leave as string."""

    # Parameter dataclass fields include both int-like and float-like values.
    for parser in (int, float):
        try:
            return parser(value)
        except ValueError:
            pass
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for sweep runs."""

    parser = argparse.ArgumentParser(description="Hyperparameter sweep")
    parser.add_argument("--trials", type=int, default=15, help="Trials per combination (default: 15)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", nargs="+", default=None, help="Tasks to include (default: all)")
    parser.add_argument("--param", default=None, help="Single param to sweep (use with --values)")
    parser.add_argument("--values", nargs="+", default=None, help="Values for --param")
    parser.add_argument("--param2", default=None, help="Optional second param for 2D sweep")
    parser.add_argument("--values2", nargs="+", default=None, help="Values for --param2")
    return parser


def main() -> None:
    """Run the CLI sweep and write CSV/PNG outputs next to this file."""

    args = build_arg_parser().parse_args()
    task_names = args.task or list(TASKS)
    grid = DEFAULT_GRID
    if args.param and args.values:
        # Supplying --param switches from the default broad grid to an explicit
        # user-defined 1D or 2D sweep.
        grid = {args.param: [parse_value(value) for value in args.values]}
        if args.param2 and args.values2:
            grid[args.param2] = [parse_value(value) for value in args.values2]

    combos = int(np.prod([len(values) for values in grid.values()]))
    print(f"\n{'=' * 60}")
    print("Hyperparam Sweep")
    print(f"{'=' * 60}")
    print(f"  Grid: {dict(grid)}")
    print(f"  Tasks: {task_names}")
    print(f"  Trials per combo: {args.trials}")
    print(f"  Total combos: {combos}")
    print(f"  Total runs: {combos * len(task_names) * args.trials}")
    print(f"{'=' * 60}\n")

    output_dir = Path(__file__).resolve().parent
    rows = run_sweep(grid, task_names, args.trials, args.seed)
    summary = summarise(rows)
    save_csv(rows, output_dir / "sweep_results.csv")
    save_csv(summary, output_dir / "sweep_summary.csv")
    plot_sweep(summary, grid, output_dir / "sweep_plots.png")
    if len(grid) == 2:
        plot_heatmaps(summary, grid, output_dir / "sweep_heatmap.png")
    print("\nDone")


if __name__ == "__main__":
    main()
