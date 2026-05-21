"""
Hyperparameter Sweep

Sweeps over specified parameter ranges, runs N trials per combination and outputs:
  1.  sweep_results.csv  - one row per (param_combo, task, trial)
  2.  sweep_summary.csv  - aggregated stats per (param_combo, task)
  3.  sweep_plots.png    - grid of metric plots

Use:
    python hyperparam_sweep.py                      # default sweep
    python hyperparam_sweep.py --trials 30          # more trials
    python hyperparam_sweep.py --param habit_strength --values 0.1 0.3 0.5 0.7 0.9
    python hyperparam_sweep.py --param gaze_switch_rate --values 0.1 0.2 0.3 0.4 0.5 0.6 0.7

The default sweep varies 4 parameters simultaneously across a grid that covers:
  - gaze_switch_rate   (information-gathering frequency)
  - habit_strength     (habitual vs goal-directed balance)
  - planning_horizon   (lookahead depth)
  - wm_capacity        (working memory slots)
"""

import argparse
import csv
import itertools
import os
import sys
from dataclasses import fields
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from planning_cascade_model import (
    DevelopmentalParams, run_trial, TASKS, TrialResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sweep configuration
# ─────────────────────────────────────────────────────────────────────────────

# Default grid: each parameter is varied independently across a range of values.
# These four parameters were chosen because they represent distinct mechanisms
# (gaze, habit, planning, and memory) that the model predicts should have
# independent and interpretable effects on task performance.
DEFAULT_GRID = {
    "gaze_switch_rate":  [0.10, 0.25, 0.40, 0.55, 0.70],
    "habit_strength":    [0.15, 0.40, 0.65, 0.85],
    "planning_horizon":  [1, 3, 5],
    "wm_capacity":       [1, 3, 5],
}

# Base configuration: mid-range parameter values used as the starting point
# for all sweep combinations. Any parameter not listed in a combo's overrides
# takes its value from here.
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
    """Create a DevelopmentalParams from BASE_PARAMS with the given values overridden.

    Habit and goal-directed strength are kept complementary: if habit_strength is
    overridden, goal_directed_strength is automatically set to 1 - habit_strength.
    This preserves the constraint that the two weights sum to 1, as in the
    developmental stage definitions.

    Args:
        overrides: dict mapping parameter field names to override values.

    Returns:
        DevelopmentalParams with BASE_PARAMS values except where overridden.
    """
    param_dict = {field.name: getattr(BASE_PARAMS, field.name) for field in fields(DevelopmentalParams)}
    param_dict.update(overrides)

    if "habit_strength" in overrides:
        param_dict["goal_directed_strength"] = 1.0 - param_dict["habit_strength"]

    param_dict["name"] = "_".join(f"{key}={value}" for key, value in sorted(overrides.items()))

    return DevelopmentalParams(**param_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Sweep execution
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    grid: dict,
    task_names: List[str],
    n_trials: int = 15,
    seed: int = 42,
    verbose: bool = True,
) -> List[dict]:
    """Run all parameter combinations in the grid and collect per-trial results.

    Args:
        grid: dict mapping parameter names to lists of values to try.
        task_names: list of task name strings from TASKS.
        n_trials: number of trials per (combo, task) pair.
        seed: base random seed; each trial uses seed + trial_index.
        verbose: print progress updates if True.

    Returns:
        list of dicts, one per (combo, task, trial). Each contains the swept
        parameter values plus: task, trial, success, timesteps, pos_error,
        angle_error, efficiency, gaze_switches, obj_fix_pct, tgt_fix_pct,
        movement_onset.
    """
    param_names = sorted(grid.keys())
    all_combos = list(itertools.product(*(grid[name] for name in param_names)))
    total_runs = len(all_combos) * len(task_names) * n_trials

    if verbose:
        print(
            f"Sweep: {len(param_names)} params × {len(all_combos)} combos "
            f"× {len(task_names)} tasks × {n_trials} trials = {total_runs} runs"
        )

    all_rows = []
    runs_completed = 0

    for combo in all_combos:
        overrides = dict(zip(param_names, combo))
        params = make_params_with_overrides(overrides)

        for task_name in task_names:
            task = TASKS[task_name]
            for trial_index in range(n_trials):
                trial_result = run_trial(params, task, seed=seed + trial_index, trial_id=trial_index)

                trial_row = {**overrides}
                trial_row["task"] = task_name
                trial_row["trial"] = trial_index
                trial_row["success"] = int(trial_result.success)
                trial_row["timesteps"] = trial_result.timesteps_used
                trial_row["pos_error"] = round(trial_result.final_pos_error, 4)
                trial_row["angle_error"] = round(trial_result.final_angle_error, 4)
                trial_row["efficiency"] = round(trial_result.efficiency, 4)
                trial_row["gaze_switches"] = trial_result.total_gaze_switches
                trial_row["obj_fix_pct"] = round(trial_result.object_fixation_pct, 4)
                trial_row["tgt_fix_pct"] = round(trial_result.target_fixation_pct, 4)
                trial_row["movement_onset"] = trial_result.movement_onset
                all_rows.append(trial_row)

                runs_completed += 1
                if verbose and runs_completed % 200 == 0:
                    print(f"  {runs_completed}/{total_runs} ({100 * runs_completed / total_runs:.0f}%)")

    if verbose:
        print(f"  {runs_completed}/{total_runs} complete.")

    return all_rows


def summarise(rows: List[dict]) -> List[dict]:
    """Aggregate per-trial rows into per-(combo, task) summary statistics.

    Args:
        rows: list of dicts from run_sweep(), one per trial.

    Returns:
        list of dicts, one per (combo, task). Each contains the swept parameter
        values plus: task, n_trials, success_rate, mean_timesteps, mean_pos_error,
        mean_angle_error, mean_efficiency, mean_gaze_switches, mean_movement_onset.
    """
    # Identify which columns are swept parameters vs. outcome metrics.
    metric_columns = {
        "task", "trial", "success", "timesteps", "pos_error", "angle_error",
        "efficiency", "gaze_switches", "obj_fix_pct", "tgt_fix_pct", "movement_onset",
    }
    param_columns = [column for column in rows[0] if column not in metric_columns]

    groups: Dict[tuple, list] = {}
    for trial_row in rows:
        group_key = tuple(trial_row[col] for col in param_columns) + (trial_row["task"],)
        groups.setdefault(group_key, []).append(trial_row)

    summary_rows = []
    for group_key, trials in groups.items():
        summary_entry = {col: trials[0][col] for col in param_columns}
        summary_entry["task"] = trials[0]["task"]
        summary_entry["n_trials"] = len(trials)
        summary_entry["success_rate"] = np.mean([trial["success"] for trial in trials])
        summary_entry["mean_timesteps"] = np.mean([trial["timesteps"] for trial in trials])
        summary_entry["mean_pos_error"] = np.mean([trial["pos_error"] for trial in trials])
        summary_entry["mean_angle_error"] = np.mean([trial["angle_error"] for trial in trials])
        summary_entry["mean_efficiency"] = np.mean([trial["efficiency"] for trial in trials])
        summary_entry["mean_gaze_switches"] = np.mean([trial["gaze_switches"] for trial in trials])
        summary_entry["mean_movement_onset"] = np.mean([trial["movement_onset"] for trial in trials])
        summary_rows.append(summary_entry)

    return summary_rows


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(rows: list, output_path: str) -> None:
    """Write a list of dicts to a CSV file.

    Args:
        rows: list of dicts with identical keys.
        output_path: file path to write to.
    """
    if not rows:
        return
    with open(output_path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {output_path}  ({len(rows)} rows)")


def plot_sweep(summary: list, grid: dict, output_path: str) -> None:
    """Plot success rate, mean timesteps, and efficiency against each swept parameter.

    Each row of the grid corresponds to one swept parameter. Each column shows a
    different outcome metric. Lines are coloured by task to make per-task patterns
    visible — this allows the reader to see whether a parameter effect is general
    or task-specific.

    Args:
        summary: list of dicts from summarise().
        grid: dict of {param_name: [values]} defining the sweep.
        output_path: file path for the output PNG.
    """
    param_names = sorted(grid.keys())
    num_params = len(param_names)

    fig, axes = plt.subplots(num_params, 3, figsize=(16, 3.8 * num_params), facecolor="#0c0e12")
    fig.subplots_adjust(hspace=0.45, wspace=0.3, left=0.07, right=0.96, top=0.94, bottom=0.06)
    fig.text(
        0.5, 0.98, "Hyperparameter Sweep Results",
        ha="center", va="top", fontsize=16, fontweight="bold",
        color="#6ee7b7", fontfamily="serif",
        path_effects=[patheffects.withStroke(linewidth=2, foreground="#0c0e12")],
    )

    metrics = [
        ("success_rate",    "Success Rate",   "#6ee7b7"),
        ("mean_timesteps",  "Mean Timesteps", "#60a5fa"),
        ("mean_efficiency", "Efficiency",     "#fbbf24"),
    ]
    task_colours = {
        "rotate_insert":        "#fb7185",
        "translate_only":       "#fbbf24",
        "rotate_only":          "#60a5fa",
        "complex_manipulation": "#a78bfa",
    }

    for param_index, param_name in enumerate(param_names):
        param_values = sorted(set(grid[param_name]))

        for metric_index, (metric_key, metric_title, metric_colour) in enumerate(metrics):
            ax = axes[param_index, metric_index] if num_params > 1 else axes[metric_index]
            ax.set_facecolor("#1a1e28")
            for spine in ax.spines.values():
                spine.set_color("#2a3148")
            ax.tick_params(colors="#636b83", labelsize=7)
            ax.set_xlabel(param_name.replace("_", " "), fontsize=8, color="#9aa1b9")
            ax.set_ylabel(metric_title, fontsize=8, color="#9aa1b9")

            for task_name, task_colour in task_colours.items():
                data_points = []
                for param_value in param_values:
                    matching = [
                        entry for entry in summary
                        if entry.get(param_name) == param_value and entry["task"] == task_name
                    ]
                    if matching:
                        mean_metric = np.mean([entry[metric_key] for entry in matching])
                        data_points.append((param_value, mean_metric))

                if data_points:
                    x_values, y_values = zip(*data_points)
                    ax.plot(
                        x_values, y_values, "o-",
                        color=task_colour, markersize=4, lw=1.5, alpha=0.8,
                        label=task_name.replace("_", " "),
                    )

            if param_index == 0 and metric_index == 0:
                ax.legend(fontsize=6, facecolor="#13161d", edgecolor="#2a3148",
                          labelcolor="#e4e8f1", loc="best")

    plt.savefig(output_path, facecolor="#0c0e12", dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_heatmaps(summary: list, grid: dict, output_path: str) -> None:
    """Plot success rate as a 2D heatmap (only applicable for 2-parameter sweeps).

    Heatmaps make the interaction between two parameters immediately visible:
    cells that are bright indicate parameter combinations where the agent succeeds
    most often. This is useful for identifying optimal parameter regions or
    detecting non-linear interactions between mechanisms.

    Args:
        summary: list of dicts from summarise().
        grid: dict with exactly 2 keys, each mapping to a list of values.
        output_path: file path for the output PNG.
    """
    param_names = sorted(grid.keys())
    if len(param_names) != 2:
        return  # Heatmaps are only defined for exactly two swept parameters.

    param1_name, param2_name = param_names
    param1_values = sorted(set(grid[param1_name]))
    param2_values = sorted(set(grid[param2_name]))
    task_names = sorted(set(entry["task"] for entry in summary))

    fig, axes = plt.subplots(1, len(task_names), figsize=(5 * len(task_names), 4.5),
                             facecolor="#0c0e12")
    if len(task_names) == 1:
        axes = [axes]
    fig.subplots_adjust(wspace=0.35, top=0.88, bottom=0.15, left=0.08, right=0.95)
    fig.text(
        0.5, 0.96, f"Success Rate Heatmap: {param1_name} vs {param2_name}",
        ha="center", fontsize=14, fontweight="bold", color="#6ee7b7",
    )

    for task_index, task_name in enumerate(task_names):
        ax = axes[task_index]
        ax.set_facecolor("#1a1e28")

        success_matrix = np.zeros((len(param2_values), len(param1_values)))

        for summary_entry in summary:
            if summary_entry["task"] == task_name:
                col_idx = param1_values.index(summary_entry[param1_name])
                row_idx = param2_values.index(summary_entry[param2_name])
                success_matrix[row_idx, col_idx] = summary_entry["success_rate"]

        heatmap_image = ax.imshow(
            success_matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=1, origin="lower"
        )
        ax.set_xticks(range(len(param1_values)))
        ax.set_xticklabels(param1_values, fontsize=7, color="#9aa1b9")
        ax.set_yticks(range(len(param2_values)))
        ax.set_yticklabels(param2_values, fontsize=7, color="#9aa1b9")
        ax.set_xlabel(param1_name.replace("_", " "), fontsize=8, color="#9aa1b9")
        ax.set_ylabel(param2_name.replace("_", " "), fontsize=8, color="#9aa1b9")
        ax.set_title(task_name.replace("_", " ").title(), fontsize=10, color="#e4e8f1")

        # Label each cell with its success rate for precise reading.
        for row_idx in range(len(param2_values)):
            for col_idx in range(len(param1_values)):
                cell_value = success_matrix[row_idx, col_idx]
                text_colour = "black" if cell_value > 0.5 else "white"
                ax.text(col_idx, row_idx, f"{cell_value:.2f}",
                        ha="center", va="center", fontsize=6.5, color=text_colour)

        fig.colorbar(heatmap_image, ax=ax, fraction=0.046, pad=0.04)

    plt.savefig(output_path, facecolor="#0c0e12", dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_value(value_string: str):
    """Parse a CLI string value as int, then float, then string.

    Args:
        value_string: raw string from argparse.

    Returns:
        int, float, or str depending on what parses successfully.
    """
    try:
        return int(value_string)
    except ValueError:
        pass
    try:
        return float(value_string)
    except ValueError:
        pass
    return value_string


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Hyperparameter sweep")
    arg_parser.add_argument("--trials", type=int, default=15, help="Trials per combination (default: 15)")
    arg_parser.add_argument("--seed", type=int, default=42)
    arg_parser.add_argument("--task", nargs="+", default=None, help="Tasks to include (default: all)")
    arg_parser.add_argument("--param", type=str, default=None, help="Single param to sweep (use with --values)")
    arg_parser.add_argument("--values", nargs="+", default=None, help="Values for --param")
    arg_parser.add_argument("--param2", type=str, default=None, help="Optional second param for 2D sweep")
    arg_parser.add_argument("--values2", nargs="+", default=None, help="Values for --param2")
    args = arg_parser.parse_args()

    output_directory = os.path.dirname(os.path.abspath(__file__))
    task_names = args.task or list(TASKS.keys())

    if args.param and args.values:
        grid = {args.param: [parse_value(value) for value in args.values]}
        if args.param2 and args.values2:
            grid[args.param2] = [parse_value(value) for value in args.values2]
    else:
        grid = DEFAULT_GRID

    num_combos = 1
    for values in grid.values():
        num_combos *= len(values)

    print(f"\n{'=' * 60}")
    print("Hyperparam Sweep")
    print(f"{'=' * 60}")
    print(f"  Grid: {dict(grid)}")
    print(f"  Tasks: {task_names}")
    print(f"  Trials per combo: {args.trials}")
    print(f"  Total combos: {num_combos}")
    print(f"  Total runs: {num_combos * len(task_names) * args.trials}")
    print(f"{'=' * 60}\n")

    all_rows = run_sweep(grid, task_names, args.trials, args.seed)
    summary_rows = summarise(all_rows)

    save_csv(all_rows, os.path.join(output_directory, "sweep_results.csv"))
    save_csv(summary_rows, os.path.join(output_directory, "sweep_summary.csv"))

    plot_sweep(summary_rows, grid, os.path.join(output_directory, "sweep_plots.png"))
    if len(grid) == 2:
        plot_heatmaps(summary_rows, grid, os.path.join(output_directory, "sweep_heatmap.png"))

    print("\nDone")
