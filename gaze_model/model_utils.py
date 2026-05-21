"""Simulation and aggregation helpers."""

import json

import numpy as np

try:
    from .model_config import DEVELOPMENTAL_STAGES, TASKS
    from .planning_cascade_model import run_trial
except ImportError:
    from model_config import DEVELOPMENTAL_STAGES, TASKS
    from planning_cascade_model import run_trial


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts NumPy scalar and array values to Python types."""

    def default(self, obj):
        """Convert NumPy objects before falling back to the parent encoder."""

        # Simulation outputs often contain NumPy scalar types from vectorized
        # calculations; JSON cannot serialize those without conversion.
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_simulation(stages=None, tasks=None, n_trials=20, seed=42):
    """
    Run all requested stage x task x trial combinations.

    Args:
        stages: optional stage-name list; defaults to all developmental stages.
        tasks: optional task-name list; defaults to all tasks.
        n_trials: number of trials per stage/task pair.
        seed: base seed; trial i uses seed + i for reproducibility.

    Returns:
        Flat list of TrialResult objects.
    """

    # None means "all known presets"; callers can still pass explicit subsets.
    stages = stages or list(DEVELOPMENTAL_STAGES)
    tasks = tasks or list(TASKS)
    # Trial seeds vary within each condition while staying identical across
    # stages/tasks for the same trial index.
    return [
        run_trial(DEVELOPMENTAL_STAGES[stage], TASKS[task], seed=seed + trial, trial_id=trial)
        for stage in stages
        for task in tasks
        for trial in range(n_trials)
    ]


def group_results(results):
    """Group a flat list of TrialResult objects by (params_name, task_name).

    Args:
        results: list of TrialResult

    Returns:
        dict mapping (params_name, task_name) → list of TrialResult
    """
    groups = {}
    for result in results:
        # Grouping by names keeps aggregation independent of object identity.
        groups.setdefault((result.params_name, result.task_name), []).append(result)
    return groups


def summarise_group(stage, task, trials):
    """Compute aggregated statistics for one (stage, task) group of trials.

    Args:
        stage: stage name string
        task: task name string
        trials: list of TrialResult sharing this stage and task

    Returns:
        dict of summary statistics
    """
    # These means expose both success/error and process-level behaviors such as
    # gaze switching and movement onset.
    return {
        "stage": stage,
        "task": task,
        "n_trials": len(trials),
        "success_rate": np.mean([trial.success for trial in trials]),
        "mean_timesteps": float(np.mean([trial.timesteps_used for trial in trials])),
        "mean_pos_error": float(np.mean([trial.final_pos_error for trial in trials])),
        "mean_angle_error": float(np.mean([trial.final_angle_error for trial in trials])),
        "mean_efficiency": float(np.mean([trial.efficiency for trial in trials])),
        "mean_movement_onset": float(np.mean([trial.movement_onset for trial in trials])),
        "mean_gaze_switches": float(np.mean([trial.total_gaze_switches for trial in trials])),
        "mean_object_fixation_pct": float(np.mean([trial.object_fixation_pct for trial in trials])),
        "mean_target_fixation_pct": float(np.mean([trial.target_fixation_pct for trial in trials])),
        # Translation before rotation is a behavioral marker of the model's
        # habitual strategy, so it is reported alongside generic error metrics.
        "translate_before_rotate_rate": np.mean([
            trial.translation_onset < trial.rotation_onset for trial in trials
        ]),
    }


def best_trial_trajectory(trials):
    """
    Return compact trajectory data for the trial closest to the goal.

    The best trial is chosen by final position plus angle error so animations and
    dashboards show a representative successful-looking attempt when available.
    """

    # Select the trajectory closest to the target rather than the first trial, so
    # downstream visualizations show the clearest example for each condition.
    trial = min(trials, key=lambda item: item.final_pos_error + item.final_angle_error)
    return {
        "steps": [
            {
                "t": step.t,
                "x": round(step.obj_x, 4),
                "y": round(step.obj_y, 4),
                "a": round(step.obj_angle, 4),
                "gz": step.gaze_target,
                "mv": step.movement_started,
                "pe": round(step.pos_error, 4),
                "ae": round(step.angle_error, 4),
            }
            for step in trial.trajectory
        ],
        "success": trial.success,
        "gaze_history": trial.gaze_history,
    }


def dev_params_as_dict(stages):
    """Return serialisable developmental parameters used by the dashboard."""

    # The dashboard only needs behaviorally interpretable fields, not every noise
    # parameter or display-independent implementation detail.
    keys = [
        "gaze_switch_rate", "fixation_duration_mean", "target_bias", "simultaneous_rate",
        "sampling_rate", "perceptual_noise", "location_acuity", "orientation_acuity",
        "relation_acuity", "wm_capacity", "wm_decay", "wm_unfixated_decay",
        "affordance_coupling", "planning_horizon", "habit_strength", "goal_directed_strength",
        "correction_rate", "initiation_threshold",
    ]
    return {name: {key: getattr(params, key) for key in keys} for name, params in stages.items()}


def compile_results(results):
    """Aggregate trial results into summary stats and best-trial trajectories.

    Args:
        results: list of TrialResult from run_simulation()

    Returns:
        dict with keys "summary", "trajectories", "developmental_params",
        "stages", "tasks" — ready for JSON serialisation.
    """
    groups = group_results(results)
    # Summary metrics and example trajectories use matching keys so the dashboard
    # can join them without another lookup table.
    return {
        "summary": {
            f"{stage}_{task}": summarise_group(stage, task, trials)
            for (stage, task), trials in groups.items()
        },
        "trajectories": {
            f"{stage}_{task}": best_trial_trajectory(trials)
            for (stage, task), trials in groups.items()
        },
        "developmental_params": dev_params_as_dict(DEVELOPMENTAL_STAGES),
        "stages": list(DEVELOPMENTAL_STAGES),
        "tasks": list(TASKS),
    }
