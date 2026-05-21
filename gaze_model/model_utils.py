"""
Simulation runner and result-aggregation utilities for the planning cascade model.
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_config import DEVELOPMENTAL_STAGES, TASKS


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder."""

    def default(self, obj):
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
    """Run all stage x task x trial combinations and return the raw results list.

    Args:
        stages: list of stage name strings (default: all in DEVELOPMENTAL_STAGES)
        tasks: list of task name strings (default: all in TASKS)
        n_trials: number of trials per (stage, task) pair
        seed: base random seed; each trial uses seed + trial_index

    Returns:
        list of TrialResult, one per trial.
    """
    from planning_cascade_model import run_trial

    if stages is None:
        stages = list(DEVELOPMENTAL_STAGES.keys())
    if tasks is None:
        tasks = list(TASKS.keys())

    results = []
    for stage_name in stages:
        stage_params = DEVELOPMENTAL_STAGES[stage_name]
        for task_name in tasks:
            task = TASKS[task_name]
            for trial_index in range(n_trials):
                results.append(run_trial(stage_params, task, seed=seed + trial_index, trial_id=trial_index))

    return results


def group_results(results):
    """Group a flat list of TrialResult objects by (params_name, task_name).

    Args:
        results: list of TrialResult

    Returns:
        dict mapping (params_name, task_name) → list of TrialResult
    """
    groups = {}
    for trial_result in results:
        group_key = (trial_result.params_name, trial_result.task_name)
        groups.setdefault(group_key, []).append(trial_result)
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
    num_trials = len(trials)

    return {
        "stage": stage,
        "task": task,
        "n_trials": num_trials,
        "success_rate": sum(trial.success for trial in trials) / num_trials,
        "mean_timesteps": float(np.mean([trial.timesteps_used for trial in trials])),
        "mean_pos_error": float(np.mean([trial.final_pos_error for trial in trials])),
        "mean_angle_error": float(np.mean([trial.final_angle_error for trial in trials])),
        "mean_efficiency": float(np.mean([trial.efficiency for trial in trials])),
        "mean_movement_onset": float(np.mean([trial.movement_onset for trial in trials])),
        "mean_gaze_switches": float(np.mean([trial.total_gaze_switches for trial in trials])),
        "mean_object_fixation_pct": float(np.mean([trial.object_fixation_pct for trial in trials])),
        "mean_target_fixation_pct": float(np.mean([trial.target_fixation_pct for trial in trials])),
        # Translation onset before rotation onset is a key behavioural index:
        # younger agents with stronger habitual biases show this pattern more often,
        # even when the task geometry requires rotation first.
        "translate_before_rotate_rate": sum(
            1 for trial in trials if trial.translation_onset < trial.rotation_onset
        ) / num_trials,
    }


def best_trial_trajectory(trials):
    """Return serialisable trajectory data for the trial closest to the goal.

    The best trial is selected by combined position and angle error so that
    the animation always shows the most informative example for each condition.

    Args:
        trials: list of TrialResult

    Returns:
        dict with keys "steps", "success", "gaze_history"
    """
    best_trial = min(trials, key=lambda trial: trial.final_pos_error + trial.final_angle_error)

    steps = [
        {
            "t": timestep_record.t,
            "x": round(timestep_record.obj_x, 4),
            "y": round(timestep_record.obj_y, 4),
            "a": round(timestep_record.obj_angle, 4),
            "gz": timestep_record.gaze_target,
            "mv": timestep_record.movement_started,
            "pe": round(timestep_record.pos_error, 4),
            "ae": round(timestep_record.angle_error, 4),
        }
        for timestep_record in best_trial.trajectory
    ]

    return {
        "steps": steps,
        "success": best_trial.success,
        "gaze_history": best_trial.gaze_history,
    }


def dev_params_as_dict(stages_dict):
    """Serialise a stages dict to plain param-value dicts.

    Args:
        stages_dict: dict of {stage_name: DevelopmentalParams}

    Returns:
        dict of {stage_name: {param_name: value}}
    """
    param_keys = [
        "gaze_switch_rate", "fixation_duration_mean", "target_bias", "simultaneous_rate",
        "sampling_rate", "perceptual_noise", "location_acuity", "orientation_acuity",
        "relation_acuity", "wm_capacity", "wm_decay", "wm_unfixated_decay",
        "affordance_coupling", "planning_horizon", "habit_strength", "goal_directed_strength",
        "correction_rate", "initiation_threshold",
    ]
    return {
        stage_name: {key: getattr(stage_params, key) for key in param_keys}
        for stage_name, stage_params in stages_dict.items()
    }


def compile_results(results):
    """Aggregate trial results into summary stats and best-trial trajectories.

    Args:
        results: list of TrialResult from run_simulation()

    Returns:
        dict with keys "summary", "trajectories", "developmental_params",
        "stages", "tasks" — ready for JSON serialisation.
    """
    groups = group_results(results)
    summary = {}
    trajectories = {}

    for (stage, task), trials in groups.items():
        group_key = f"{stage}_{task}"
        summary[group_key] = summarise_group(stage, task, trials)
        trajectories[group_key] = best_trial_trajectory(trials)

    return {
        "summary": summary,
        "trajectories": trajectories,
        "developmental_params": dev_params_as_dict(DEVELOPMENTAL_STAGES),
        "stages": list(DEVELOPMENTAL_STAGES.keys()),
        "tasks": list(TASKS.keys()),
    }
