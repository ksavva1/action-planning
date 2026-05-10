"""
Configuration dataclasses and preset constants for the planning cascade model.
"""

from dataclasses import dataclass
from typing import List


# Parameters
@dataclass
class DevelopmentalParams:
    """
    Attributes:
        name: identifier for config
        gaze_switch_rate: base probability of switching fixation per timestep (0-1)
        fixation_duration_mean: mean dwell timesteps before switch becomes likely
        target_bias: probability of looking at goal rather than object when switching (0-1)
        simultaneous_rate: probability of extracting relational features when both entities have been recently fixated (0-1)
        sampling_rate: probability that each feature is sampled per timestep (0-1)
        perceptual_noise: std dev of Gaussian noise added to sampled percepts
        location_acuity: noise reduction factor for position features (0-1, higher=clearer)
        orientation_acuity: noise reduction factor for angle features (0-1)
        relation_acuity: noise reduction factor for relational features (0-1)
        wm_capacity: maximum number of strong memory traces maintained simultaneously
        wm_decay: per-timestep decay rate for traces of the fixated entity (0-1)
        wm_unfixated_decay: faster decay rate for traces of the non-fixated entity (0-1)
        affordance_coupling: scaling factor for the percept-to-affordance weight matrix (0-1)
        affordance_noise: std dev of noise in affordance estimation
        planning_horizon: timesteps of motor lookahead (1=reactive, 6=anticipatory)
        motor_noise: std dev of execution noise on motor commands
        habit_strength: weight of the habitual translate-first bias (0-1)
        goal_directed_strength: weight of goal-directed motor control (0-1)
        correction_rate: gain of online error-correction signal (0-1)
        correction_delay: timesteps of processing lag before correction activates
        initiation_threshold: minimum mean WM strength required to begin movement (0-1)
    """

    name: str = "B"
    # Gaze
    gaze_switch_rate: float = 0.30
    fixation_duration_mean: float = 3.0
    target_bias: float = 0.40
    simultaneous_rate: float = 0.15
    # Perception
    sampling_rate: float = 0.40
    perceptual_noise: float = 0.30
    location_acuity: float = 0.85
    orientation_acuity: float = 0.40
    relation_acuity: float = 0.15
    # WM
    wm_capacity: int = 3
    wm_decay: float = 0.12
    wm_unfixated_decay: float = 0.28
    # Affordance
    affordance_coupling: float = 0.40
    affordance_noise: float = 0.25
    # Motor
    planning_horizon: int = 2
    motor_noise: float = 0.25
    # Habit
    habit_strength: float = 0.55
    goal_directed_strength: float = 0.45
    # Correction
    correction_rate: float = 0.12
    correction_delay: int = 2
    # Initiation
    initiation_threshold: float = 0.35


DEVELOPMENTAL_STAGES = {
    "A": DevelopmentalParams(
        name="A",
        gaze_switch_rate=0.12, fixation_duration_mean=5.0,
        target_bias=0.20, simultaneous_rate=0.03,
        sampling_rate=0.30, perceptual_noise=0.35,
        location_acuity=0.70, orientation_acuity=0.15, relation_acuity=0.05,
        wm_capacity=2, wm_decay=0.18, wm_unfixated_decay=0.40,
        affordance_coupling=0.25, affordance_noise=0.25,
        planning_horizon=1, motor_noise=0.20,
        habit_strength=0.78, goal_directed_strength=0.22,
        correction_rate=0.12, correction_delay=2,
        initiation_threshold=0.15,
    ),
    "B": DevelopmentalParams(
        name="B",
        gaze_switch_rate=0.25, fixation_duration_mean=3.5,
        target_bias=0.35, simultaneous_rate=0.12,
        sampling_rate=0.45, perceptual_noise=0.28,
        location_acuity=0.85, orientation_acuity=0.30, relation_acuity=0.12,
        wm_capacity=3, wm_decay=0.12, wm_unfixated_decay=0.28,
        affordance_coupling=0.40, affordance_noise=0.22,
        planning_horizon=2, motor_noise=0.22,
        habit_strength=0.70, goal_directed_strength=0.30,
        correction_rate=0.14, correction_delay=2,
        initiation_threshold=0.28,
    ),
    "C": DevelopmentalParams(
        name="C",
        gaze_switch_rate=0.40, fixation_duration_mean=2.5,
        target_bias=0.50, simultaneous_rate=0.28,
        sampling_rate=0.65, perceptual_noise=0.16,
        location_acuity=0.93, orientation_acuity=0.55, relation_acuity=0.30,
        wm_capacity=4, wm_decay=0.08, wm_unfixated_decay=0.18,
        affordance_coupling=0.60, affordance_noise=0.14,
        planning_horizon=4, motor_noise=0.16,
        habit_strength=0.40, goal_directed_strength=0.60,
        correction_rate=0.22, correction_delay=1,
        initiation_threshold=0.40,
    ),
    "D": DevelopmentalParams(
        name="D",
        gaze_switch_rate=0.55, fixation_duration_mean=2.0,
        target_bias=0.60, simultaneous_rate=0.50,
        sampling_rate=0.85, perceptual_noise=0.08,
        location_acuity=0.98, orientation_acuity=0.88, relation_acuity=0.55,
        wm_capacity=5, wm_decay=0.04, wm_unfixated_decay=0.10,
        affordance_coupling=0.85, affordance_noise=0.08,
        planning_horizon=6, motor_noise=0.08,
        habit_strength=0.10, goal_directed_strength=0.90,
        correction_rate=0.28, correction_delay=0,
        initiation_threshold=0.45,
    ),
}


# Tasks
@dataclass
class TaskConfig:
    """
    Defines an object manipulation task with start/goal poses and success criteria.

    Attributes:
        name: identifier string for this task
        start_x, start_y, start_angle: initial object pose
        goal_x, goal_y, goal_angle: target slot pose the object must reach
        obj_width, obj_height: dimensions of the manipulated object
        position_tolerance: max Euclidean distance from goal centre for success
        angle_tolerance: max absolute angular difference from goal angle for success (radians)
        max_timesteps: trial terminates after this many steps regardless of success
    """

    name: str = "rotate_insert"
    start_x: float = 0.0; start_y: float = 0.0; start_angle: float = 0.0
    goal_x: float = 0.5; goal_y: float = 0.5; goal_angle: float = 1.2
    obj_width: float = 0.3; obj_height: float = 0.6
    position_tolerance: float = 0.05
    angle_tolerance: float = 0.10
    max_timesteps: int = 120


TASKS = {
    "rotate_insert": TaskConfig(
        name="rotate_insert", start_x=0.0, start_y=0.0, start_angle=0.0,
        goal_x=0.5, goal_y=0.5, goal_angle=1.2,
        obj_width=0.3, obj_height=0.6, max_timesteps=120,
    ),
    "translate_only": TaskConfig(
        name="translate_only", start_x=0.0, start_y=0.0, start_angle=0.0,
        goal_x=0.6, goal_y=0.4, goal_angle=0.0,
        obj_width=0.4, obj_height=0.4, max_timesteps=120,
    ),
    "rotate_only": TaskConfig(
        name="rotate_only", start_x=0.5, start_y=0.5, start_angle=0.0,
        goal_x=0.5, goal_y=0.5, goal_angle=1.5,
        obj_width=0.3, obj_height=0.6, max_timesteps=120,
    ),
    "complex_manipulation": TaskConfig(
        name="complex_manipulation", start_x=-0.3, start_y=-0.2, start_angle=-0.5,
        goal_x=0.5, goal_y=0.6, goal_angle=1.0,
        obj_width=0.25, obj_height=0.7, max_timesteps=120,
    ),
}


# Timestep and trial records
@dataclass
class TimestepRecord:
    """
    State snapshot at one simulation timestep.

    Attributes:
        t: timestep index
        obj_x, obj_y, obj_angle: current object pose after movement
        gaze_target: "object" or "target" — what the eye is fixating
        movement_started: whether the information threshold has been reached
        obj_info: mean WM trace strength for object features [0:5]
        tgt_info: mean WM trace strength for target features [5:8]
        rel_info: mean WM trace strength for relational features [8:11]
        pos_error: Euclidean distance from object centre to goal centre
        angle_error: absolute angular distance from object angle to goal angle
        gaze_switches: cumulative number of fixation switches so far
    """
    t: int
    obj_x: float; obj_y: float; obj_angle: float
    gaze_target: str          # "object" or "target"
    movement_started: bool
    obj_info: float; tgt_info: float; rel_info: float
    pos_error: float; angle_error: float
    gaze_switches: int


@dataclass
class TrialResult:
    """
    Complete results from one trial.

    Attributes:
        params_name: name of the DevelopmentalParams config.
        task_name: name of the TaskConfig.
        trial_id: index of this trial within a batch.
        success: True if object was fitted into the target slot within tolerance.
        timesteps_used: number of timesteps before trial ended.
        final_pos_error: Euclidean position error at trial end.
        final_angle_error: absolute angle error at trial end.
        trajectory: list of TimestepRecord, one per timestep.
        movement_onset: timestep when motor output first occurred.
        rotation_onset: timestep of first significant rotation command.
        translation_onset: timestep of first significant translation command.
        efficiency: ratio of optimal straight-line distance to actual path length (0-1).
        total_gaze_switches: total object-to-target fixation switches during trial.
        object_fixation_pct: fraction of timesteps spent fixating the object.
        target_fixation_pct: fraction of timesteps spent fixating the target.
        gaze_history: full sequence of "object"/"target" strings, one per timestep.
    """
    params_name: str; task_name: str; trial_id: int
    success: bool; timesteps_used: int
    final_pos_error: float; final_angle_error: float
    trajectory: List[TimestepRecord]
    movement_onset: int; rotation_onset: int; translation_onset: int
    efficiency: float
    total_gaze_switches: int
    object_fixation_pct: float; target_fixation_pct: float
    gaze_history: List[str]
