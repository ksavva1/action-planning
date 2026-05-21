"""
Configuration dataclasses and preset constants for the planning cascade model.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class DevelopmentalParams:
    """
    All parameters governing one developmental stage of the model.

    Attributes:
        name: identifier for this configuration
        gaze_switch_rate: base probability of switching fixation per timestep (0-1)
        fixation_duration_mean: mean dwell timesteps before a switch becomes likely
        target_bias: probability of looking at the goal slot (rather than the object) when a switch occurs (0-1)
        simultaneous_rate: probability of extracting relational features when both entities have been recently fixated (0-1)
        sampling_rate: probability that each feature dimension is sampled on a given timestep (0-1)
        perceptual_noise: standard deviation of Gaussian noise added to sampled percepts
        location_acuity: noise reduction factor for position features (0-1, higher = clearer)
        orientation_acuity: noise reduction factor for angle features (0-1)
        relation_acuity: noise reduction factor for relational features (0-1)
        wm_capacity: maximum number of strong memory traces maintained simultaneously
        wm_decay: per-timestep decay rate for traces of the currently fixated entity (0-1)
        wm_unfixated_decay: faster decay rate for traces of the non-fixated entity (0-1)
        affordance_coupling: scaling factor applied to the percept-to-affordance weight matrix (0-1)
        affordance_noise: standard deviation of noise in affordance estimation
        planning_horizon: timesteps of motor lookahead (1 = purely reactive, 6 = anticipatory)
        motor_noise: standard deviation of execution noise added to motor commands
        habit_strength: weight of the habitual translate-first bias (0-1)
        goal_directed_strength: weight of goal-directed motor control (0-1)
        correction_rate: gain of the online error-correction signal (0-1)
        correction_delay: timesteps of processing lag before correction activates
        initiation_threshold: minimum mean WM strength required to begin movement (0-1)
    """

    name: str = "B"

    # Gaze control parameters.
    # Infants vary in how often and where they look during object manipulation.
    # Younger infants usually dwell on the object longer and switch
    # gaze infrequently, limiting access to information about the target slot.
    # Older infants switch more rapidly and show a stronger target bias, which
    # allows them to build up a richer and more current representation of both
    # entities simultaneously.
    gaze_switch_rate: float = 0.30
    fixation_duration_mean: float = 3.0
    target_bias: float = 0.40
    simultaneous_rate: float = 0.15

    # Perceptual parameters.
    # Perceptual acuity and sampling efficiency increase across development.
    # Orientation acuity is lower than location acuity because angle perception
    # is a later-developing skill — infants reliably code where things are before
    # they code how they are oriented. Relational acuity is lowest because spatial
    # comparison (computing the gap between object and goal) requires integrating
    # two separately fixated representations, a more demanding operation.
    sampling_rate: float = 0.40
    perceptual_noise: float = 0.30
    location_acuity: float = 0.85
    orientation_acuity: float = 0.40
    relation_acuity: float = 0.15

    # Working memory parameters.
    # WM capacity and decay rates determine how much task-relevant information
    # the agent can hold across gaze shifts. Faster decay in younger infants means
    # that information gathered during one fixation fades quickly, forcing more
    # frequent re-sampling via gaze. The unfixated-entity decay is always higher
    # than the fixated-entity decay, reflecting attentional prioritisation of the
    # currently viewed item.
    wm_capacity: int = 3
    wm_decay: float = 0.12
    wm_unfixated_decay: float = 0.28

    # Affordance parameters.
    # Affordance coupling controls how strongly WM representations activate
    # action possibilities (reach, grasp, rotate, translate). Weaker coupling
    # in younger infants means that action selection is less precisely tuned to
    # the current task demands — the mapping from perception to action is less
    # differentiated.
    affordance_coupling: float = 0.40
    affordance_noise: float = 0.25

    # Motor planning parameters.
    # Planning horizon captures how far ahead the motor system looks when
    # generating commands. A horizon of 1 produces purely reactive, stimulus-driven
    # movements. A horizon of 6 produces anticipatory, goal-corrected movements
    # characteristic of older infants who plan the whole action sequence before
    # beginning to move.
    planning_horizon: int = 2
    motor_noise: float = 0.25

    # Habit vs. goal-directed control balance.
    # Younger infants rely more heavily on habitual motor routines (e.g. translating
    # the object before rotating it) inherited from simpler tasks encountered earlier
    # in development. Older infants increasingly use goal-directed control to select
    # actions appropriate to the current task geometry, overriding prior habits.
    habit_strength: float = 0.55
    goal_directed_strength: float = 0.45

    # Online error correction parameters.
    # The correction rate and delay model the sensorimotor feedback loop.
    # A longer delay reflects immature neural conduction velocity and processing
    # speed: the infant continues moving based on outdated error information.
    # A shorter delay (more mature stages) allows faster mid-movement corrections.
    correction_rate: float = 0.12
    correction_delay: int = 2

    # Movement initiation threshold.
    # The agent will not begin moving until the mean WM trace strength reaches
    # this value. This operationalises the observation that infants pause before
    # acting, gathering information about the task before committing to a movement.
    # Higher thresholds (more mature stages) mean a longer information-gathering
    # phase but typically more accurate initial movements.
    initiation_threshold: float = 0.35


DEVELOPMENTAL_STAGES = {
    "A": DevelopmentalParams(
        name="A",
        gaze_switch_rate=0.12,
        fixation_duration_mean=5.0,
        target_bias=0.20,
        simultaneous_rate=0.03,
        sampling_rate=0.30,
        perceptual_noise=0.35,
        location_acuity=0.70,
        orientation_acuity=0.15,
        relation_acuity=0.05,
        wm_capacity=2,
        wm_decay=0.18,
        wm_unfixated_decay=0.40,
        affordance_coupling=0.25,
        affordance_noise=0.25,
        planning_horizon=1,
        motor_noise=0.20,
        habit_strength=0.78,
        goal_directed_strength=0.22,
        correction_rate=0.12,
        correction_delay=2,
        initiation_threshold=0.15,
    ),
    "B": DevelopmentalParams(
        name="B",
        gaze_switch_rate=0.25,
        fixation_duration_mean=3.5,
        target_bias=0.35,
        simultaneous_rate=0.12,
        sampling_rate=0.45,
        perceptual_noise=0.28,
        location_acuity=0.85,
        orientation_acuity=0.30,
        relation_acuity=0.12,
        wm_capacity=3,
        wm_decay=0.12,
        wm_unfixated_decay=0.28,
        affordance_coupling=0.40,
        affordance_noise=0.22,
        planning_horizon=2,
        motor_noise=0.22,
        habit_strength=0.70,
        goal_directed_strength=0.30,
        correction_rate=0.14,
        correction_delay=2,
        initiation_threshold=0.28,
    ),
    "C": DevelopmentalParams(
        name="C",
        gaze_switch_rate=0.40,
        fixation_duration_mean=2.5,
        target_bias=0.50,
        simultaneous_rate=0.28,
        sampling_rate=0.65,
        perceptual_noise=0.16,
        location_acuity=0.93,
        orientation_acuity=0.55,
        relation_acuity=0.30,
        wm_capacity=4,
        wm_decay=0.08,
        wm_unfixated_decay=0.18,
        affordance_coupling=0.60,
        affordance_noise=0.14,
        planning_horizon=4,
        motor_noise=0.16,
        habit_strength=0.40,
        goal_directed_strength=0.60,
        correction_rate=0.22,
        correction_delay=1,
        initiation_threshold=0.40,
    ),
    "D": DevelopmentalParams(
        name="D",
        gaze_switch_rate=0.55,
        fixation_duration_mean=2.0,
        target_bias=0.60,
        simultaneous_rate=0.50,
        sampling_rate=0.85,
        perceptual_noise=0.08,
        location_acuity=0.98,
        orientation_acuity=0.88,
        relation_acuity=0.55,
        wm_capacity=5,
        wm_decay=0.04,
        wm_unfixated_decay=0.10,
        affordance_coupling=0.85,
        affordance_noise=0.08,
        planning_horizon=6,
        motor_noise=0.08,
        habit_strength=0.10,
        goal_directed_strength=0.90,
        correction_rate=0.28,
        correction_delay=0,
        initiation_threshold=0.45,
    ),
}


@dataclass
class TaskConfig:
    """
    Defines an object manipulation task with start and goal poses and success criteria.

    Attributes:
        name: identifier string for this task
        start_x, start_y, start_angle: initial object pose
        goal_x, goal_y, goal_angle: target slot pose the object must reach
        obj_width, obj_height: dimensions of the manipulated object
        position_tolerance: maximum Euclidean distance from goal centre for success
        angle_tolerance: maximum absolute angular difference from goal angle for success (radians)
        max_timesteps: trial terminates after this many steps regardless of success
    """

    name: str = "rotate_insert"
    start_x: float = 0.0
    start_y: float = 0.0
    start_angle: float = 0.0
    goal_x: float = 0.5
    goal_y: float = 0.5
    goal_angle: float = 1.2
    obj_width: float = 0.3
    obj_height: float = 0.6
    position_tolerance: float = 0.05
    angle_tolerance: float = 0.10
    max_timesteps: int = 120


TASKS = {
    "rotate_insert": TaskConfig(
        name="rotate_insert",
        start_x=0.0, start_y=0.0, start_angle=0.0,
        goal_x=0.5, goal_y=0.5, goal_angle=1.2,
        obj_width=0.3, obj_height=0.6,
        max_timesteps=120,
    ),
    "translate_only": TaskConfig(
        name="translate_only",
        start_x=0.0, start_y=0.0, start_angle=0.0,
        goal_x=0.6, goal_y=0.4, goal_angle=0.0,
        obj_width=0.4, obj_height=0.4,
        max_timesteps=120,
    ),
    "rotate_only": TaskConfig(
        name="rotate_only",
        start_x=0.5, start_y=0.5, start_angle=0.0,
        goal_x=0.5, goal_y=0.5, goal_angle=1.5,
        obj_width=0.3, obj_height=0.6,
        max_timesteps=120,
    ),
    "complex_manipulation": TaskConfig(
        name="complex_manipulation",
        start_x=-0.3, start_y=-0.2, start_angle=-0.5,
        goal_x=0.5, goal_y=0.6, goal_angle=1.0,
        obj_width=0.25, obj_height=0.7,
        max_timesteps=120,
    ),
}


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
    obj_x: float
    obj_y: float
    obj_angle: float
    gaze_target: str
    movement_started: bool
    obj_info: float
    tgt_info: float
    rel_info: float
    pos_error: float
    angle_error: float
    gaze_switches: int


@dataclass
class TrialResult:
    """
    Complete results from one simulated trial.

    Attributes:
        params_name: name of the DevelopmentalParams config used
        task_name: name of the TaskConfig used
        trial_id: index of this trial within a batch
        success: True if the object was fitted into the target slot within tolerance
        timesteps_used: number of timesteps before the trial ended
        final_pos_error: Euclidean position error at trial end
        final_angle_error: absolute angle error at trial end
        trajectory: list of TimestepRecord, one per timestep
        movement_onset: timestep when motor output first occurred
        rotation_onset: timestep of first significant rotation command
        translation_onset: timestep of first significant translation command
        efficiency: ratio of optimal straight-line distance to actual path length (0-1)
        total_gaze_switches: total object-to-target fixation switches during trial
        object_fixation_pct: fraction of timesteps spent fixating the object
        target_fixation_pct: fraction of timesteps spent fixating the target
        gaze_history: full sequence of "object"/"target" strings, one per timestep
    """

    params_name: str
    task_name: str
    trial_id: int
    success: bool
    timesteps_used: int
    final_pos_error: float
    final_angle_error: float
    trajectory: List[TimestepRecord]
    movement_onset: int
    rotation_onset: int
    translation_onset: int
    efficiency: float
    total_gaze_switches: int
    object_fixation_pct: float
    target_fixation_pct: float
    gaze_history: List[str]
