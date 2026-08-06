"""Shared configuration for the three experiments."""

import math
from model_utils import DevelopmentalParams, TaskConfig

# Seeding
SEED_SETS = (42, 1_037, 20_461, 31_907, 58_243)
TRIALS_PER_SEED_SET = 30  # Experiment 1: 5 x 30 = 150 trials per cell
SWEEP_TRIALS_PER_SEED_SET = 20  # Experiment 2 primary sweep: 5 x 20 = 100
MATRIX_TRIALS_PER_SEED_SET = 20  # Experiment 3: 5 x 20 = 100 trials per cell

EXTENDED_SWEEP_VALUES = 5
EXTENDED_SWEEP_TRIALS = 20


# Developmental profiles
DEVELOPMENTAL_PROFILES = {
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

PROFILE_NAMES = list(DEVELOPMENTAL_PROFILES)
PROFILE_COLOURS = {"A": "#0584CD", "B": "#E76C1F", "C": "#7C0DC2", "D": "#6A6A6A"}


# Experiment 1 task battery
BATTERY_DISTANCES = (0.35, 0.70, 1.10)  # short / medium / far
BATTERY_ROTATIONS = (0.0, math.pi / 2, math.pi)  # 0 / 90 / 180 degrees
BATTERY_ASPECTS = (0.4, 1.0, 2.2)  # tall-narrow / square / wide-flat
BATTERY_BASE_HEIGHT = 0.4


def step_limit_for_distance(
    distance: float, intercept: float = 80.0, slope: float = 90.0
) -> int:
    """Step limit for a task at the given goal distance.

    Args:
        distance: goal distance for the task.
        intercept: fixed timestep allowance.
        slope: timesteps allowed per unit of distance.
    Returns:
        Maximum number of timesteps for the task.
    """

    return int(intercept + distance * slope)


def build_battery(
    position_tolerance: float = 0.05,
    angle_tolerance: float = 0.10,
    angular_symmetry: int | None = None,
    intercept: float = 80.0,
    slope: float = 90.0,
) -> tuple[dict, dict]:
    """
    Build the 27-task battery and its metadata.

    Args:
        position_tolerance: success tolerance on position, in model units.
        angle_tolerance: success tolerance on orientation, in radians.
        angular_symmetry: order of the object's rotational symmetry group; None
            treats objects as directional, which is the default used for the
            main results.
        intercept, slope: parameters of the step-limit formula.

    Returns:
        (tasks, metadata) dictionaries keyed by task name.
    """

    tasks, metadata = {}, {}
    for distance_index, distance in enumerate(BATTERY_DISTANCES):
        for rotation_index, rotation in enumerate(BATTERY_ROTATIONS):
            for aspect_index, aspect in enumerate(BATTERY_ASPECTS):
                name = f"d{distance_index}r{rotation_index}a{aspect_index}"
                tasks[name] = TaskConfig(
                    name=name,
                    start_x=0.0,
                    start_y=0.0,
                    start_angle=0.0,
                    goal_x=distance,
                    goal_y=0.0,
                    goal_angle=rotation,
                    obj_width=round(aspect * BATTERY_BASE_HEIGHT, 3),
                    obj_height=BATTERY_BASE_HEIGHT,
                    position_tolerance=position_tolerance,
                    angle_tolerance=angle_tolerance,
                    angular_symmetry=angular_symmetry,
                    max_timesteps=step_limit_for_distance(distance, intercept, slope),
                )
                metadata[name] = {
                    "dist": distance,
                    "rot": rotation,
                    "aspect": aspect,
                    "dist_idx": distance_index,
                    "rot_idx": rotation_index,
                    "aspect_idx": aspect_index,
                    # Elongation as distance from a square, on a scale that
                    # treats reciprocal ratios symmetrically.
                    "log_aspect": abs(math.log(aspect)),
                    "shape": {0: "tall", 1: "square", 2: "wide"}[aspect_index],
                }
    return tasks, metadata


BATTERY_TASKS, TASK_META = build_battery()

TOLERANCE_VARIANTS = {
    "strict": (0.03, 0.06),
    "default": (0.05, 0.10),
    "lenient": (0.08, 0.16),
}
STEP_LIMIT_VARIANTS = {
    "short": (60.0, 60.0),
    "default": (80.0, 90.0),
    "long": (120.0, 150.0),
}


# Experiment 2: baselines, tasks and sweep ranges
SWEEP_BASELINES = {
    "immature": DevelopmentalParams(
        name="baseline_immature",
        gaze_switch_rate=0.20,
        fixation_duration_mean=4.0,
        target_bias=0.30,
        simultaneous_rate=0.10,
        sampling_rate=0.40,
        perceptual_noise=0.30,
        location_acuity=0.78,
        orientation_acuity=0.25,
        relation_acuity=0.10,
        wm_capacity=2,
        wm_decay=0.15,
        wm_unfixated_decay=0.34,
        affordance_coupling=0.35,
        affordance_noise=0.22,
        planning_horizon=2,
        motor_noise=0.21,
        habit_strength=0.72,
        goal_directed_strength=0.28,
        correction_rate=0.13,
        correction_delay=2,
        initiation_threshold=0.22,
    ),
    "mid": DevelopmentalParams(
        name="baseline_mid",
        gaze_switch_rate=0.35,
        fixation_duration_mean=3.5,
        target_bias=0.40,
        simultaneous_rate=0.20,
        sampling_rate=0.55,
        perceptual_noise=0.20,
        location_acuity=0.85,
        orientation_acuity=0.45,
        relation_acuity=0.20,
        wm_capacity=3,
        wm_decay=0.10,
        wm_unfixated_decay=0.25,
        affordance_coupling=0.50,
        affordance_noise=0.18,
        planning_horizon=3,
        motor_noise=0.18,
        habit_strength=0.50,
        goal_directed_strength=0.50,
        correction_rate=0.18,
        correction_delay=1,
        initiation_threshold=0.28,
    ),
    "mature": DevelopmentalParams(
        name="baseline_mature",
        gaze_switch_rate=0.48,
        fixation_duration_mean=2.2,
        target_bias=0.55,
        simultaneous_rate=0.38,
        sampling_rate=0.75,
        perceptual_noise=0.12,
        location_acuity=0.95,
        orientation_acuity=0.70,
        relation_acuity=0.42,
        wm_capacity=4,
        wm_decay=0.06,
        wm_unfixated_decay=0.14,
        affordance_coupling=0.72,
        affordance_noise=0.11,
        planning_horizon=5,
        motor_noise=0.12,
        habit_strength=0.25,
        goal_directed_strength=0.75,
        correction_rate=0.25,
        correction_delay=1,
        initiation_threshold=0.42,
    ),
}

PRIMARY_BASELINE = "mid"
PRIMARY_SWEEP_TASK = "hard_balanced"

SWEEP_TASKS = {
    # The original sweep task: far, large rotation, wide-flat object.
    "hard_balanced": TaskConfig(
        name="hard_balanced",
        start_x=0.0,
        start_y=0.0,
        start_angle=0.0,
        goal_x=0.90,
        goal_y=0.0,
        goal_angle=math.pi * (3 / 4),
        obj_width=0.60,
        obj_height=0.30,
        max_timesteps=180,
    ),
    # Distance: the object must be carried a long way but barely turned
    "translation_dominant": TaskConfig(
        name="translation_dominant",
        start_x=0.0,
        start_y=0.0,
        start_angle=0.0,
        goal_x=1.10,
        goal_y=0.0,
        goal_angle=0.20,
        obj_width=0.40,
        obj_height=0.40,
        max_timesteps=180,
    ),
    # Rotation: the object hardly moves but must be turned a long way
    "rotation_dominant": TaskConfig(
        name="rotation_dominant",
        start_x=0.0,
        start_y=0.0,
        start_angle=0.0,
        goal_x=0.35,
        goal_y=0.0,
        goal_angle=math.pi * (3 / 4),
        obj_width=0.60,
        obj_height=0.30,
        max_timesteps=180,
    ),
}


def linvals(low: float, high: float, n: int = 7) -> list[float]:
    """Evenly spaced sweep values, rounded for readable labels.

    Args:
        low: smallest value.
        high: largest value.
        n: number of values to generate.

    Returns:
        List of ``n`` values evenly spaced between ``low`` and ``high``.
    """

    if n == 1:
        return [round(low, 4)]
    step = (high - low) / (n - 1)
    return [round(low + step * index, 4) for index in range(n)]


def sweep_config(n_values: int = 7) -> dict:
    """Sweep ranges for the 20 free parameters.

    Habit strength and goal-directed strength are constrained to sum to one and
    are therefore treated as a single free parameter.

    Args:
        n_values: number of values to sample per parameter.

    Returns:
        Mapping from parameter name to its list of sweep values.
    """

    return {
        # Gaze
        "gaze_switch_rate": linvals(0.05, 0.70, n_values),
        "fixation_duration_mean": linvals(1.0, 10.0, n_values),
        "target_bias": linvals(0.05, 0.80, n_values),
        "simultaneous_rate": linvals(0.01, 0.60, n_values),
        # Perception
        "sampling_rate": linvals(0.10, 0.95, n_values),
        "perceptual_noise": linvals(0.01, 0.50, n_values),
        "location_acuity": linvals(0.30, 1.00, n_values),
        "orientation_acuity": linvals(0.05, 1.00, n_values),
        "relation_acuity": linvals(0.01, 0.80, n_values),
        # Working memory
        "wm_capacity": [1, 2, 3, 4, 5, 6, 7][:n_values],
        "wm_decay": linvals(0.02, 0.30, n_values),
        "wm_unfixated_decay": linvals(0.05, 0.55, n_values),
        # Affordance
        "affordance_coupling": linvals(0.10, 1.00, n_values),
        "affordance_noise": linvals(0.01, 0.45, n_values),
        # Motor and control
        "planning_horizon": [1, 2, 3, 4, 5, 6][:n_values],
        "motor_noise": linvals(0.01, 0.45, n_values),
        "habit_strength": linvals(0.05, 0.95, n_values),
        "correction_rate": linvals(0.02, 0.40, n_values),
        "correction_delay": [0, 1, 2, 3, 4, 5][:n_values],
        "initiation_threshold": linvals(0.05, 0.60, n_values),
    }


SWEEP_CONFIG = sweep_config(7)

PARAM_GROUPS = {
    "Gaze": [
        "gaze_switch_rate",
        "fixation_duration_mean",
        "target_bias",
        "simultaneous_rate",
    ],
    "Perception": [
        "sampling_rate",
        "perceptual_noise",
        "location_acuity",
        "orientation_acuity",
        "relation_acuity",
    ],
    "Working Memory": ["wm_capacity", "wm_decay", "wm_unfixated_decay"],
    "Affordance": ["affordance_coupling", "affordance_noise"],
    "Motor & Control": [
        "planning_horizon",
        "motor_noise",
        "habit_strength",
        "correction_rate",
        "correction_delay",
        "initiation_threshold",
    ],
}

# Metrics entering the relative sensitivity index.
SENSITIVITY_METRICS = [
    ("success_rate", "Success\nrate"),
    ("mean_efficiency", "Path\nefficiency"),
    ("mean_pos_error", "Pos\nerror"),
    ("mean_angle_error", "Angle\nerror"),
    ("mean_movement_onset", "Movement\nonset"),
    ("mean_gaze_switches", "Gaze\nswitches"),
    ("mean_target_fixation", "Target\nfixation"),
    ("translate_before_rotate_rate", "Trans\nbefore rot"),
]

# Experiment 3: affordance matrix task set
MATRIX_TASKS = {
    # Calibrated so that profile A is off floor here
    "very_easy": TaskConfig(
        name="very_easy",
        goal_x=0.30,
        goal_y=0.0,
        goal_angle=0.53,
        obj_width=0.40,
        obj_height=0.40,
        max_timesteps=120,
    ),
    "easy": TaskConfig(
        name="easy",
        goal_x=0.35,
        goal_y=0.0,
        goal_angle=0.60,
        obj_width=0.30,
        obj_height=0.45,
        max_timesteps=120,
    ),
    "medium_rotated": TaskConfig(
        name="medium_rotated",
        goal_x=0.50,
        goal_y=0.0,
        goal_angle=1.20,
        obj_width=0.30,
        obj_height=0.60,
        max_timesteps=120,
    ),
    "very_hard": TaskConfig(
        name="very_hard",
        goal_x=0.95,
        goal_y=0.0,
        goal_angle=2.60,
        obj_width=0.60,
        obj_height=0.30,
        max_timesteps=170,
    ),
    "extreme": TaskConfig(
        name="extreme",
        goal_x=1.10,
        goal_y=0.0,
        goal_angle=math.pi,
        obj_width=0.60,
        obj_height=0.30,
        max_timesteps=190,
    ),
    # Profile D cannot be brought off ceiling by just geometry so use step limit
    "time_pressured": TaskConfig(
        name="time_pressured",
        goal_x=1.10,
        goal_y=0.0,
        goal_angle=math.pi,
        obj_width=0.60,
        obj_height=0.30,
        max_timesteps=62,
    ),
}

# A cell is treated as informative about the affordance matrix only if its
# success rate lies away from both bounds
INFORMATIVE_BAND = (0.10, 0.90)
