"""
Planning Model — state containers, layer functions, and trial runner.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_config import (
    DevelopmentalParams, DEVELOPMENTAL_STAGES,
    TaskConfig, TASKS,
    TimestepRecord, TrialResult,
)

__all__ = [
    "DevelopmentalParams", "DEVELOPMENTAL_STAGES",
    "TaskConfig", "TASKS",
    "TimestepRecord", "TrialResult",
    "GazeState", "WorkingMemoryState", "AffordanceWeights",
    "MotorWeights", "HabitState", "CorrectionState",
    "run_trial",
]


# ─────────────────────────────────────────────────────────────────────────────
# State containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GazeState:
    """Mutable state for the gaze controller. Reset at the start of each trial."""
    current_fixation: str = "object"
    dwell_time: int = 0
    switch_count: int = 0
    object_fixation_count: int = 0
    target_fixation_count: int = 0
    fixation_history: List[str] = field(default_factory=list)


@dataclass
class WorkingMemoryState:
    """Mutable state for working memory. Reset at the start of each trial."""
    memory_buffer: np.ndarray = field(default_factory=lambda: np.zeros(11))
    trace_strength: np.ndarray = field(default_factory=lambda: np.zeros(11))


@dataclass
class AffordanceWeights:
    """Immutable affordance weight matrix, initialised once per trial with random noise."""
    weight_matrix: np.ndarray   # shape (11, 4)
    action_bias: np.ndarray     # shape (4,)


@dataclass
class MotorWeights:
    """Immutable motor weight matrix, initialised once per trial with random noise."""
    weight_matrix: np.ndarray   # shape (4, 4)


@dataclass
class HabitState:
    """Mutable state for the habit layer. Reset at the start of each trial."""
    step_count: int = 0


@dataclass
class CorrectionState:
    """Mutable state for the correction layer. Reset at the start of each trial."""
    error_history: List = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Weight initialisation
# ─────────────────────────────────────────────────────────────────────────────

def build_affordance_weights(params: DevelopmentalParams, rng: np.random.Generator) -> AffordanceWeights:
    """
    Build the affordance weight matrix, scaled by affordance_coupling and jittered by noise.

    The weight structure encodes which WM features drive which actions:
    object/target location → reach and translate; angle features → rotate;
    object size → grasp; relational features (rows 8–10) → translate and rotate
    most strongly, since they encode the gap between current and goal configuration.
    Less mature agents (lower coupling) have a weaker, noisier perception-to-action mapping.
    """
    weight_matrix = np.zeros((11, 4))
    weight_matrix[0, :] = [.6, .1,  0, .5]   # object x → reach, translate
    weight_matrix[1, :] = [.6, .1,  0, .5]   # object y → reach, translate
    weight_matrix[2, :] = [ 0, .2, .7,  0]   # object angle → rotate
    weight_matrix[3, :] = [.1, .6, .3, .1]   # object width → grasp
    weight_matrix[4, :] = [.1, .6, .3, .1]   # object height → grasp
    weight_matrix[5, :] = [.3,  0,  0, .4]   # target x → reach, translate
    weight_matrix[6, :] = [.3,  0,  0, .4]   # target y → reach, translate
    weight_matrix[7, :] = [ 0, .1, .5,  0]   # target angle → rotate
    weight_matrix[8, :] = [.2,  0, .1, .8]   # rel dx → translate (dominant)
    weight_matrix[9, :] = [.2,  0, .1, .8]   # rel dy → translate (dominant)
    weight_matrix[10, :] = [0, .1, .9,  0]   # rel d_angle → rotate (dominant)

    scaled_matrix = np.clip(
        weight_matrix * params.affordance_coupling + rng.normal(0, .03, weight_matrix.shape),
        0, 1,
    )
    return AffordanceWeights(weight_matrix=scaled_matrix, action_bias=np.array([.15, .1, .05, .2]))


def build_motor_weights(params: DevelopmentalParams, rng: np.random.Generator) -> MotorWeights:
    """
    Build the motor weight matrix with small initialisation noise.

    The near-diagonal structure maps each affordance primarily to its corresponding
    motor dimension, with small cross-coupling between reach and translate.
    """
    weight_matrix = (
        np.array([[.9, 0, 0, 0], [0, .9, 0, 0], [0, 0, .9, 0], [.5, .5, 0, 0]])
        + rng.normal(0, .02, (4, 4))
    )
    return MotorWeights(weight_matrix=weight_matrix)


# ─────────────────────────────────────────────────────────────────────────────
# Layer functions
# ─────────────────────────────────────────────────────────────────────────────

def step_gaze(gaze_state: GazeState, params: DevelopmentalParams, rng: np.random.Generator) -> str:
    """
    Advance the gaze controller by one timestep, possibly switching fixation.

    The hazard function increases switch probability with dwell time — short fixations
    are unlikely to trigger a switch; long ones become increasingly likely to, modelling
    the tendency for gaze to remain on an entity until it has been sufficiently processed.
    """
    gaze_state.dwell_time += 1
    switch_hazard = 1.0 - np.exp(-gaze_state.dwell_time / params.fixation_duration_mean)
    switch_occurred = rng.random() < params.gaze_switch_rate * switch_hazard

    if switch_occurred:
        if gaze_state.current_fixation == "object":
            # When leaving the object, always move to the target slot.
            # The object is the primary manipulation target, so the only meaningful
            # "choice" is whether to go check the goal slot.
            gaze_state.current_fixation = "target"
        else:
            # When leaving the target slot, target_bias controls where gaze lands next.
            # A higher target_bias means more likely to return to (or remain at) the target,
            # reflecting stronger goal-directed attention in more mature stages.
            if rng.random() < (1 - params.target_bias):
                gaze_state.current_fixation = "object"
            else:
                gaze_state.current_fixation = "target"

        gaze_state.dwell_time = 0
        gaze_state.switch_count += 1

    gaze_state.fixation_history.append(gaze_state.current_fixation)

    if gaze_state.current_fixation == "object":
        gaze_state.object_fixation_count += 1
    else:
        gaze_state.target_fixation_count += 1

    return gaze_state.current_fixation


def sample_percept(
    params: DevelopmentalParams,
    object_state: np.ndarray,
    target_state: np.ndarray,
    gaze_target: str,
    both_recently_fixated: bool,
    rng: np.random.Generator,
) -> tuple:
    """
    Sample noisy perceptual features from whichever entity is currently fixated.

    11-dim feature vector: [0:5] object features, [5:8] target features, [8:11] relational.
    Only the fixated entity's features are sampled. Relational features require both entities
    to have been recently active in working memory.

    Acuity reduces effective noise per feature type:
    location acuity is highest (spatial position is well coded early in development),
    orientation acuity is intermediate, and relational acuity is lowest because comparing
    two separately fixated representations is a later-developing skill.
    """
    percept = np.zeros(11)
    sampled_mask = np.zeros(11)

    object_acuity = np.array([params.location_acuity] * 2 + [params.orientation_acuity] * 3)
    target_acuity = np.array([params.location_acuity] * 2 + [params.orientation_acuity])
    relational_acuity = np.array([params.relation_acuity] * 3)

    if gaze_target == "object":
        features_sampled = rng.random(5) < params.sampling_rate
        perceptual_noise = rng.normal(0, params.perceptual_noise, 5) * (1 - object_acuity)
        percept[:5] = (object_state[:5] + perceptual_noise) * features_sampled
        sampled_mask[:5] = features_sampled.astype(float)
    else:
        features_sampled = rng.random(3) < params.sampling_rate
        perceptual_noise = rng.normal(0, params.perceptual_noise, 3) * (1 - target_acuity)
        percept[5:8] = (target_state[:3] + perceptual_noise) * features_sampled
        sampled_mask[5:8] = features_sampled.astype(float)

    # Relational features encode the gap between current object pose and goal pose.
    # They are only accessible when both entities are active in working memory,
    # modelling the cognitive cost of comparing two separately fixated representations.
    if both_recently_fixated and rng.random() < params.simultaneous_rate:
        relational_values = np.array([
            target_state[0] - object_state[0],
            target_state[1] - object_state[1],
            target_state[2] - object_state[2],
        ])
        features_sampled = rng.random(3) < params.sampling_rate
        perceptual_noise = rng.normal(0, params.perceptual_noise, 3) * (1 - relational_acuity)
        percept[8:11] = (relational_values + perceptual_noise) * features_sampled
        sampled_mask[8:11] = features_sampled.astype(float)

    return percept, sampled_mask


def update_wm(
    wm_state: WorkingMemoryState,
    params: DevelopmentalParams,
    percept: np.ndarray,
    sampled_mask: np.ndarray,
    gaze_target: str,
) -> np.ndarray:
    """
    Integrate a new percept into working memory with differential decay and a capacity limit.

    Features of the currently fixated entity decay slowly; the other entity's features decay
    faster. Relational features always decay at the faster rate because they require simultaneous
    coding of both entities — any gaze shift disrupts their maintenance.

    When a trace already exists for a feature, the new percept is blended with the stored value
    rather than replacing it — a form of Bayesian updating where existing memory partially anchors
    the new estimate, reducing sensitivity to momentary noise.

    When more feature traces are active than the capacity allows, the least-active ones are
    weakened, modelling competition between memory slots.
    """
    decay_rates = np.full(11, params.wm_decay)
    if gaze_target == "object":
        decay_rates[5:8] = params.wm_unfixated_decay
    else:
        decay_rates[:5] = params.wm_unfixated_decay

    decay_rates[8:11] = params.wm_unfixated_decay
    wm_state.trace_strength *= (1 - decay_rates)

    for feature_index in range(11):
        if sampled_mask[feature_index] > 0.5:
            if wm_state.trace_strength[feature_index] > 0.1:
                wm_state.memory_buffer[feature_index] = (
                    0.4 * wm_state.memory_buffer[feature_index]
                    + 0.6 * percept[feature_index]
                )
            else:
                wm_state.memory_buffer[feature_index] = percept[feature_index]

            wm_state.trace_strength[feature_index] = min(1.0, wm_state.trace_strength[feature_index] + 0.5)

    num_active_traces = np.sum(wm_state.trace_strength > 0.1)
    if num_active_traces > params.wm_capacity:
        capacity_threshold = np.sort(wm_state.trace_strength)[::-1][params.wm_capacity]
        wm_state.trace_strength[wm_state.trace_strength < capacity_threshold] *= 0.3

    return wm_state.memory_buffer * wm_state.trace_strength


def estimate_affordances(
    affordance_weights: AffordanceWeights,
    params: DevelopmentalParams,
    working_memory: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Map working memory state onto affordance activations [reach, grasp, rotate, translate].

    Sigmoid activation with a steep slope around 0.3 creates a soft threshold:
    weak WM signals produce near-zero affordances, strong signals near-one affordances.
    """
    weighted_activations = np.dot(working_memory, affordance_weights.weight_matrix) + affordance_weights.action_bias * 0.1
    noisy_activations = weighted_activations + rng.normal(0, params.affordance_noise, 4)
    
    return 1 / (1 + np.exp(-5 * (noisy_activations - 0.3)))


def plan_motor_command(
    motor_weights: MotorWeights,
    params: DevelopmentalParams,
    affordances: np.ndarray,
    goal_motor_state: np.ndarray,
    current_motor_state: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a motor command by blending affordance-driven and goal-error-driven contributions.

    planning_weight scales from 0 (pure affordance, reactive) to 1 (pure goal-error, anticipatory).
    Higher planning horizons produce commands more closely aligned with the goal, modelling the
    anticipatory control characteristic of older infants. Deceleration near the goal prevents overshoot.
    """
    affordance_command = np.dot(affordances, motor_weights.weight_matrix)
    goal_error = goal_motor_state[:4] - current_motor_state[:4]
    planning_weight = min(1.0, params.planning_horizon / 6.0)

    raw_command = (
        (1 - planning_weight) * affordance_command
        + planning_weight * goal_error * 0.3
        + rng.normal(0, params.motor_noise, 4)
    )

    goal_distance = np.sqrt(np.sum(goal_error[:3] ** 2))
    deceleration_factor = min(1.0, goal_distance / 0.5)

    return np.clip(raw_command * deceleration_factor, -1, 1)


def blend_habit(
    habit_state: HabitState,
    params: DevelopmentalParams,
    goal_directed_command: np.ndarray,
) -> np.ndarray:
    """
    Blend a habitual translate-first motor bias with the goal-directed command.

    Phase 1: strong translation, no rotation — the habitual "push" move.
    Phase 2: strong rotation, no translation — correcting orientation after placement.
    This two-phase pattern mirrors the translate-then-rotate sequence observed in younger
    infants during shape-sorter and object-insertion tasks.

    The habitual weight fades to zero over 3× phase_length steps, transferring its share to
    the goal-directed component so the two weights always sum to 1. The habit does not simply
    drop out — it is replaced by goal-directed control, modelling the gradual developmental
    transition from habitual to flexible action selection.
    """
    habit_state.step_count += 1
    phase_length = int(4 + 12 * params.habit_strength)

    if habit_state.step_count < phase_length:
        habitual_command = np.array([0.7, 0.7, 0.0, 0.2])
    else:
        habitual_command = np.array([0.0, 0.0, 0.8, 0.5])

    habit_fade_factor = max(0.0, 1.0 - habit_state.step_count / (phase_length * 3))
    effective_habit_weight = params.habit_strength * habit_fade_factor
    effective_goal_weight = params.goal_directed_strength + params.habit_strength * (1 - habit_fade_factor)
    total_weight = effective_habit_weight + effective_goal_weight

    return (effective_habit_weight / total_weight) * habitual_command + (effective_goal_weight / total_weight) * goal_directed_command


def apply_correction(
    correction_state: CorrectionState,
    params: DevelopmentalParams,
    current_motor_state: np.ndarray,
    goal_motor_state: np.ndarray,
    current_time: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute a corrective motor adjustment based on a delayed error signal.

    Uses the error from correction_delay timesteps ago, modelling the neural conduction
    and processing lag in sensorimotor feedback loops. The infant is always correcting for
    where the object was, not where it is now. A shorter delay (more mature) allows faster
    mid-movement corrections. Correction is scaled down near the goal to prevent oscillation.
    """
    current_error = goal_motor_state[:4] - current_motor_state[:4]
    correction_state.error_history.append(current_error.copy())

    if current_time < params.correction_delay:
        return np.zeros(4)

    delayed_error_index = max(0, len(correction_state.error_history) - 1 - params.correction_delay)
    delayed_error = correction_state.error_history[delayed_error_index]

    correction = (
        params.correction_rate * delayed_error
        + rng.normal(0, params.motor_noise * 0.5, 4)
    )

    distance_to_goal = np.sqrt(np.sum(current_error[:3] ** 2))
    deceleration_factor = min(1.0, distance_to_goal / 0.4)

    return correction * deceleration_factor


# ─────────────────────────────────────────────────────────────────────────────
# Trial runner helpers
# ─────────────────────────────────────────────────────────────────────────────

def both_entities_recently_fixated(current_time, last_object_fixation_time, last_target_fixation_time):
    """
    Return True if both object and target have been fixated within the last 3 timesteps.

    Relational features can only be extracted when both representations are simultaneously
    warm in working memory. The 3-timestep window operationalises the idea that perceptual
    binding of spatially separated entities requires both to still be recently active.
    """
    object_seen_recently = (current_time - last_object_fixation_time) <= 3
    target_seen_recently = (current_time - last_target_fixation_time) <= 3
    return object_seen_recently and target_seen_recently


def extract_wm_info_strengths(wm_strengths):
    """
    Return mean WM trace strength for object, target, and relational feature groups.

    These three averages track knowledge build-up over a trial. Object and target information
    accumulate separately; relational information lags behind both, since it requires
    simultaneous activation of both representations.
    """
    object_info = float(np.mean(wm_strengths[:5]))
    target_info = float(np.mean(wm_strengths[5:8]))
    relational_info = float(np.mean(wm_strengths[8:11]))
    return object_info, target_info, relational_info


def compute_near_goal_command(raw_command, object_x, object_y, object_angle, task):
    """
    Blend the raw motor command with a direct error signal as the object nears the goal.

    Within 0.6 units of the target, a proportional direct-error term is mixed in with
    increasing weight, modelling the shift from ballistic to fine-grained control in the
    final phase of insertion. Speed is also reduced near the goal to prevent overshoot.
    """
    dx = task.goal_x - object_x
    dy = task.goal_y - object_y
    da = task.goal_angle - object_angle
    distance_to_goal = np.sqrt(dx ** 2 + dy ** 2 + da ** 2)

    if distance_to_goal < 0.6:
        closeness = 1.0 - min(distance_to_goal / 0.6, 1.0)
        direct_error_command = np.array([dx, dy, da, 0]) * 2.0
        raw_command = raw_command * (1 - closeness) + direct_error_command * closeness

    speed_scale = min(1.0, distance_to_goal * 2.5)
    return np.clip(raw_command * speed_scale, -1, 1)


def apply_motor_command(object_x, object_y, object_angle, final_command):
    """
    Advance the object pose by applying the final motor command.

    Step size 0.10 sets the maximum movement per timestep — small enough that the
    correction layer has a meaningful opportunity to act between successive frames.
    """
    step_size = 0.10
    new_x = object_x + final_command[0] * step_size
    new_y = object_y + final_command[1] * step_size
    new_angle = object_angle + final_command[2] * step_size
    return new_x, new_y, new_angle


def update_onset_times(final_command, movement_started, current_time, rotation_onset, translation_onset, max_timesteps):
    """
    Record the first timestep at which significant rotation and translation occur.

    The translate-before-rotate measure is a key behavioural index: younger agents tend to move
    the object laterally before rotating it, even when rotation is required first. These onset
    times capture that ordering across trials.
    """
    if not movement_started:
        return rotation_onset, translation_onset

    if abs(final_command[2]) > 0.1 and rotation_onset == max_timesteps:
        rotation_onset = current_time
    if (abs(final_command[0]) > 0.1 or abs(final_command[1]) > 0.1) and translation_onset == max_timesteps:
        translation_onset = current_time

    return rotation_onset, translation_onset


def compute_path_efficiency(trajectory, task):
    """
    Compute the ratio of optimal straight-line path length to actual path length.

    Efficiency of 1.0 means the object followed a direct path from start to goal.
    Lower values indicate redundant movement. More mature agents with longer planning
    horizons and weaker habitual biases tend to be more efficient.
    """
    optimal_path_length = np.sqrt(
        (task.goal_x - task.start_x) ** 2
        + (task.goal_y - task.start_y) ** 2
        + (task.goal_angle - task.start_angle) ** 2
    )

    actual_path_length = 0.0
    for i in range(1, len(trajectory)):
        step_distance = np.sqrt(
            (trajectory[i].obj_x - trajectory[i - 1].obj_x) ** 2
            + (trajectory[i].obj_y - trajectory[i - 1].obj_y) ** 2
            + (trajectory[i].obj_angle - trajectory[i - 1].obj_angle) ** 2
        )
        actual_path_length += step_distance

    return min(optimal_path_length / max(actual_path_length, 0.01), 1.0)


# Trial runner
# ─────────────────────────────────────────────────────────────────────────────

def run_trial(params: DevelopmentalParams, task: TaskConfig, seed: int = 42, trial_id: int = 0) -> TrialResult:
    """
    Simulate one trial of fitting the object into the target slot.

    The trial loop runs the full processing cascade each timestep:
    gaze → perceive → WM update → initiation check → affordances →
    motor plan → habitual blend → correction → execute → success check.

    The rng is shared between weight initialisation and the trial loop so that each
    (params, seed) combination produces a unique but fully reproducible trajectory.

    Args:
        params: defines this trial's cognitive profile (developmental stage)
        task: defines start/goal poses and tolerances
        seed: random seed for reproducibility
        trial_id: index for this trial (used in output only)

    Returns:
        TrialResult containing success flag, trajectory, timing metrics,
        gaze statistics, and movement efficiency.
    """
    # The rng is created here and shared with weight building so the entire random
    # sequence — weight jitter plus trial noise — is determined by a single seed.
    rng = np.random.default_rng(seed)
    affordance_weights = build_affordance_weights(params, rng)
    motor_weights = build_motor_weights(params, rng)

    gaze_state = GazeState()
    wm_state = WorkingMemoryState()
    habit_state = HabitState()
    correction_state = CorrectionState()

    object_x = task.start_x
    object_y = task.start_y
    object_angle = task.start_angle
    target_state = np.array([task.goal_x, task.goal_y, task.goal_angle])
    goal_motor_state = np.array([task.goal_x, task.goal_y, task.goal_angle, 0.5])

    trajectory = []
    movement_started = False
    movement_onset = task.max_timesteps
    rotation_onset = task.max_timesteps
    translation_onset = task.max_timesteps
    last_object_fixation_time = -10
    last_target_fixation_time = -10

    for current_time in range(task.max_timesteps):
        # Compute positional and angular error before this timestep's movement.
        position_error = np.sqrt(
            (object_x - task.goal_x) ** 2 + (object_y - task.goal_y) ** 2
        )
        angle_error = abs(object_angle - task.goal_angle)

        # Step 1: Gaze — update fixation target, possibly switching.
        gaze_target = step_gaze(gaze_state, params, rng)
        if gaze_target == "object":
            last_object_fixation_time = current_time
        else:
            last_target_fixation_time = current_time

        both_fixated = both_entities_recently_fixated(
            current_time, last_object_fixation_time, last_target_fixation_time
        )

        # Step 2: Perceive — sample noisy features from the fixated entity.
        object_state = np.array([object_x, object_y, object_angle, task.obj_width, task.obj_height])
        percept, sampled_mask = sample_percept(
            params, object_state, target_state, gaze_target, both_fixated, rng
        )

        # Step 3: Working memory — integrate the percept with decay and capacity limits.
        working_memory = update_wm(wm_state, params, percept, sampled_mask, gaze_target)
        wm_strengths = wm_state.trace_strength.copy()
        object_info, target_info, relational_info = extract_wm_info_strengths(wm_strengths)

        # Step 4: Initiation check — begin moving once WM strength is sufficient.
        # The infant does not act until it has gathered enough information about both
        # the object and the target to guide a purposeful movement.
        if not movement_started and np.mean(wm_strengths) >= params.initiation_threshold:
            movement_started = True
            movement_onset = current_time

        # Step 5: Affordances — map WM state onto action possibilities.
        affordances = estimate_affordances(affordance_weights, params, working_memory, rng)

        # Steps 6-8: Build the motor command from three contributions.
        
        # Motor plan (step 6): converts affordance activations and goal-error into a raw command.
        # planning_horizon controls how much the command leans toward directly tracking goal error
        # (anticipatory) versus following learned affordances (reactive).
        
        # Habitual blend (step 7): overlays a translate-first bias on the motor plan.
        # The habitual influence fades over the course of the trial, gradually handing control
        # back to the goal-directed signal.
        
        # Online correction (step 8): adds a delayed corrective signal modelling the processing
        # lag in sensorimotor feedback loops.
        current_motor_state = np.array([object_x, object_y, object_angle, 0.3])
        goal_directed_command = plan_motor_command(motor_weights, params, affordances, goal_motor_state, current_motor_state, rng)
        blended_command = blend_habit(habit_state, params, goal_directed_command)
        correction_command = apply_correction(correction_state, params, current_motor_state, goal_motor_state, current_time, rng)

        # Step 9: Produce the final command.
        # When movement has started, apply near-goal deceleration and blend in a direct error
        # signal for fine positional control. Before initiation, the command is zero.
        if movement_started:
            raw_command = blended_command + correction_command
            final_command = compute_near_goal_command(raw_command, object_x, object_y, object_angle, task)
        else:
            final_command = np.zeros(4)

        # Step 10: Execute — update the object pose.
        object_x, object_y, object_angle = apply_motor_command(object_x, object_y, object_angle, final_command)
        rotation_onset, translation_onset = update_onset_times(
            final_command, movement_started, current_time,
            rotation_onset, translation_onset, task.max_timesteps,
        )

        trajectory.append(TimestepRecord(
            t=current_time,
            obj_x=object_x,
            obj_y=object_y,
            obj_angle=object_angle,
            gaze_target=gaze_target,
            movement_started=movement_started,
            obj_info=object_info,
            tgt_info=target_info,
            rel_info=relational_info,
            pos_error=position_error,
            angle_error=angle_error,
            gaze_switches=gaze_state.switch_count,
        ))

        # Step 11: Check success & stop early if the object is within tolerance.
        if position_error < task.position_tolerance and angle_error < task.angle_tolerance:
            break

    # Compute final metrics.
    final_position_error = np.sqrt(
        (object_x - task.goal_x) ** 2 + (object_y - task.goal_y) ** 2
    )
    final_angle_error = abs(object_angle - task.goal_angle)
    success = (
        final_position_error < task.position_tolerance
        and final_angle_error < task.angle_tolerance
    )
    efficiency = compute_path_efficiency(trajectory, task)
    total_fixations = len(gaze_state.fixation_history)

    return TrialResult(
        params_name=params.name,
        task_name=task.name,
        trial_id=trial_id,
        success=success,
        timesteps_used=len(trajectory),
        final_pos_error=final_position_error,
        final_angle_error=final_angle_error,
        trajectory=trajectory,
        movement_onset=movement_onset,
        rotation_onset=rotation_onset,
        translation_onset=translation_onset,
        efficiency=efficiency,
        total_gaze_switches=gaze_state.switch_count,
        object_fixation_pct=gaze_state.object_fixation_count / max(total_fixations, 1),
        target_fixation_pct=gaze_state.target_fixation_count / max(total_fixations, 1),
        gaze_history=gaze_state.fixation_history,
    )


