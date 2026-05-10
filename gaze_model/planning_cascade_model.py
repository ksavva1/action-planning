"""
Planning Model
    Trials run up to max_timesteps.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json, os


# Parameters
# -------------------------------------------------------------------------

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

# Layers
# -------------------------------------------------------------------------


class GazeController:
    """
    Controls which entity (object or target) the infant fixates each timestep.
    Switch probability increases with dwell time via a hazard function.

    Args:
        p: DevelopmentalParams governing gaze_switch_rate, fixation_duration_mean, and target_bias
    """
    def __init__(self, p):
        self.p = p; self.reset()

    def reset(self):
        """Reset all gaze state to initial values for a new trial."""
        self.cur = "object"; self.dwell = 0; self.switches = 0
        self.obj_fixes = 0; self.tgt_fixes = 0; self.history = []
    
    def step(self, rng):
        """Advance one timestep: possibly switch fixation target.

        Args:
            rng: numpy random Generator for stochastic decisions.
        Returns:
            str: current fixation target, either "object" or "target".
        """
        self.dwell += 1
        # switch probability grows with dwell time
        hazard = 1.0 - np.exp(-self.dwell / self.p.fixation_duration_mean)

        if rng.random() < self.p.gaze_switch_rate * hazard:
            self.cur = "target" if self.cur == "object" else (
                "object" if rng.random() < (1 - self.p.target_bias) else "target")
            self.dwell = 0; self.switches += 1
        
        self.history.append(self.cur)
        
        if self.cur == "object": 
            self.obj_fixes += 1
        
        else: 
            self.tgt_fixes += 1

        return self.cur

class PerceptualLayer:
    """Samples noisy features from whichever entity is currently fixated.

    11-dim feature vector:
      [0:5]  object features  (x, y, angle, width, height)
      [5:8]  target features  (goal_x, goal_y, goal_angle)
      [8:11] relational feats (dx, dy, d_angle between object and target)

    Only the fixated entity's slots are populated.
    Relational features require both entities fixated within the last 3 timesteps.

    Args:
        p: DevelopmentalParams with sampling_rate, perceptual_noise, and acuity values.
    """

    def __init__(self, p):
        self.p = p
        self.ac_obj = np.array([p.location_acuity]*2 + [p.orientation_acuity]*3)
        self.ac_tgt = np.array([p.location_acuity]*2 + [p.orientation_acuity])
        self.ac_rel = np.array([p.relation_acuity]*3)

    def sample(self, obj, tgt, gaze, both_recent, rng):
        """
        Sample perceptual features given current gaze.

        Args:
            obj: np.ndarray of shape (5,) - object state [x, y, angle, width, height]
            tgt: np.ndarray of shape (3,) - target state [goal_x, goal_y, goal_angle]
            gaze: str - "object" or "target", current fixation
            both_recent: bool - whether both entities were fixated within last 3 timesteps
            rng: numpy random gen

        Returns:
            tuple of (percept, mask):
                percept: np.ndarray of shape (11,) - sampled feature values (noisy)
                mask: np.ndarray of shape (11,) - 1 where feature was sampled, 0 otherwise
        """
        perc = np.zeros(11); mask = np.zeros(11)
        sr, ns = self.p.sampling_rate, self.p.perceptual_noise

        if gaze == "object":
            s = rng.random(5) < sr
            perc[:5] = (obj[:5] + rng.normal(0, ns, 5)*(1-self.ac_obj)) * s
            mask[:5] = s.astype(float)

        else:
            s = rng.random(3) < sr
            perc[5:8] = (tgt[:3] + rng.normal(0, ns, 3)*(1-self.ac_tgt)) * s
            mask[5:8] = s.astype(float)

        if both_recent and rng.random() < self.p.simultaneous_rate:
            rel = np.array([tgt[0]-obj[0], tgt[1]-obj[1], tgt[2]-obj[2]])
            s = rng.random(3) < sr
            perc[8:11] = (rel + rng.normal(0, ns, 3)*(1-self.ac_rel)) * s
            mask[8:11] = s.astype(float)

        return perc, mask

class WMLayer:
    """
    Working memory with differential decay for fixated vs non-fixated entities.

    Maintains a buffer of 11 feature values with associated trace strengths.
    A soft capacity limit weakens the least-active traces when too many are strong.

    Args:
        p: DevelopmentalParams with wm_capacity, wm_decay, and wm_unfixated_decay.
    """

    def __init__(self, p):
        self.p = p; self.buf = np.zeros(11); self.str = np.zeros(11)

    def reset(self):
        """Clear all memory traces for a new trial."""
        self.buf[:] = 0; self.str[:] = 0

    def update(self, perc, mask, gaze):
        """Integrate a new percept into mem, applying decay and capacity limits.

        Args:
            perc: np.ndarray of shape (11,) - sampled feature values from PerceptualLayer
            mask: np.ndarray of shape (11,) - which features were sampled (1 or 0)
            gaze: str - "object" or "target", determines which traces decay faster

        Returns:
            np.ndarray of shape (11,) - current WM contents (values * strengths)
        """
        decay = np.full(11, self.p.wm_decay)
        
        if gaze == "object": 
            decay[5:8] = self.p.wm_unfixated_decay
        else: 
            decay[:5] = self.p.wm_unfixated_decay

        decay[8:11] = self.p.wm_unfixated_decay
        self.str *= (1 - decay)

        for i in range(11):
            if mask[i] > .5:
                if self.str[i] > .1:
                    self.buf[i] = .4*self.buf[i] + .6*perc[i]
                else:
                    self.buf[i] = perc[i]
                self.str[i] = min(1.0, self.str[i] + .5)

        if np.sum(self.str > .1) > self.p.wm_capacity:
            th = np.sort(self.str)[::-1][self.p.wm_capacity]
            self.str[self.str < th] *= .3

        return self.buf * self.str

class AffordanceLayer:
    """
    Maps 11-dim WM state to 4 affordances [reach, grasp, rotate, translate] via a structured weight matrix.

    Args:
        p: DevelopmentalParams with affordance_coupling and affordance_noise
        rng: numpy random gen for weight initialisation noise
    """

    def __init__(self, p, rng):
        self.p = p
        W = np.zeros((11, 4))
        W[0,:] = [.6,.1,0,.5]; W[1,:] = [.6,.1,0,.5]
        W[2,:] = [0,.2,.7,0]; W[3,:] = [.1,.6,.3,.1]; W[4,:] = [.1,.6,.3,.1]
        W[5,:] = [.3,0,0,.4]; W[6,:] = [.3,0,0,.4]; W[7,:] = [0,.1,.5,0]
        W[8,:] = [.2,0,.1,.8]; W[9,:] = [.2,0,.1,.8]; W[10,:] = [0,.1,.9,0]
        self.W = np.clip(W * p.affordance_coupling + rng.normal(0,.03,W.shape), 0, 1)
        self.b = np.array([.15,.1,.05,.2])

    def estimate(self, wm, rng):
        """
        Compute affordance activations from working memory state.

        Args:
            wm: np.ndarray of shape (11,) - current WM contents
            rng: numpy random gen for activation noise

        Returns:
            np.ndarray of shape (4,) - affordance activations in [0, 1] for [reach, grasp, rotate, translate].
        """
        r = np.dot(wm, self.W) + self.b*.1
        return 1/(1+np.exp(-5*(r + rng.normal(0, self.p.affordance_noise, 4) - .3)))

class MotorLayer:
    """
    Converts affordance activations to motor commands.

    Args:
        p: DevelopmentalParams with planning_horizon and motor_noise
        rng: numpy random gen for weight initialisation noise
    """
    def __init__(self, p, rng):
        self.p = p
        self.W = np.array([[.9,0,0,0],[0,.9,0,0],[0,0,.9,0],[.5,.5,0,0]]) + rng.normal(0,.02,(4,4))
    
    def plan(self, aff, goal, cur, rng):
        """
        Generate a motor command from affordances and goal error.

        Args:
            aff: np.ndarray of shape (4,) - affordance activations
            goal: np.ndarray of shape (4,) - goal motor state [x, y, angle, grip]
            cur: np.ndarray of shape (4,) - current motor state [x, y, angle, grip]
            rng: numpy random gen for execution noise

        Returns:
            np.ndarray of shape (4,) - motor command [dx, dy, d_angle, d_grip],
                clipped to [-1, 1], with deceleration applied near the goal
        """
        raw = np.dot(aff, self.W)
        ge = goal[:4] - cur[:4]
        pw = min(1.0, self.p.planning_horizon / 6.0)
        cmd = (1-pw)*raw + pw*ge*.3 + rng.normal(0, self.p.motor_noise, 4)
        
        # Decelerate near goal, scale by distance
        dist = np.sqrt(np.sum(ge[:3]**2))
        brake = min(1.0, dist / 0.5)  # Slow down within 0.5 of goal
        
        return np.clip(cmd * brake, -1, 1)

class HabitLayer:
    """
    Translate-first habitual bias that competes with goal-directed control.
    Two phases: translate (dx, dy only) then rotate (d_angle only)

    Args:
        p: DevelopmentalParams with habit_strength and goal_directed_strength
    """
        
    def __init__(self, p):
        self.p = p; self.reset()

    def reset(self):
        """Reset step counter for a new trial."""
        self.step_n = 0

    def blend(self, goal_cmd):
        """
        Blend habitual and goal-directed motor commands.

        Args:
            goal_cmd: np.ndarray of shape (4,) - goal-directed motor command

        Returns:
            np.ndarray of shape (4,) - weighted blend of habitual default and
                goal-directed command, with habitual weight fading over time
        """
        self.step_n += 1
        phase_len = int(4 + 12 * self.p.habit_strength)
        hab = np.array([.7,.7,0,.2]) if self.step_n < phase_len else np.array([0,0,.8,.5])
        
        # Habitual influence decays over time
        fade = max(0.0, 1.0 - self.step_n / (phase_len * 3))
        h_eff = self.p.habit_strength * fade
        g_eff = self.p.goal_directed_strength + self.p.habit_strength * (1 - fade)

        return (h_eff/(h_eff+g_eff))*hab + (g_eff/(h_eff+g_eff))*goal_cmd

class CorrectionLayer:
    """
    Online error correction during movement, subject to processing delay.

    Compares current state to goal and generates corrective motor adjustments.
    Decelerates correction near the goal to prevent overshoot.

    Args:
        p: DevelopmentalParams with correction_rate, correction_delay, and motor_noise.
    """
    def __init__(self, p):
        self.p = p; self.hist = []

    def reset(self):
        """Clear error history for a new trial."""
        self.hist = []

    def correct(self, cur, goal, t, rng):
        """Compute a corrective motor adjustment based on delayed error.

        Args:
            cur: np.ndarray of shape (4,) — current motor state [x, y, angle, grip]
            goal: np.ndarray of shape (4,) — goal motor state
            t: int — current timestep (correction inactive before correction_delay)
            rng: numpy random gen for correction noise

        Returns:
            np.ndarray of shape (4,) — corrective motor adjustment, scaled down
                near the goal to prevent overshoot. Zero if t < correction_delay.
        """
        err = goal[:4] - cur[:4]; self.hist.append(err.copy())
        
        if t < self.p.correction_delay: 
            return np.zeros(4)
        
        idx = max(0, len(self.hist)-1-self.p.correction_delay)
        corr = self.p.correction_rate * self.hist[idx] + rng.normal(0, self.p.motor_noise*.5, 4)
        dist = np.sqrt(np.sum(err[:3]**2))
        brake = min(1.0, dist / 0.4)

        return corr * brake


# Task
# -------------------------------------------------------------------------

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
    
    # Set success tolerances
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


# Timestep record
# -------------------------------------------------------------------------

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


# Network
# -------------------------------------------------------------------------

class PlanningCascadeNetwork:
    """
    Full perception-action cascade bringing all layers together.

    Args:
        params: DevelopmentalParams defining this network's cognitive profile
        seed: int random seed for reproducibility
    """
    def __init__(self, params: DevelopmentalParams, seed=42):
        self.p = params
        self.rng = np.random.default_rng(seed)
        self.gaze = GazeController(params)
        self.perc = PerceptualLayer(params)
        self.wm = WMLayer(params)
        self.aff = AffordanceLayer(params, self.rng)
        self.mot = MotorLayer(params, self.rng)
        self.hab = HabitLayer(params)
        self.corr = CorrectionLayer(params)

    def run_trial(self, task: TaskConfig, trial_id=0) -> TrialResult:
        """
        Simulate one trial: the infant tries to fit the object into the target slot.

        The trial loop runs the full cascade each timestep:
        gaze -> perceive -> WM update -> check initiation -> affordances ->
        motor plan -> habitual blend -> correction -> execute -> check success.

        Args:
            task: TaskConfig defining start/goal poses and tolerances
            trial_id: int index for this trial (used in output only)

        Returns:
            TrialResult containing success flag, trajectory, timing metrics,
                gaze statistics, and efficiency.
        """
        self.gaze.reset(); self.wm.reset(); self.hab.reset(); self.corr.reset()

        ox, oy, oa = task.start_x, task.start_y, task.start_angle
        tgt = np.array([task.goal_x, task.goal_y, task.goal_angle])
        goal_m = np.array([task.goal_x, task.goal_y, task.goal_angle, .5])
        traj = []; started = False
        mv_on = rot_on = tx_on = task.max_timesteps
        last_obj_t = last_tgt_t = -10

        for t in range(task.max_timesteps):
            obj_s = np.array([ox, oy, oa, task.obj_width, task.obj_height])
            
            # Error computed before movement
            pe = np.sqrt((ox - task.goal_x)**2 + (oy - task.goal_y)**2)
            ae = abs(oa - task.goal_angle)
            
            # 1. Gaze
            gz = self.gaze.step(self.rng)
            if gz == "object": 
                last_obj_t = t
            else: 
                last_tgt_t = t
            
            both = (t - last_obj_t <= 3 and t - last_tgt_t <= 3)

            # 2. Percieve
            perc, mask = self.perc.sample(obj_s, tgt, gz, both, self.rng)

            # 3. Working memory
            wm = self.wm.update(perc, mask, gz)
            s = self.wm.str.copy()
            oi, ti, ri = float(np.mean(s[:5])), float(np.mean(s[5:8])), float(np.mean(s[8:11]))

            # 4. Initiation check
            if not started and np.mean(s) >= self.p.initiation_threshold:
                started = True; mv_on = t

            # 5. Affordances
            aff = self.aff.estimate(wm, self.rng)

            # 6-8. Motor plan, habitual blend, and correction
            cur_m = np.array([ox, oy, oa, .3])
            gcmd = self.mot.plan(aff, goal_m, cur_m, self.rng)
            bl = self.hab.blend(gcmd)
            cr = self.corr.correct(cur_m, goal_m, t, self.rng)

            # 9. Combine with near-goal deceleration
            if started:
                raw = bl + cr
                # Direct error vector to goal
                dx = task.goal_x - ox
                dy = task.goal_y - oy
                da = task.goal_angle - oa
                dist = np.sqrt(dx**2 + dy**2 + da**2)

                # When close, override with pure proportional control toward goal
                if dist < 0.6:
                    # Blend toward direct error-correction as we get closer
                    closeness = 1.0 - min(dist / 0.6, 1.0)  # 0 far, 1 close
                    direct = np.array([dx, dy, da, 0]) * 2.0  # proportional gain
                    raw = raw * (1 - closeness) + direct * closeness

                # Scale command by distance to prevent overshoot
                speed = min(1.0, dist * 2.5)
                final = np.clip(raw * speed, -1, 1)
            else:
                final = np.zeros(4)

            # 10. Execute
            step = 0.10
            ox += final[0]*step; oy += final[1]*step; oa += final[2]*step

            # Track
            if started:
                if abs(final[2]) > .1 and rot_on == task.max_timesteps: rot_on = t
                if (abs(final[0]) > .1 or abs(final[1]) > .1) and tx_on == task.max_timesteps: tx_on = t

            traj.append(TimestepRecord(
                t=t, obj_x=ox, obj_y=oy, obj_angle=oa,
                gaze_target=gz, movement_started=started,
                obj_info=oi, tgt_info=ti, rel_info=ri,
                pos_error=pe, angle_error=ae,
                gaze_switches=self.gaze.switches,
            ))

            # 11. Check success
            if pe < task.position_tolerance and ae < task.angle_tolerance:
                break

        # Calculate trial-level metrics
        fpe = np.sqrt((ox-task.goal_x)**2 + (oy-task.goal_y)**2)
        fae = abs(oa - task.goal_angle)
        succ = fpe < task.position_tolerance and fae < task.angle_tolerance
        opt = np.sqrt((task.goal_x-task.start_x)**2+(task.goal_y-task.start_y)**2+(task.goal_angle-task.start_angle)**2)
        act = sum(np.sqrt((traj[i].obj_x-traj[i-1].obj_x)**2+(traj[i].obj_y-traj[i-1].obj_y)**2+(traj[i].obj_angle-traj[i-1].obj_angle)**2) for i in range(1,len(traj)))
        eff = min(opt / max(act, .01), 1.0)
        tot = len(self.gaze.history)

        return TrialResult(
            params_name=self.p.name, task_name=task.name, trial_id=trial_id,
            success=succ, timesteps_used=len(traj),
            final_pos_error=fpe, final_angle_error=fae,
            trajectory=traj, movement_onset=mv_on,
            rotation_onset=rot_on, translation_onset=tx_on,
            efficiency=eff, total_gaze_switches=self.gaze.switches,
            object_fixation_pct=self.gaze.obj_fixes/max(tot,1),
            target_fixation_pct=self.gaze.tgt_fixes/max(tot,1),
            gaze_history=self.gaze.history,
        )


# Batch
# -------------------------------------------------------------------------

def run_simulation(stages=None, tasks=None, n_trials=20, seed=42):
    """Run all stage x task x trial combinations.

    Args:
        stages: list of stage name strings (default: all in DEVELOPMENTAL_STAGES)
        tasks: list of task name strings (default: all in TASKS)
        n_trials: int number of trials per (stage, task) pair
        seed: int base random seed; each trial uses seed + trial_index

    Returns:
        dict with keys "summary", "trajectories", "developmental_params",
            "stages", "tasks" — ready for JSON serialisation.
    """
    if stages is None: 
        stages = list(DEVELOPMENTAL_STAGES.keys())
    if tasks is None: 
        tasks = list(TASKS.keys())
    results = []

    for sn in stages:
        p = DEVELOPMENTAL_STAGES[sn]

        for tn in tasks:
            task = TASKS[tn]

            for i in range(n_trials):
                net = PlanningCascadeNetwork(p, seed=seed+i)
                results.append(net.run_trial(task, trial_id=i))

    return compile_results(results)

def compile_results(results):
    """
    Aggregate trial-level results into summary stats and best-trial trajectories.

    Args:
        results: list of TrialResult objects from multiple trials

    Returns:
        dict with keys:
            "summary": dict of {stage_task: aggregated stats}
            "trajectories": dict of {stage_task: best trial's timestep data}
            "developmental_params": dict of {stage: param values}
            "stages": list of stage names
            "tasks": list of task names
    """
    summary, trajectories = {}, {}
    groups = {}

    for r in results:
        groups.setdefault((r.params_name, r.task_name), []).append(r)

    for (stage, task), trials in groups.items():
        k = f"{stage}_{task}"; n = len(trials)

        summary[k] = {
            "stage": stage, "task": task, "n_trials": n,
            "success_rate": sum(t.success for t in trials)/n,
            "mean_timesteps": float(np.mean([t.timesteps_used for t in trials])),
            "mean_pos_error": float(np.mean([t.final_pos_error for t in trials])),
            "mean_angle_error": float(np.mean([t.final_angle_error for t in trials])),
            "mean_efficiency": float(np.mean([t.efficiency for t in trials])),
            "mean_movement_onset": float(np.mean([t.movement_onset for t in trials])),
            "mean_gaze_switches": float(np.mean([t.total_gaze_switches for t in trials])),
            "mean_object_fixation_pct": float(np.mean([t.object_fixation_pct for t in trials])),
            "mean_target_fixation_pct": float(np.mean([t.target_fixation_pct for t in trials])),
            "translate_before_rotate_rate": sum(1 for t in trials if t.translation_onset < t.rotation_onset)/n,
        }

        best = min(trials, key=lambda t: t.final_pos_error + t.final_angle_error)
        trajectories[k] = {
            "steps": [{"t":s.t,"x":round(s.obj_x,4),"y":round(s.obj_y,4),"a":round(s.obj_angle,4),
                        "gz":s.gaze_target,"mv":s.movement_started,
                        "pe":round(s.pos_error,4),"ae":round(s.angle_error,4)}
                       for s in best.trajectory],
            "success": best.success, "gaze_history": best.gaze_history,
        }

    dev_params = {}

    for name, p in DEVELOPMENTAL_STAGES.items():
        dev_params[name] = {k: getattr(p, k) for k in [
            "gaze_switch_rate","fixation_duration_mean","target_bias","simultaneous_rate",
            "sampling_rate","perceptual_noise","location_acuity","orientation_acuity","relation_acuity",
            "wm_capacity","wm_decay","wm_unfixated_decay","affordance_coupling",
            "planning_horizon","habit_strength","goal_directed_strength",
            "correction_rate","initiation_threshold"]}
        
    return {"summary":summary,"trajectories":trajectories,
            "developmental_params":dev_params,
            "stages":list(DEVELOPMENTAL_STAGES.keys()),"tasks":list(TASKS.keys())}

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy bool_, integer, floating, and ndarray types."""
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

if __name__ == "__main__":
    print("Running simulation...")

    res = run_simulation(n_trials=25)
    d = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(d, "simulation_results.json")

    with open(p, "w") as f: 
        json.dump(res, f, indent=2, cls=NumpyEncoder)

    print(f"Done → {p}")
    hdr = f"{'Stg':<4} {'Task':<22} {'Succ':>5} {'Steps':>6} {'PosE':>6} {'AngE':>6} {'Sw':>4} {'Eff':>5}"
    print(f"\n{hdr}\n{'-'*60}")
    
    for k,v in sorted(res["summary"].items()):
        print(f"{v['stage']:<4} {v['task']:<22} {v['success_rate']:>5.2f} "
              f"{v['mean_timesteps']:>6.1f} {v['mean_pos_error']:>6.3f} "
              f"{v['mean_angle_error']:>6.3f} {v['mean_gaze_switches']:>4.0f} "
              f"{v['mean_efficiency']:>5.3f}")