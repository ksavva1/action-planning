"""
Planning Model — layer classes and network.

    Trials run up to max_timesteps.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_config import (
    DevelopmentalParams, DEVELOPMENTAL_STAGES,
    TaskConfig, TASKS,
    TimestepRecord, TrialResult,
)

# Re-exported so existing callers (animate_model, hyperparam_sweep) keep working.
__all__ = [
    "DevelopmentalParams", "DEVELOPMENTAL_STAGES",
    "TaskConfig", "TASKS",
    "TimestepRecord", "TrialResult",
    "GazeController", "PerceptualLayer", "WMLayer",
    "AffordanceLayer", "MotorLayer", "HabitLayer", "CorrectionLayer",
    "PlanningCascadeNetwork",
]


# Layers
# -------------------------------------------------------------------------

class GazeController:
    """
    Controls which entity (object or target) the infant fixates each timestep.
    Switch probability increases with dwell time via a hazard function.

    Args:
        p: governs gaze_switch_rate, fixation_duration_mean, and target_bias
    """
    def __init__(self, p: DevelopmentalParams) -> None:
        self.p = p
        self.reset()

    def reset(self) -> None:
        """Reset all gaze state to initial values for a new trial."""
        self.cur = "object"
        self.dwell = 0
        self.switches = 0
        self.obj_fixes = 0
        self.tgt_fixes = 0
        self.history = []

    def step(self, rng: np.random.Generator) -> str:
        """Advance one timestep: possibly switch fixation target.

        Args:
            rng: for stochastic decisions.

        Returns:
            current fixation target, either "object" or "target".
        """
        self.dwell += 1
        hazard = 1.0 - np.exp(-self.dwell / self.p.fixation_duration_mean)

        if rng.random() < self.p.gaze_switch_rate * hazard:
            if self.cur == "object":
                self.cur = "target"
            else:
                # When leaving target, bias controls where gaze lands next
                if rng.random() < (1 - self.p.target_bias):
                    self.cur = "object"
                else:
                    self.cur = "target"
            self.dwell = 0
            self.switches += 1

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
        p: governs sampling_rate, perceptual_noise, and acuity values.
    """

    def __init__(self, p: DevelopmentalParams) -> None:
        self.p = p
        self.ac_obj = np.array([p.location_acuity] * 2 + [p.orientation_acuity] * 3)
        self.ac_tgt = np.array([p.location_acuity] * 2 + [p.orientation_acuity])
        self.ac_rel = np.array([p.relation_acuity] * 3)

    def sample(
        self,
        obj: np.ndarray,
        tgt: np.ndarray,
        gaze: str,
        both_recent: bool,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample perceptual features given current gaze.

        Args:
            obj: object state [x, y, angle, width, height], shape (5,)
            tgt: target state [goal_x, goal_y, goal_angle], shape (3,)
            gaze: current fixation, "object" or "target"
            both_recent: whether both entities were fixated within last 3 timesteps
            rng: for sampling noise

        Returns:
            percept: sampled feature values (noisy), shape (11,)
            mask: 1 where feature was sampled, 0 otherwise, shape (11,)
        """
        perc = np.zeros(11)
        mask = np.zeros(11)
        sr = self.p.sampling_rate
        ns = self.p.perceptual_noise

        if gaze == "object":
            sampled = rng.random(5) < sr
            noise = rng.normal(0, ns, 5) * (1 - self.ac_obj)
            perc[:5] = (obj[:5] + noise) * sampled
            mask[:5] = sampled.astype(float)
        else:
            sampled = rng.random(3) < sr
            noise = rng.normal(0, ns, 3) * (1 - self.ac_tgt)
            perc[5:8] = (tgt[:3] + noise) * sampled
            mask[5:8] = sampled.astype(float)

        if both_recent and rng.random() < self.p.simultaneous_rate:
            rel = np.array([tgt[0] - obj[0], tgt[1] - obj[1], tgt[2] - obj[2]])
            sampled = rng.random(3) < sr
            noise = rng.normal(0, ns, 3) * (1 - self.ac_rel)
            perc[8:11] = (rel + noise) * sampled
            mask[8:11] = sampled.astype(float)

        return perc, mask


class WMLayer:
    """
    Working memory with differential decay for fixated vs non-fixated entities.

    Maintains a buffer of 11 feature values with associated trace strengths.
    A soft capacity limit weakens the least-active traces when too many are strong.

    Args:
        p: governs wm_capacity, wm_decay, and wm_unfixated_decay.
    """

    def __init__(self, p: DevelopmentalParams) -> None:
        self.p = p
        self.buf = np.zeros(11)
        self.strength = np.zeros(11)

    def reset(self) -> None:
        """Clear all memory traces for a new trial."""
        self.buf[:] = 0
        self.strength[:] = 0

    def update(self, perc: np.ndarray, mask: np.ndarray, gaze: str) -> np.ndarray:
        """Integrate a new percept into memory, applying decay and capacity limits.

        Args:
            perc: sampled feature values from PerceptualLayer, shape (11,)
            mask: which features were sampled (1 or 0), shape (11,)
            gaze: "object" or "target", determines which traces decay faster

        Returns:
            current WM contents (values * strengths), shape (11,)
        """
        decay = np.full(11, self.p.wm_decay)

        if gaze == "object":
            decay[5:8] = self.p.wm_unfixated_decay
        else:
            decay[:5] = self.p.wm_unfixated_decay

        decay[8:11] = self.p.wm_unfixated_decay
        self.strength *= (1 - decay)

        for i in range(11):
            if mask[i] > 0.5:
                if self.strength[i] > 0.1:
                    # Blend new percept with existing memory
                    self.buf[i] = 0.4 * self.buf[i] + 0.6 * perc[i]
                else:
                    self.buf[i] = perc[i]
                self.strength[i] = min(1.0, self.strength[i] + 0.5)

        if np.sum(self.strength > 0.1) > self.p.wm_capacity:
            threshold = np.sort(self.strength)[::-1][self.p.wm_capacity]
            self.strength[self.strength < threshold] *= 0.3

        return self.buf * self.strength


class AffordanceLayer:
    """
    Maps 11-dim WM state to 4 affordances [reach, grasp, rotate, translate]
    via a structured weight matrix.

    Args:
        p: governs affordance_coupling and affordance_noise
        rng: for weight initialisation noise
    """

    def __init__(self, p: DevelopmentalParams, rng: np.random.Generator) -> None:
        self.p = p
        W = np.zeros((11, 4))
        W[0, :] = [.6, .1, 0, .5]
        W[1, :] = [.6, .1, 0, .5]
        W[2, :] = [0, .2, .7, 0]
        W[3, :] = [.1, .6, .3, .1]
        W[4, :] = [.1, .6, .3, .1]
        W[5, :] = [.3, 0, 0, .4]
        W[6, :] = [.3, 0, 0, .4]
        W[7, :] = [0, .1, .5, 0]
        W[8, :] = [.2, 0, .1, .8]
        W[9, :] = [.2, 0, .1, .8]
        W[10, :] = [0, .1, .9, 0]
        self.W = np.clip(W * p.affordance_coupling + rng.normal(0, .03, W.shape), 0, 1)
        self.b = np.array([.15, .1, .05, .2])

    def estimate(self, wm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Compute affordance activations from working memory state.

        Args:
            wm: current WM contents, shape (11,)
            rng: for activation noise

        Returns:
            affordance activations in [0, 1] for [reach, grasp, rotate, translate], shape (4,)
        """
        raw = np.dot(wm, self.W) + self.b * 0.1
        noisy = raw + rng.normal(0, self.p.affordance_noise, 4)
        return 1 / (1 + np.exp(-5 * (noisy - 0.3)))


class MotorLayer:
    """
    Converts affordance activations to motor commands.

    Args:
        p: governs planning_horizon and motor_noise
        rng: for weight initialisation noise
    """
    def __init__(self, p: DevelopmentalParams, rng: np.random.Generator) -> None:
        self.p = p
        self.W = (
            np.array([[.9, 0, 0, 0], [0, .9, 0, 0], [0, 0, .9, 0], [.5, .5, 0, 0]])
            + rng.normal(0, .02, (4, 4))
        )

    def plan(
        self,
        aff: np.ndarray,
        goal: np.ndarray,
        cur: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate a motor command from affordances and goal error.

        Args:
            aff: affordance activations, shape (4,)
            goal: goal motor state [x, y, angle, grip], shape (4,)
            cur: current motor state [x, y, angle, grip], shape (4,)
            rng: for execution noise

        Returns:
            motor command [dx, dy, d_angle, d_grip] clipped to [-1, 1],
            with deceleration applied near the goal, shape (4,)
        """
        affordance_cmd = np.dot(aff, self.W)
        goal_error = goal[:4] - cur[:4]
        plan_weight = min(1.0, self.p.planning_horizon / 6.0)
        cmd = (
            (1 - plan_weight) * affordance_cmd
            + plan_weight * goal_error * 0.3
            + rng.normal(0, self.p.motor_noise, 4)
        )

        dist = np.sqrt(np.sum(goal_error[:3] ** 2))
        brake = min(1.0, dist / 0.5)

        return np.clip(cmd * brake, -1, 1)


class HabitLayer:
    """
    Translate-first habitual bias that competes with goal-directed control.
    Two phases: translate (dx, dy only) then rotate (d_angle only)

    Args:
        p: governs habit_strength and goal_directed_strength
    """

    def __init__(self, p: DevelopmentalParams) -> None:
        self.p = p
        self.reset()

    def reset(self) -> None:
        """Reset step counter for a new trial."""
        self.step_n = 0

    def blend(self, goal_cmd: np.ndarray) -> np.ndarray:
        """Blend habitual and goal-directed motor commands.

        Args:
            goal_cmd: goal-directed motor command, shape (4,)

        Returns:
            weighted blend of habitual default and goal-directed command,
            with habitual weight fading over time, shape (4,)
        """
        self.step_n += 1
        phase_len = int(4 + 12 * self.p.habit_strength)

        # The habit follows a two-phase sequence that mirrors a common infant motor
        # pattern: first push the object laterally into position (translate-first),
        # then rotate it to fit the slot. phase_len grows with habit_strength, so a
        # more habitual agent spends longer in each phase before switching.
        if self.step_n < phase_len:
            hab = np.array([0.7, 0.7, 0.0, 0.2])   # Phase 1: strong dx/dy, no rotation
        else:
            hab = np.array([0.0, 0.0, 0.8, 0.5])   # Phase 2: strong d_angle, no translation

        # The habitual weight fades to zero linearly over 3× phase_len steps.
        # As it fades, its share is transferred to g_eff rather than simply dropped,
        # so the two weights always sum to 1 and goal-directed control fills the gap
        # left by the retreating habit. A more habitual agent (high habit_strength)
        # starts with a larger h_eff and takes longer to fully relinquish control.
        fade = max(0.0, 1.0 - self.step_n / (phase_len * 3))
        h_eff = self.p.habit_strength * fade
        g_eff = self.p.goal_directed_strength + self.p.habit_strength * (1 - fade)

        total = h_eff + g_eff
        return (h_eff / total) * hab + (g_eff / total) * goal_cmd


class CorrectionLayer:
    """
    Online error correction during movement, subject to processing delay.

    Compares current state to goal and generates corrective motor adjustments.
    Decelerates correction near the goal to prevent overshoot.

    Args:
        p: governs correction_rate, correction_delay, and motor_noise.
    """
    def __init__(self, p: DevelopmentalParams) -> None:
        self.p = p
        self.hist = []

    def reset(self) -> None:
        """Clear error history for a new trial."""
        self.hist = []

    def correct(
        self,
        cur: np.ndarray,
        goal: np.ndarray,
        t: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Compute a corrective motor adjustment based on delayed error.

        Args:
            cur: current motor state [x, y, angle, grip], shape (4,)
            goal: goal motor state, shape (4,)
            t: current timestep (correction inactive before correction_delay)
            rng: for correction noise

        Returns:
            corrective motor adjustment scaled down near the goal, shape (4,).
            Zero if t < correction_delay.
        """
        err = goal[:4] - cur[:4]
        self.hist.append(err.copy())

        if t < self.p.correction_delay:
            return np.zeros(4)

        delayed_idx = max(0, len(self.hist) - 1 - self.p.correction_delay)
        delayed_err = self.hist[delayed_idx]
        corr = self.p.correction_rate * delayed_err + rng.normal(0, self.p.motor_noise * 0.5, 4)
        dist = np.sqrt(np.sum(err[:3] ** 2))
        brake = min(1.0, dist / 0.4)

        return corr * brake


# Network
# -------------------------------------------------------------------------

class PlanningCascadeNetwork:
    """
    Full perception-action cascade bringing all layers together.

    Args:
        params: defines this network's cognitive profile
        seed: random seed for reproducibility
    """
    def __init__(self, params: DevelopmentalParams, seed: int = 42) -> None:
        self.p = params
        self.rng = np.random.default_rng(seed)
        self.gaze = GazeController(params)
        self.perc = PerceptualLayer(params)
        self.wm = WMLayer(params)
        self.aff = AffordanceLayer(params, self.rng)
        self.mot = MotorLayer(params, self.rng)
        self.hab = HabitLayer(params)
        self.corr = CorrectionLayer(params)

    def run_trial(self, task: TaskConfig, trial_id: int = 0) -> TrialResult:
        """
        Simulate one trial: the infant tries to fit the object into the target slot.

        The trial loop runs the full cascade each timestep:
        gaze -> perceive -> WM update -> check initiation -> affordances ->
        motor plan -> habitual blend -> correction -> execute -> check success.

        Args:
            task: defines start/goal poses and tolerances
            trial_id: index for this trial (used in output only)

        Returns:
            contains success flag, trajectory, timing metrics, gaze statistics,
            and efficiency.
        """
        self.gaze.reset()
        self.wm.reset()
        self.hab.reset()
        self.corr.reset()

        ox, oy, oa = task.start_x, task.start_y, task.start_angle
        tgt = np.array([task.goal_x, task.goal_y, task.goal_angle])
        goal_m = np.array([task.goal_x, task.goal_y, task.goal_angle, .5])
        traj = []
        started = False
        mv_on = rot_on = tx_on = task.max_timesteps
        last_obj_t = last_tgt_t = -10

        for t in range(task.max_timesteps):
            obj_s = np.array([ox, oy, oa, task.obj_width, task.obj_height])
            pe = np.sqrt((ox - task.goal_x) ** 2 + (oy - task.goal_y) ** 2)
            ae = abs(oa - task.goal_angle)

            # 1. Gaze
            gz = self.gaze.step(self.rng)
            if gz == "object":
                last_obj_t = t
            else:
                last_tgt_t = t
            both = (t - last_obj_t <= 3 and t - last_tgt_t <= 3)

            # 2. Perceive
            perc, mask = self.perc.sample(obj_s, tgt, gz, both, self.rng)

            # 3. Working memory
            wm = self.wm.update(perc, mask, gz)
            s = self.wm.strength.copy()
            oi = float(np.mean(s[:5]))
            ti = float(np.mean(s[5:8]))
            ri = float(np.mean(s[8:11]))

            # 4. Initiation check
            if not started and np.mean(s) >= self.p.initiation_threshold:
                started = True
                mv_on = t

            # 5. Affordances
            aff = self.aff.estimate(wm, self.rng)

            # 6–8. Build the motor command from three contributions that together
            #      capture both habitual tendencies and goal-directed feedback:
            #
            #   Motor plan (step 6): converts affordance activations and the current
            #   goal-error vector into a raw [dx, dy, d_angle, d_grip] command.
            #   The planning_horizon parameter controls how heavily the command leans
            #   toward directly tracking goal error versus following learned affordances
            #   — higher horizon is more anticipatory, lower is more reactive.
            #
            #   Habitual blend (step 7): overlays a developmental bias on the motor
            #   plan. Early in each trial a translate-first pattern partially overrides
            #   the goal-directed command; this habitual influence fades over time,
            #   gradually handing full control back to the goal-directed signal.
            #   See HabitLayer.blend() for the weighting mechanics.
            #
            #   Online correction (step 8): adds a delayed corrective signal computed
            #   from the error that existed correction_delay timesteps ago, modelling
            #   the processing lag inherent in sensorimotor feedback loops. A more
            #   mature agent (shorter correction_delay, higher correction_rate) responds
            #   faster and more strongly to mid-movement errors.
            cur_m = np.array([ox, oy, oa, .3])
            gcmd = self.mot.plan(aff, goal_m, cur_m, self.rng)
            bl = self.hab.blend(gcmd)
            cr = self.corr.correct(cur_m, goal_m, t, self.rng)

            # 9. Produce the final command and apply near-goal deceleration.
            #    When the object is far from the slot, the blended + corrected command
            #    drives movement freely. As it approaches (dist < 0.6 units), a direct
            #    proportional feedback signal — an error vector pointing straight at the
            #    goal, scaled by "closeness" (0 at d = 0.6, 1 at d = 0) — is blended
            #    in, providing the fine positional control needed for accurate insertion.
            #    Overall speed is additionally scaled by distance to prevent overshoot.
            if started:
                raw = bl + cr
                dx = task.goal_x - ox
                dy = task.goal_y - oy
                da = task.goal_angle - oa
                dist = np.sqrt(dx ** 2 + dy ** 2 + da ** 2)

                if dist < 0.6:
                    closeness = 1.0 - min(dist / 0.6, 1.0)
                    direct = np.array([dx, dy, da, 0]) * 2.0
                    raw = raw * (1 - closeness) + direct * closeness

                speed = min(1.0, dist * 2.5)
                final = np.clip(raw * speed, -1, 1)
            else:
                final = np.zeros(4)

            # 10. Execute
            step = 0.10
            ox += final[0] * step
            oy += final[1] * step
            oa += final[2] * step

            # Track movement onsets
            if started:
                if abs(final[2]) > .1 and rot_on == task.max_timesteps:
                    rot_on = t
                if (abs(final[0]) > .1 or abs(final[1]) > .1) and tx_on == task.max_timesteps:
                    tx_on = t

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

        # Final metrics
        fpe = np.sqrt((ox - task.goal_x) ** 2 + (oy - task.goal_y) ** 2)
        fae = abs(oa - task.goal_angle)
        succ = fpe < task.position_tolerance and fae < task.angle_tolerance

        opt = np.sqrt(
            (task.goal_x - task.start_x) ** 2
            + (task.goal_y - task.start_y) ** 2
            + (task.goal_angle - task.start_angle) ** 2
        )
        act = sum(
            np.sqrt(
                (traj[i].obj_x - traj[i - 1].obj_x) ** 2
                + (traj[i].obj_y - traj[i - 1].obj_y) ** 2
                + (traj[i].obj_angle - traj[i - 1].obj_angle) ** 2
            )
            for i in range(1, len(traj))
        )
        eff = min(opt / max(act, 0.01), 1.0)
        tot = len(self.gaze.history)

        return TrialResult(
            params_name=self.p.name, task_name=task.name, trial_id=trial_id,
            success=succ, timesteps_used=len(traj),
            final_pos_error=fpe, final_angle_error=fae,
            trajectory=traj, movement_onset=mv_on,
            rotation_onset=rot_on, translation_onset=tx_on,
            efficiency=eff, total_gaze_switches=self.gaze.switches,
            object_fixation_pct=self.gaze.obj_fixes / max(tot, 1),
            target_fixation_pct=self.gaze.tgt_fixes / max(tot, 1),
            gaze_history=self.gaze.history,
        )


if __name__ == "__main__":
    from model_utils import run_simulation, compile_results, NumpyEncoder

    print("Running simulation...")

    trials = run_simulation(n_trials=25)
    res = compile_results(trials)
    d = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(d, "simulation_results.json")

    with open(p, "w") as f:
        json.dump(res, f, indent=2, cls=NumpyEncoder)

    print(f"Done → {p}")
    hdr = f"{'Stg':<4} {'Task':<22} {'Succ':>5} {'Steps':>6} {'PosE':>6} {'AngE':>6} {'Sw':>4} {'Eff':>5}"
    print(f"\n{hdr}\n{'-' * 60}")
    for k, v in sorted(res["summary"].items()):
        print(f"{v['stage']:<4} {v['task']:<22} {v['success_rate']:>5.2f} "
              f"{v['mean_timesteps']:>6.1f} {v['mean_pos_error']:>6.3f} "
              f"{v['mean_angle_error']:>6.3f} {v['mean_gaze_switches']:>4.0f} "
              f"{v['mean_efficiency']:>5.3f}")
