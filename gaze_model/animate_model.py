"""
Animated Visualisation

Displays 4 parameter configurations side-by-side, each showing:
  - The object (coloured rectangle) moving and rotating
  - The target slot (dashed gold outline) the object must fit into
  - A gaze line from an eye icon to whichever entity the infant is fixating
  - The movement trace left behind by the object

"""

import numpy as np, matplotlib, matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib import transforms, patheffects
import sys, os, argparse

# Import model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from planning_cascade_model import (
    DEVELOPMENTAL_STAGES, TASKS, PlanningCascadeNetwork, DevelopmentalParams, TaskConfig)

# Visual constants
STAGES = ["A", "B", "C", "D"]
SC = {"A": "#fb7185", "B": "#fbbf24", "C": "#6ee7b7", "D": "#60a5fa"}  # stage colours
BG = "#0c0e12"; SURF = "#1a1e28"; BORDER = "#2a3148"
TEXT = "#e4e8f1"; DIM = "#636b83"; ACCENT = "#6ee7b7"
GAZE_OBJ = "#fbbf24"    # yellow looking at object
GAZE_TGT = "#a78bfa"    # purple looking at target
GOAL = "#fbbf24"


def plabel(p):
    """Parameter summary label for panel title."""
    h = int(100 * p.habit_strength / (p.habit_strength + p.goal_directed_strength))

    return (f"samp={p.sampling_rate:.2f}  noise={p.perceptual_noise:.2f}  "
            f"gaze_sw={p.gaze_switch_rate:.2f}\n"
            f"WM={p.wm_capacity}  horizon={p.planning_horizon}  "
            f"habit={h}%  corr={p.correction_rate:.2f}")


def sim(task_name, seed=42):
    """
    Run one trial per config and collect results.
    
    Args:
        task_name: str key into TASKS dict.
        seed: int random seed for reproducibility.

    Returns:
        tuple of (results, task):
            results: dict mapping stage name ("A"–"D") to TrialResult.
            task: TaskConfig used for all trials.
    """
    task = TASKS[task_name]
    r = {}

    for s in STAGES:
        r[s] = PlanningCascadeNetwork(DEVELOPMENTAL_STAGES[s], seed=seed).run_trial(task)

    return r, task


def rrect(ax, cx, cy, w, h, a, col, **kw):
    """
    Draw a rotated rounded rectangle centred at (cx, cy).
    
    Args:
        ax: matplotlib axes to draw on
        cx, cy: float centre coordinates of the rectangle
        w, h: float width and height
        a: float rotation angle in radians
        col: str colour for edge
        **kw: optional overrides - fc (facecolor), lw (linewidth),
              ls (linestyle), alpha, z (zorder).

    Returns:
        matplotlib.patches.FancyBboxPatch: the drawn patch.    
    
    """
    r = patches.FancyBboxPatch(
        (-w/2, -h/2), w, h, boxstyle="round,pad=0.012",
        facecolor=kw.get("fc", col), edgecolor=col,
        linewidth=kw.get("lw", 1.5), linestyle=kw.get("ls", "-"),
        alpha=kw.get("alpha", 1), zorder=kw.get("z", 3))

    r.set_transform(transforms.Affine2D().rotate(a).translate(cx, cy) + ax.transData)
    ax.add_patch(r)

    return r


def animate(task_name="rotate_insert", save=False, seed=42):
    """Build and run the matplotlib animation.

    Args:
        task_name: str key into TASKS dict.
        save: bool — if True, save as GIF; if False, show interactive window.
        seed: int random seed.

    Returns:
        matplotlib.animation.FuncAnimation: animation object
    """
    results, task = sim(task_name, seed)
    maxT = max(len(r.trajectory) for r in results.values())
    nf = maxT + 6  # frames
    ow, oh = task.obj_width, task.obj_height

    # ── Set up figure: 1 row of 4 workspace panels ──
    fig, axes = plt.subplots(1, 4, figsize=(20, 6.5), facecolor=BG)
    fig.subplots_adjust(left=.03, right=.98, top=.82, bottom=.04, wspace=.15)
    fig.text(.5, .97, "Planning Model Simulation",
             ha="center", va="top", fontsize=16, fontweight="bold",
             color=ACCENT, fontfamily="serif",
             path_effects=[patheffects.withStroke(linewidth=2, foreground=BG)])
    fig.text(.5, .935, f"Task: {task.name.replace('_',' ').title()}  |  "
             f"Fit into slot at ({task.goal_x},{task.goal_y}) "
             f"rotated {np.degrees(task.goal_angle):.0f}\u00b0",
             ha="center", va="top", fontsize=9, color=DIM)
    ttext = fig.text(.5, .905, "", ha="center", fontsize=9, color=DIM)

    # ── Per-panel persistent elements ──
    trails = {}   # movement trace lines
    status = {}   # status text (error or FITTED/MISSED)
    EX, EY = -.55, 1.05  # eye icon position (top-left of each panel)

    for c, s in enumerate(STAGES):
        ax = axes[c]
        ax.set_facecolor(SURF)
        ax.set_xlim(-.7, 1.1); ax.set_ylim(-.6, 1.25); ax.set_aspect("equal")
        ax.tick_params(colors=DIM, labelsize=5)
        for sp in ax.spines.values(): sp.set_color(BORDER)
        # Title: parameter summary for this config
        ax.set_title(plabel(DEVELOPMENTAL_STAGES[s]), fontsize=6.2,
                     fontfamily="monospace", color=SC[s], pad=5, linespacing=1.3)
        # Static goal slot (dashed outline)
        rrect(ax, task.goal_x, task.goal_y, ow+.05, oh+.05, task.goal_angle,
              GOAL, fc="none", lw=1.8, ls="--", alpha=.5, z=1)
        ax.text(task.goal_x, task.goal_y+oh/2+.08, "target",
                fontsize=6, ha="center", color=f"{GOAL}88", zorder=1)
        # Trail line (updated each frame)
        ln, = ax.plot([], [], color=SC[s], alpha=.4, lw=1.2, zorder=2)
        trails[s] = ln
        # Status text (bottom-right corner)
        st = ax.text(.97, .03, "", transform=ax.transAxes, fontsize=7.5,
                     ha="right", va="bottom", fontfamily="monospace", color=TEXT, zorder=15)
        status[s] = st

    def cleanup(ax):
        """
        Remove all dynamic artists tagged with _d=True from previous frame.
        Args:
            ax: matplotlib Axes to clean.
        """
        for a in list(ax.patches) + list(ax.texts) + list(ax.lines):
            if getattr(a, "_d", False):
                a.remove()

    def update(frame):
        """Draw one frame of the animation."""
        ttext.set_text(f"t = {min(frame, maxT-1)}")

        for c, s in enumerate(STAGES):
            ax = axes[c]
            trial = results[s]
            traj = trial.trajectory
            ti = min(frame, len(traj)-1)
            ts = traj[ti]
            cleanup(ax)

            # Movement trace
            xs = [traj[i].obj_x for i in range(ti+1)]
            ys = [traj[i].obj_y for i in range(ti+1)]
            trails[s].set_data(xs, ys)

            # Object rectangle at current pose
            p = rrect(ax, ts.obj_x, ts.obj_y, ow, oh, ts.obj_angle, SC[s], alpha=.88)
            p._d = True  # tag for cleanup

            # Gaze visualisation
            looking_obj = ts.gaze_target == "object"
            gc = GAZE_OBJ if looking_obj else GAZE_TGT  # colour encodes fixation target
            # Fixation point: centre of whichever entity is being looked at
            fx = ts.obj_x if looking_obj else task.goal_x
            fy = ts.obj_y if looking_obj else task.goal_y

            # Eye icon
            eye = plt.Circle((EX, EY), .035, fc=gc, ec="none", alpha=.85, zorder=10)
            eye._d = True; ax.add_patch(eye)
            pupil = plt.Circle((EX, EY), .015, fc=BG, ec="none", alpha=.9, zorder=11)
            pupil._d = True; ax.add_patch(pupil)

            # Gaze line from eye to fixation point
            ln, = ax.plot([EX, fx], [EY, fy], color=gc, lw=1.0, ls="-", alpha=.45, zorder=5)
            ln._d = True

            # Small dot on the fixated entity
            fdot = plt.Circle((fx, fy), .025, fc=gc, ec="white", lw=.4, alpha=.7, zorder=8)
            fdot._d = True; ax.add_patch(fdot)

            # Label next to eye: "obj" or "tgt"
            lbl = ax.text(EX+.08, EY, "obj" if looking_obj else "tgt",
                          fontsize=5.5, color=gc, va="center",
                          fontfamily="monospace", alpha=.8, zorder=12)
            lbl._d = True

            # Status text
            if frame >= len(traj):
                # Trial ended, show outcome
                if trial.success:
                    status[s].set_text("FITTED"); status[s].set_color(ACCENT)
                else:
                    status[s].set_text("MISSED"); status[s].set_color("#fb7185")
            else:
                # During trial, show live error
                pe, ae = ts.pos_error, ts.angle_error
                status[s].set_text(f"pos={pe:.2f} ang={ae:.2f}")
                status[s].set_color(DIM)
        return []

    anim = FuncAnimation(fig, update, frames=nf, interval=200, blit=False, repeat=True)

    if save:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"planning_cascade_{task_name}.gif")
        print(f"Saving -> {p}")
        anim.save(p, writer="pillow", fps=5, savefig_kwargs={"facecolor": BG})
        print(f"Done: {p}")
    else:
        plt.show()
    return anim


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--task", default="rotate_insert", choices=list(TASKS.keys()))
    pa.add_argument("--save", action="store_true")
    pa.add_argument("--seed", type=int, default=42)
    a = pa.parse_args()
    
    if a.save: matplotlib.use("Agg") 
    animate(a.task, a.save, a.seed)