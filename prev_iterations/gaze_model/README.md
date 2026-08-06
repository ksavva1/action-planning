# Planning Model

A computational model of infant action planning in object manipulation tasks. It simulates how an agent looks between an object and a target slot, samples noisy perceptual information, maintains that information in working memory, and then plans and corrects movement.

## Background

The model captures:

- **Translation bias**: younger parameter configurations rely more on a translate-first habitual motor routine; more mature configurations rely more on goal-directed control.
- **Relational information**: displacement and angular difference between object and target can only be sampled when both have been fixated recently, governed by `simultaneous_rate`.
- **Gradual development**: later configurations have higher gaze-switching, better perceptual acuity, stronger working memory, longer planning horizon, less habit, and faster online correction.

## Architecture

```text
LOOK                    PROCESS                 ACT
┌──────────┐           ┌──────────────┐        ┌──────────────┐
│  Gaze    │──────────▶│   Working    │───────▶│    Motor     │
│Controller│           │   Memory     │        │   Planning   │
│          │           └──────┬───────┘        └──────┬───────┘
│ object ◀─┼─▶ target         │                       │
└──────────┘           ┌──────▼───────┐        ┌──────▼───────┐
      │                │  Affordance  │        │   Habitual   │
      │                │  Estimation  │        │     Bias     │
      │                └──────────────┘        └──────┬───────┘
      │                                               │
      │                                        ┌──────▼───────┐
      │                                        │    Online    │
      └────────────────────────────────────────│  Correction  │
                     feedback                  └──────────────┘
```

**Gaze Controller** - switches fixation between object and target with a dwell-time hazard model. The base rate (`gaze_switch_rate`) is dynamically modulated each timestep by four cognitive factors (see [Dynamic Gaze Coupling](#dynamic-gaze-coupling) below). Parameters: `gaze_switch_rate`, `fixation_duration_mean`, `target_bias`.

**Perceptual Sampling** - samples object features (x, y, angle, width, height) when looking at the object, target features (goal x, y, angle) when looking at the target, and relational features (dx, dy, d_angle) when both representations are recently active.

**Working Memory** - stores feature values and trace strengths. Attended features decay slowly; unattended and relational features decay faster. Capacity limits weaken the least-active traces.

**Affordance Estimation** - uses a hand-coded weight matrix, selected by `affordance_matrix_variant`, scaled by `affordance_coupling`, and jittered per trial, to map working memory to reach, grasp, rotate, and translate affordances.

**Motor Planning** - blends affordance-driven action with direct goal-error control. Longer `planning_horizon` values make commands more anticipatory.

**Habitual Bias** - adds a translate-first routine whose influence scales with `habit_strength` and fades across a trial.

**Online Correction** - applies delayed error correction; mature stages use shorter delays and stronger correction rates.

## Files

| File | Description |
| ---- | ----------- |
| `model_utils.py` | `DevelopmentalParams`, `TaskConfig`, `TimestepRecord`, and `TrialResult` dataclasses; batch simulation (`run_simulation`), grouping, aggregation, and JSON helpers. |
| `planning_cascade_model.py` | Core model state, layer functions, and `run_trial()`. |
| `affordance_matrices.py` | Named affordance-layer matrix variants and feature/action labels. |
| `animate_model.py` | Matplotlib animation library - call `animate(task, stages)` from a notebook. |
| `experiments.ipynb` | Cross-task × stage experiment: 3 × 3 distance/rotation grid, success heatmaps, metric bar charts, trajectory plots. |
| `parameter_sweep.ipynb` | One parameter at a time sensitivity sweep across all 20 `DevelopmentalParams` fields; heatmap, per-group plots, and an animated 6 × 3 trial comparison. |
| `affordance_matrix_experiments.ipynb` | Compares every affordance matrix variant across developmental stages with tables, heatmaps, and trajectory plots. |
| `param_sweep_animation.html` | Pre-rendered HTML animation from `parameter_sweep.ipynb`. |
| `output_docs/` | Generated documentation output. |
| `requirements.txt` | Python dependencies. |

## Dynamic Gaze Coupling

`gaze_switch_rate` is a baseline rate, not a fixed probability. Each timestep `step_gaze` computes an **effective rate** by multiplying the baseline by two dynamic factors and an effective fixation duration derived from noise:

| Factor | Parameters read | Direction |
| ------ | --------------- | --------- |
| **WM saturation** | `wm.trace_strength` (fixated slice) | High trace on current item → diminishing returns → switch sooner |
| **Decay urgency** | `wm.trace_strength` (unfixated slice) | Trace on other item has fallen below 0.5 → urgency to refresh → switch sooner |
| **Noise / acuity stretch** | `perceptual_noise`, `location_acuity`, `orientation_acuity` | Higher noise or lower acuity → longer effective `fixation_duration_mean` → hazard builds more slowly → longer dwell before switching |
| **Planning horizon bias** | `planning_horizon` | Longer horizon adds up to +0.20 to `target_bias`, making proactive target fixations more likely when a switch does occur |
| **Pre-movement scanning** | `initiation_threshold`, current mean WM strength | Before motor onset, info-gap = threshold − mean strength inflates switch rate by up to 1.5×; fades to zero as WM approaches threshold |

The combined multiplier is `wm_multiplier × scan_multiplier` (range approximately 1.0 – 2.7); `gaze_switch_rate` therefore acts as a developmental floor rather than an absolute rate.

## Usage
Stage presets and task definitions are the notebook's responsibility. Pass `DevelopmentalParams` and `TaskConfig` objects explicitly to every library function.

### Run one trial

```python
import math
from model_utils import DevelopmentalParams, TaskConfig
from planning_cascade_model import run_trial

params = DevelopmentalParams(
    name="stage_D",
    gaze_switch_rate=0.55, fixation_duration_mean=2.0, target_bias=0.60,
    simultaneous_rate=0.50, sampling_rate=0.85, perceptual_noise=0.08,
    location_acuity=0.98, orientation_acuity=0.88, relation_acuity=0.55,
    wm_capacity=5, wm_decay=0.04, wm_unfixated_decay=0.10,
    affordance_coupling=0.85, affordance_noise=0.08, planning_horizon=6,
    motor_noise=0.08, habit_strength=0.10, goal_directed_strength=0.90,
    correction_rate=0.28, correction_delay=0, initiation_threshold=0.45,
)

task = TaskConfig(
    name="medium_rot",
    start_x=0.0, start_y=0.0, start_angle=0.0,
    goal_x=0.5,  goal_y=0.0,  goal_angle=math.pi / 4,
    obj_width=0.3, obj_height=0.5, max_timesteps=120,
)

result = run_trial(params, task, seed=42)
print(result.success, result.timesteps_used, result.final_pos_error)
```

### Run a batch simulation

```python
from model_utils import run_simulation, group_results, summarise_group

# stages and tasks are dicts defined in your notebook
results = run_simulation(stages, tasks, n_trials=25, seed=42)
groups  = group_results(results)
summary = [
    summarise_group(stage, task, trials)
    for (stage, task), trials in groups.items()
]
```

### Animate the model

`animate_model.py` is a library — call it from a notebook or script and pass stages and task explicitly:

```python
from animate_model import animate

# Show an interactive window
anim = animate(task, stages, seed=42)

# Save a GIF next to animate_model.py
anim = animate(task, stages, save=True, seed=42)
```

### Notebooks

| Notebook | What it does |
| -------- | ------------ |
| `experiments.ipynb` | Defines 4 developmental stages and a 3 × 3 task grid (short/medium/long × 0°/45°/90°). Runs 25 trials per condition and produces success heatmaps, metric bar charts, and 2D trajectory plots. |
| `parameter_sweep.ipynb` | Sweeps each of the 20 `DevelopmentalParams` fields in isolation against a fixed baseline and medium-difficulty task. Produces a sensitivity heatmap, per-group line plots, and an animated 6 × 3 trial comparison (`param_sweep_animation.html`). |
| `affordance_matrix_experiments.ipynb` | Tests every affordance matrix variant across developmental stages. Produces matrix heatmaps, summary tables, outcome bar charts, and example trajectory plots. |

## Affordance Matrix Variants

`affordance_matrix_variant` controls the structural mapping from working-memory features to action affordances, before developmental coupling and trial noise are applied.

| Variant | Purpose |
| ------- | ------- |
| `baseline` | Original mapping. Relational features strongly drive translate and rotate. |
| `object_dominant` | Object-local pose and size drive action more strongly than relational gap features. |
| `relational_dominant` | Object-target gap features dominate translate and rotate, testing a more comparison-based strategy. |
| `diffuse` | Most features weakly activate several actions, modelling a less differentiated perception-action mapping. |
