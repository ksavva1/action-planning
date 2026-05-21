# Planning Model

A computational model of infant action planning in object manipulation tasks. It simulates how an agent looks between an object and a target slot, samples noisy perceptual information, maintains that information in working memory, and then plans and corrects movement.

## Background

The model captures:

- **Translation bias**: younger parameter configurations rely more on a translate-first habitual motor routine; more mature configurations rely more on goal-directed control.
- **Relational information**: displacement and angular difference between object and target can only be sampled when both have been fixated recently, governed by `simultaneous_rate`.
- **Gradual development**: later configurations have higher gaze-switching, better perceptual acuity, stronger working memory, longer planning horizon, less habit, and faster online correction.

## Architecture

The simulation contains these components:

```
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

**Gaze Controller** - switches fixation between object and target with a dwell-time hazard model. Parameters: `gaze_switch_rate`, `fixation_duration_mean`, `target_bias`.

**Perceptual Sampling** - samples object features (x, y, angle, width, height) when looking at the object, target features (goal x, y, angle) when looking at the target, and relational features (dx, dy, d_angle) when both representations are recently active.

**Working Memory** - stores feature values and trace strengths. Attended features decay slowly; unattended and relational features decay faster. Capacity limits weaken the least-active traces.

**Affordance Estimation** - uses an interpretable hand-coded weight matrix, selected by `affordance_matrix_variant`, scaled by `affordance_coupling`, and jittered per trial, to map working memory to reach, grasp, rotate, and translate affordances.

**Motor Planning** - blends affordance-driven action with direct goal-error control. Longer `planning_horizon` values make commands more anticipatory.

**Habitual Bias** - adds a translate-first routine whose influence scales with `habit_strength` and fades across a trial.

**Online Correction** - applies delayed error correction; mature stages use shorter delays and stronger correction rates.

## Files

| File | Description |
|------|-------------|
| `affordance_matrices.py` | Named affordance-layer matrix variants and feature/action labels. |
| `affordance_matrix_experiments.ipynb` | Notebook that compares every affordance matrix variant with tables, matrix heatmaps, outcome plots, and example trajectories. |
| `model_config.py` | Developmental parameter presets, task definitions, and result dataclasses. |
| `planning_cascade_model.py` | Core model state, layer functions, and `run_trial()`. Import this module from scripts or notebooks; it is not a standalone CLI. |
| `model_utils.py` | Batch simulation and result aggregation helpers, including JSON encoding support. |
| `animate_model.py` | Matplotlib animation comparing all four developmental stages on one task. |
| `hyperparam_sweep.py` | CLI for 1D, 2D, and default grid hyperparameter sweeps. |
| `architecture_diagram.png` | Static architecture image. |
| `gaze_model_animation.mov` | Example rendered animation. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

The command examples in this section assume your current directory is `gaze_model/`.
From the repository root, prefix script paths with `gaze_model/` and use package imports such as `from gaze_model.planning_cascade_model import ...`.

### Run one trial

```bash
python3 - <<'PY'
from planning_cascade_model import DEVELOPMENTAL_STAGES, TASKS, run_trial

result = run_trial(DEVELOPMENTAL_STAGES["D"], TASKS["rotate_insert"], seed=42)
print(result.success, result.timesteps_used, result.final_pos_error, result.final_angle_error)
PY
```

From the repository root, import with the package path:

```python
from gaze_model.planning_cascade_model import DEVELOPMENTAL_STAGES, TASKS, run_trial
```

### Run a batch simulation

```bash
python3 - <<'PY'
import json
from model_utils import NumpyEncoder, compile_results, run_simulation

results = run_simulation(n_trials=20, seed=42)
with open("simulation_results.json", "w") as output_file:
    json.dump(compile_results(results), output_file, cls=NumpyEncoder, indent=2)
PY
```

### Animate the model

Show an interactive animation:

```bash
python3 animate_model.py
```

Save a GIF:

```bash
python3 animate_model.py --task rotate_insert --save --seed 42
```

The animation CLI supports these tasks: `rotate_insert`, `translate_only`, `rotate_only`, and `complex_manipulation`.

### Hyperparameter sweep

Single-parameter sweep:

```bash
python3 hyperparam_sweep.py --param gaze_switch_rate --values 0.1 0.3 0.5 0.7
```

Two-parameter sweep with heatmaps:

```bash
python3 hyperparam_sweep.py \
    --param gaze_switch_rate --values 0.1 0.25 0.4 0.55 0.7 \
    --param2 habit_strength --values2 0.1 0.3 0.5 0.7 0.9
```

Default 4-parameter grid:

```bash
python3 hyperparam_sweep.py --trials 15
```

Affordance matrix experiment:

```bash
python3 hyperparam_sweep.py \
    --param affordance_matrix_variant --values baseline object_dominant relational_dominant diffuse \
    --trials 20
```

### Affordance matrix notebook

Open `affordance_matrix_experiments.ipynb` in Jupyter to compare the variants visually. The notebook:

- displays each matrix as a heatmap,
- runs all variants across the configured tasks,
- displays a summary table,
- plots success, efficiency, and timestep metrics,
- plots example trajectories for `rotate_insert`.

Sweep outputs are written next to `hyperparam_sweep.py`:

- `sweep_results.csv`: one row per trial.
- `sweep_summary.csv`: aggregated rows by parameter combination and task.
- `sweep_plots.png`: metric plots for each swept parameter.
- `sweep_heatmap.png`: generated only for 2-parameter sweeps.

## Parameter Configurations

Four default developmental profiles are defined in `DEVELOPMENTAL_STAGES`:

| Config | Strategy | Habit | Horizon | WM | Gaze Switch |
|--------|----------|-------|---------|----|-------------|
| A | Strong translate-first habit | 78% | 1 | 2 | 0.12 |
| B | Mostly sequential | 70% | 2 | 3 | 0.25 |
| C | Mixed habit and goal-directed control | 40% | 4 | 4 | 0.40 |
| D | Mostly goal-directed control | 10% | 6 | 5 | 0.55 |

## Custom Sweep Params

Any field of `DevelopmentalParams` can be swept:

```bash
python3 hyperparam_sweep.py --param correction_rate --values 0.05 0.1 0.15 0.2 0.25 0.3
python3 hyperparam_sweep.py --param wm_decay --values 0.05 0.1 0.2 0.3 --param2 sampling_rate --values2 0.2 0.4 0.6 0.8
```

## Affordance Matrix Variants

`affordance_matrix_variant` controls the structural mapping from working-memory features to action affordances before developmental coupling and trial noise are applied.

Available variants:

| Variant | Purpose |
|---------|---------|
| `baseline` | Original mapping. Relational features strongly drive translate and rotate. |
| `object_dominant` | Object-local pose and size drive action more strongly than relational gap features. |
| `relational_dominant` | Object-target gap features dominate translate and rotate, testing a more comparison-based strategy. |
| `diffuse` | Most features weakly activate several actions, modelling a less differentiated perception-action mapping. |
