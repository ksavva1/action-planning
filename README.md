# action-planning

A computational model of the real-time planning cascade in the development of
manual action.

`planning_model` contains the code referenced in the dissertation, `prev_iterations` contains older versions.

Outputs have been intentionally left in Jupyer notebooks for initial ease of viewing.

## Layout of planning_model

| File | Purpose |
|---|---|
| `planning_cascade_model.py` | The six-layer model and `run_trial`. |
| `model_utils.py` | Parameter/task/result containers, seeding policy, censoring-aware aggregation. |
| `affordance_matrices.py` | The five named affordance weight matrices. |
| `experiment_config.py` | Profiles, task batteries, sweep baselines and ranges, seed sets. All design choices live here. |
| `analysis.py` | Uncertainty, Monte Carlo convergence, seed-set replication, sensitivity index, robustness checks. |
| `plots.py` | All figures. |
| `run_experiments.py` | Runs the three experiments and every robustness check, writing JSON to `results/`. |
| `animate_model.py` | Trial animation helpers. |
| `dev_stages.ipynb` | Experiment 1: developmental profiles across the battery. |
| `parameter_sweep.ipynb` | Experiment 2: local one-factor-at-a-time sensitivity. |
| `affordance_matrix_experiments.ipynb` | Experiment 3: robustness to the affordance mapping. |

## Reproducing the results
```bash
pip install -r requirements.txt
python run_experiments.py --out results     # ~30 minutes single-threaded
```

Then run the three notebooks, which load `results/*.json`. Use `--quick` for a
smoke test; it does not reproduce the reported figures. Individual phases can be
run with `--only exp1,robust,exp2,exp3`.