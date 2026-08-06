"""Uncertainty, convergence and robustness analysis for the three experiments."""

import math
from dataclasses import replace

import numpy as np
from scipy import stats

from experiment_config import INFORMATIVE_BAND, SENSITIVITY_METRICS
from model_utils import make_rng, summarise_group


def add_success_interval(cell: dict) -> dict:
    """Add success rate and Wilson interval fields to a count cell."""

    cell["success_rate"] = cell["successes"] / cell["n"]
    cell["ci_low"], cell["ci_high"] = wilson_interval(cell["successes"], cell["n"])
    return cell


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: number of successes observed.
        n: number of trials.
        z: z-score for the desired confidence level.

    Returns:
        Tuple of (lower, upper) bounds; ``(nan, nan)`` if ``n`` is 0.
    """

    if n == 0:
        return float("nan"), float("nan")
    proportion = successes / n
    denominator = 1 + z**2 / n
    centre = (proportion + z**2 / (2 * n)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / n + z**2 / (4 * n**2))
        / denominator
    )
    return float(centre - half_width), float(centre + half_width)


def bootstrap_ci(
    values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of a continuous measure.

    Args:
        values: measurements, possibly containing ``None`` or NaN entries.
        n_boot: number of bootstrap resamples.
        alpha: two-sided significance level.
        seed: seed for the resampling generator.

    Returns:
        Tuple of (lower, upper) bounds; ``(nan, nan)`` if fewer than two
        defined values remain.
    """

    defined = np.array([value for value in values if value is not None], dtype=float)
    defined = defined[~np.isnan(defined)]
    if len(defined) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(defined, size=(n_boot, len(defined)), replace=True).mean(axis=1)
    return float(np.quantile(draws, alpha / 2)), float(
        np.quantile(draws, 1 - alpha / 2)
    )


def summarise_trials(trials, label: str = "condition", task: str = "task") -> dict:
    """Summarise a list of trials, for sweep conditions that have no profile.

    Args:
        trials: list of TrialResult objects.
        label: condition label to attach to the summary.
        task: task name to attach to the summary.

    Returns:
        Summary dict from :func:`model_utils.summarise_group`.
    """

    return summarise_group(label, task, trials)


# Running conditions in independent batches
def run_batched(
    conditions: dict,
    seed_sets,
    trials_per_set: int,
    seed_policy: str = "independent",
    progress: bool = False,
) -> dict:
    """
    Run every condition once per base seed and keep the batches separate.

    Args:
        conditions: mapping from condition label to ``(params, task)``.
        seed_sets: iterable of base seeds, one per independent batch.
        trials_per_set: trials run within each batch.
        seed_policy: passed to :func:`model_utils.make_rng`. "independent" gives
            every (condition, batch, trial) triple its own spawned stream.
        progress: print a line per batch.

    Returns:
        ``{label: {"trials": [...], "batches": {seed: [...]}}}``. Pooling the
        batches gives the headline estimate; comparing them gives the
        seed-robustness check.
    """

    from planning_cascade_model import run_trial

    output = {}
    for label, (params, task) in conditions.items():
        batches = {}
        for base_seed in seed_sets:
            batch = []
            for trial in range(trials_per_set):
                rng = make_rng(base_seed, label, trial, seed_policy)
                batch.append(run_trial(params, task, trial_id=trial, rng=rng))
            batches[base_seed] = batch
        output[label] = {
            "batches": batches,
            "trials": [trial for batch in batches.values() for trial in batch],
        }
        if progress:
            print(f"  {label}: {sum(len(b) for b in batches.values())} trials")
    return output


def between_batch_spread(
    batches: dict, statistic=lambda trials: np.mean([t.success for t in trials])
) -> dict:
    """Spread of a statistic across independent seed batches.

    Args:
        batches: mapping from seed to a list of trials.
        statistic: function computing a scalar from a list of trials.

    Returns:
        Dict with the per-batch values and their count, mean, SD, min, max and
        range.
    """

    values = np.array([statistic(batch) for batch in batches.values()], dtype=float)
    return {
        "n_batches": len(values),
        "values": values.tolist(),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
    }


# Monte Carlo convergence
def convergence_curve(trials, grid=None, statistic="success") -> list[dict]:
    """
    Running estimate and interval half-width as trials accumulate.

    Args:
        trials: the trials of one condition, in the order they were run.
        grid: trial counts at which to evaluate; defaults to a log-ish ladder.
        statistic: "success" for a proportion (Wilson interval) or the name of a
            continuous attribute (bootstrap interval).

    Returns:
        A list of dicts with the estimate and half-width at each n.
    """

    if grid is None:
        grid = [
            n
            for n in (10, 20, 25, 50, 75, 100, 125, 150, 200, 300, 400, 500)
            if n <= len(trials)
        ]
        if grid and grid[-1] != len(trials):
            grid.append(len(trials))

    curve = []
    for n in grid:
        subset = trials[:n]
        if statistic == "success":
            successes = int(
                sum(
                    (
                        bool(item)
                        if isinstance(item, (bool, np.bool_))
                        else bool(item.success)
                    )
                    for item in subset
                )
            )
            low, high = wilson_interval(successes, n)
            estimate = successes / n
        else:
            values = [getattr(trial, statistic) for trial in subset]
            defined = np.array([v for v in values if v is not None], dtype=float)
            estimate = float(np.mean(defined)) if len(defined) else float("nan")
            low, high = bootstrap_ci(values)
        curve.append(
            {
                "n": n,
                "estimate": float(estimate),
                "ci_low": low,
                "ci_high": high,
                "half_width": (
                    (high - low) / 2 if not math.isnan(high) else float("nan")
                ),
            }
        )
    return curve


def convergence_report(curves: dict, target_half_width: float = 0.05) -> dict:
    """Summarise convergence across conditions.

    Args:
        curves: mapping from condition label to a curve from
            :func:`convergence_curve`.
        target_half_width: half-width threshold defining "converged".

    Returns:
        Mapping from condition label to the n needed, final estimate, final
        half-width, and maximum post-target drift.
    """

    report = {}
    for label, curve in curves.items():
        reached = next(
            (point["n"] for point in curve if point["half_width"] <= target_half_width),
            None,
        )
        final = curve[-1]["estimate"]
        drift = max(
            abs(point["estimate"] - final)
            for point in curve
            if point["n"] >= (reached or curve[-1]["n"])
        )
        report[label] = {
            "n_for_target": reached,
            "final_estimate": final,
            "final_half_width": curve[-1]["half_width"],
            "max_drift_after_target": float(drift),
        }
    return report


# Sensitivity index
def relative_sensitivity_index(sweep_summaries: dict, metrics=None) -> dict:
    """
    Descriptive sensitivity of each parameter, per metric and averaged.

    Args:
        sweep_summaries: ``{parameter: {value: summary_dict}}``.
        metrics: list of ``(key, label)`` pairs; defaults to
            :data:`experiment_config.SENSITIVITY_METRICS`.

    Returns:
        ``{"raw": ..., "normalised": ..., "mean_index": ..., "order": [...]}``.
    """

    metrics = metrics or SENSITIVITY_METRICS
    keys = [key for key, _ in metrics]

    raw = {}
    for parameter, by_value in sweep_summaries.items():
        raw[parameter] = {}
        for key in keys:
            values = np.array(
                [by_value[value].get(key, np.nan) for value in by_value], dtype=float
            )
            values = values[~np.isnan(values)]
            raw[parameter][key] = (
                float(np.max(values) - np.min(values))
                if len(values) > 1
                else float("nan")
            )

    normalised = {parameter: {} for parameter in raw}
    for key in keys:
        column = np.array([raw[parameter][key] for parameter in raw], dtype=float)
        largest = np.nanmax(column) if np.any(~np.isnan(column)) else np.nan
        for parameter in raw:
            value = raw[parameter][key]
            normalised[parameter][key] = (
                float(value / largest)
                if largest and largest > 0 and not math.isnan(value)
                else 0.0
            )

    mean_index = {
        parameter: float(np.mean(list(values.values())))
        for parameter, values in normalised.items()
    }
    order = sorted(mean_index, key=lambda parameter: -mean_index[parameter])
    return {
        "raw": raw,
        "normalised": normalised,
        "mean_index": mean_index,
        "order": order,
    }


def rank_stability(rankings: dict) -> dict:
    """
    Agreement between parameter rankings obtained under different designs.

    Args:
        rankings: ``{design_label: {parameter: index}}``.

    Returns:
        Pairwise Kendall tau and Spearman rho, the mean rank of each parameter
        across designs, and how far each parameter's rank moves.
    """

    labels = list(rankings)
    parameters = sorted(set().union(*(set(ranking) for ranking in rankings.values())))

    ranks = {}
    for label in labels:
        ordered = sorted(
            parameters, key=lambda parameter: -rankings[label].get(parameter, 0.0)
        )
        ranks[label] = {
            parameter: position + 1 for position, parameter in enumerate(ordered)
        }

    pairwise = {}
    for i, first in enumerate(labels):
        for second in labels[i + 1 :]:
            a = [ranks[first][parameter] for parameter in parameters]
            b = [ranks[second][parameter] for parameter in parameters]
            pairwise[f"{first} vs {second}"] = {
                "kendall_tau": float(stats.kendalltau(a, b).statistic),
                "spearman_rho": float(stats.spearmanr(a, b).statistic),
            }

    per_parameter = {
        parameter: {
            "mean_rank": float(np.mean([ranks[label][parameter] for label in labels])),
            "min_rank": int(min(ranks[label][parameter] for label in labels)),
            "max_rank": int(max(ranks[label][parameter] for label in labels)),
        }
        for parameter in parameters
    }
    return {"ranks": ranks, "pairwise": pairwise, "per_parameter": per_parameter}


# Experiment 1 marginal analysis
def marginal_by_factor(summaries, task_meta, factor: str) -> dict:
    """Success rate for each profile at each level of one task factor.

    Returns counts as well as proportions so that intervals can be computed on
    the pooled trials rather than on an average of per-task proportions.

    Args:
        summaries: list of per-(profile, task) summary dicts.
        task_meta: mapping from task name to its metadata dict.
        factor: key into each task's metadata giving its factor level.

    Returns:
        Nested dict ``{profile: {level: cell}}`` with success counts, rate and
        Wilson interval.
    """

    output = {}
    for row in summaries:
        level = task_meta[row["task"]][factor]
        cell = output.setdefault(row["stage"], {}).setdefault(
            level, {"successes": 0, "n": 0}
        )
        cell["successes"] += row["successes"]
        cell["n"] += row["n_trials"]
    for profile, levels in output.items():
        for level, cell in levels.items():
            add_success_interval(cell)
    return output


def demand_contrasts(marginals: dict) -> dict:
    """Change in success rate from the least to the most demanding level, per profile.

    Args:
        marginals: mapping from profile to {level: cell}, as returned by
            :func:`marginal_by_factor`.

    Returns:
        Mapping from profile to the easiest/hardest rates, their difference,
        standard error and 95% interval.
    """

    contrasts = {}
    for profile, levels in marginals.items():
        ordered = sorted(levels)
        first, last = levels[ordered[0]], levels[ordered[-1]]
        difference = last["success_rate"] - first["success_rate"]
        standard_error = math.sqrt(
            first["success_rate"] * (1 - first["success_rate"]) / first["n"]
            + last["success_rate"] * (1 - last["success_rate"]) / last["n"]
        )
        contrasts[profile] = {
            "easiest": first["success_rate"],
            "hardest": last["success_rate"],
            "change": difference,
            "se": standard_error,
            "ci_low": difference - 1.96 * standard_error,
            "ci_high": difference + 1.96 * standard_error,
        }
    return contrasts


def elongation_effect(summaries, task_meta) -> dict:
    """Aspect-ratio effect

    Args:
        summaries: list of per-(profile, task) summary dicts.
        task_meta: mapping from task name to its metadata dict.

    Returns:
        Dict with ``by_shape`` and ``by_log_aspect`` cell tables.
    """

    by_shape, by_log = {}, {}
    for row in summaries:
        meta = task_meta[row["task"]]
        for store, key in (
            (by_shape, meta["shape"]),
            (by_log, round(meta["log_aspect"], 3)),
        ):
            cell = store.setdefault(row["stage"], {}).setdefault(
                key, {"successes": 0, "n": 0}
            )
            cell["successes"] += row["successes"]
            cell["n"] += row["n_trials"]
    for store in (by_shape, by_log):
        for levels in store.values():
            for cell in levels.values():
                add_success_interval(cell)
    return {"by_shape": by_shape, "by_log_aspect": by_log}


# Design robustness
def compare_designs(design_results: dict, statistic: str = "success_rate") -> dict:
    """Compare a per-profile statistic across alternative design choices.

    Args:
        design_results: mapping from design label to {profile: summary dict}.
        statistic: key of the summary field to compare.

    Returns:
        Dict with the per-design table, per-design orderings, whether the
        ordering is preserved across designs, and each profile's max shift.
    """

    profiles = sorted(next(iter(design_results.values())))
    table = {
        design: {profile: values[profile][statistic] for profile in profiles}
        for design, values in design_results.items()
    }
    orderings = {
        design: sorted(profiles, key=lambda profile: row[profile])
        for design, row in table.items()
    }
    reference = orderings[next(iter(orderings))]
    return {
        "table": table,
        "orderings": orderings,
        "ordering_preserved": all(order == reference for order in orderings.values()),
        "max_shift": {
            profile: float(
                max(row[profile] for row in table.values())
                - min(row[profile] for row in table.values())
            )
            for profile in profiles
        },
    }


def matrix_separation_vs_jitter(
    variants: dict, coupling: float, jitter_sd: float
) -> dict:
    """Size of the per-trial affordance jitter relative to the difference between variants.

    Args:
        variants: mapping from variant name to its base weight matrix.
        coupling: affordance coupling strength applied to the base matrices.
        jitter_sd: standard deviation of the per-trial Gaussian jitter.

    Returns:
        Dict with the mean and minimum pairwise variant separation, the
        expected absolute jitter, their ratio, the fraction of zero entries,
        and the clipping bias on those entries.
    """

    names = list(variants)
    separations = [
        float(np.abs(variants[a] * coupling - variants[b] * coupling).mean())
        for index, a in enumerate(names)
        for b in names[index + 1 :]
    ]
    mean_separation = float(np.mean(separations))
    expected_jitter = jitter_sd * math.sqrt(2 / math.pi)
    zero_fraction = float(np.mean([np.mean(variants[name] == 0) for name in names]))
    return {
        "coupling": coupling,
        "mean_variant_separation": mean_separation,
        "min_variant_separation": float(np.min(separations)),
        "expected_absolute_jitter": expected_jitter,
        "jitter_to_separation_ratio": (
            expected_jitter / mean_separation if mean_separation else float("nan")
        ),
        "zero_entry_fraction": zero_fraction,
        "clipping_bias_on_zero_entries": jitter_sd / math.sqrt(2 * math.pi),
    }


def informative_cells(summaries, band=INFORMATIVE_BAND) -> dict:
    """Flag which (profile, task) cells can express a between-condition difference.

    Args:
        summaries: list of per-(profile, task) summary dicts.
        band: (low, high) success-rate bounds defining "informative".

    Returns:
        Nested dict ``{profile: {task: bool}}``.
    """

    low, high = band
    flags = {}
    for row in summaries:
        flags.setdefault(row["stage"], {})[row["task"]] = bool(
            low <= row["success_rate"] <= high
        )
    return flags


def matrix_contrast(summaries_by_variant: dict) -> dict:
    """Best-versus-worst matrix contrast within one cell, with counts and interval.

    Args:
        summaries_by_variant: mapping from matrix variant name to its summary
            dict for this cell.

    Returns:
        Dict identifying the best and worst variants, their success counts,
        the difference in rate, its standard error, 95% interval, and whether
        the interval excludes zero.
    """

    variants = list(summaries_by_variant)
    rates = {
        variant: summaries_by_variant[variant]["success_rate"] for variant in variants
    }
    best = max(rates, key=rates.get)
    worst = min(rates, key=rates.get)
    best_row, worst_row = summaries_by_variant[best], summaries_by_variant[worst]
    difference = rates[best] - rates[worst]
    standard_error = math.sqrt(
        rates[best] * (1 - rates[best]) / best_row["n_trials"]
        + rates[worst] * (1 - rates[worst]) / worst_row["n_trials"]
    )
    return {
        "best": best,
        "worst": worst,
        "best_successes": best_row["successes"],
        "best_n": best_row["n_trials"],
        "worst_successes": worst_row["successes"],
        "worst_n": worst_row["n_trials"],
        "difference": difference,
        "se": standard_error,
        "ci_low": difference - 1.96 * standard_error,
        "ci_high": difference + 1.96 * standard_error,
        "excludes_zero": (difference - 1.96 * standard_error) > 0,
    }


def with_overrides(params, **overrides):
    """Return a copy of a parameter set with the given fields replaced.

    Args:
        params: a dataclass instance (typically DevelopmentalParams).
        **overrides: field names and replacement values.

    Returns:
        A new instance with the given fields overridden.
    """

    return replace(params, **overrides)
