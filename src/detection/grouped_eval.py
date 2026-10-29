"""
Case-level error bars, shared by the final-iteration analyses (#35, #36, #39).

A case's fields are not independent samples. The perturbation benchmark scores
1 clean and 30 damaged fields of the same trained model, and the natural-error
study scores 7 checkpoints of the same training run. Resampling single fields
would treat them as independent, and would also break up each case's fixed set
of damage types and strengths, so its error bars could be too narrow or too
wide. So:

    case_bootstrap()   redraws whole cases, never single fields, keeping the
                       number of cases per k, and returns a metric with its
                       95% percentile range. It works with any metric.
    clopper_pearson()  gives the exact 95% range of a count, such as
                       "2 false alarms out of 30 clean fields".

Example:
    from sklearn.metrics import roc_auc_score
    from src.detection.grouped_eval import case_bootstrap, clopper_pearson

    def auc(fields):
        return roc_auc_score(fields["label"] == "hallucinated", fields["Score3_PHS_full"])

    estimate, low, high, n_boot = case_bootstrap(scores_df, auc)
    low, high = clopper_pearson(2, 30)
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import beta


class BootstrapCI(NamedTuple):
    estimate: float  # stat_fn on the data as given
    low: float       # Lower end of the range
    high: float      # Upper end of the range
    n_boot: int      # Resamples the range is based on: all of them, unless stat_fn returned NaN on some


def case_bootstrap(df: pd.DataFrame, stat_fn, n_boot: int = 2000, seed: int = 0, strata="k",
                   case_col: str = "case_id", alpha: float = 0.05) -> BootstrapCI:
    """
    Returns stat_fn(df) and its case-level bootstrap range.

    Each resample draws, separately for every value of `strata` (every k by
    default), as many cases as that value has, at random and with replacement,
    and stacks all the rows of every drawn case. A case's fields therefore always
    stay together, and the number of cases per k never changes. The range is the
    alpha/2 and 1 - alpha/2 percentiles of stat_fn over the resamples.

    A case drawn twice brings its rows twice and keeps its case_id. stat_fn
    should therefore compute its metric over rows (AUC, recall, F1, a mean, ...),
    not group by case_id, which would merge the two copies into one case.

    A resample on which stat_fn returns NaN, because the metric is undefined
    there (for example recall when no drawn field is damaged), is left out of
    the range, and the result's n_boot says how many resamples were used.

    Inputs:
        df (pd.DataFrame): One row per field, with case_col and strata columns.
        stat_fn (callable): Takes a DataFrame shaped like df and returns a float.
        n_boot (int): Number of resamples.
        seed (int): Seed of the resampling. The same seed and data give the same result.
        strata (str | None): Column whose number of cases per value is kept in every
                             resample. None resamples all cases together.
        case_col (str): Column naming each row's case.
        alpha (float): 0.05 gives the 95% range.

    Outputs:
        BootstrapCI: (estimate, low, high, n_boot).
    """
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    check_cols = [case_col] if strata is None else [case_col, strata]
    missing = [col for col in check_cols if df[col].isna().any()]
    if missing:
        raise ValueError(f"Columns {missing} have missing values")

    cases = df.groupby(case_col, sort=True)
    rows_of_case = cases.indices  # {case_id: positions of the case's rows in df}
    if strata is None:
        case_groups = [sorted(rows_of_case)]
    else:
        n_values = cases[strata].nunique()
        mixed = n_values.index[n_values > 1].tolist()
        if mixed:
            raise ValueError(f"Each case needs a single '{strata}' value, but {mixed} have several")
        stratum_of_case = cases[strata].first()
        case_groups = [stratum_of_case.index[stratum_of_case == value].tolist()
                       for value in sorted(stratum_of_case.unique())]
    row_groups = [[rows_of_case[case_id] for case_id in group] for group in case_groups]

    estimate = float(stat_fn(df))

    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for b in range(n_boot):
        drawn = [group[i] for group in row_groups for i in rng.integers(len(group), size=len(group))]
        # A fresh index, since a case drawn twice would otherwise repeat its index labels
        values[b] = stat_fn(df.take(np.concatenate(drawn)).reset_index(drop=True))

    used = values[~np.isnan(values)]
    if len(used) == 0:
        return BootstrapCI(estimate, np.nan, np.nan, 0)
    low, high = np.percentile(used, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(estimate, float(low), float(high), len(used))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Returns the exact (Clopper-Pearson) range of the rate k/n, for example
    clopper_pearson(2, 30) for 2 false alarms out of 30 clean fields.

    This is the interval of scipy.stats.binomtest(k, n).proportion_ci(method="exact"),
    written with the beta distribution.

    Inputs:
        k (int): The count, from 0 to n.
        n (int): Out of how many.
        alpha (float): 0.05 gives the 95% range.

    Outputs:
        (float, float): The lower and upper ends of the range, as rates between 0 and 1.
    """
    if k != int(k) or n != int(n) or not 0 <= k <= n or n < 1:
        raise ValueError(f"Need whole numbers with 0 <= k <= n and n >= 1, got k={k}, n={n}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    low = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    high = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(low), float(high)
