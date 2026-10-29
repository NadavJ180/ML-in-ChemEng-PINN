"""
Unit tests for src.detection.grouped_eval (#41).

Verifies that case_bootstrap():
  - Never splits a case: every resample is made of whole cases.
  - Keeps the number of cases per k in every resample.
  - Gives the same result for the same seed, whatever the row order.
  - Treats a case as one unit: repeating each case's field 31 times leaves the range unchanged.
  - Returns the percentiles of the resampled values, leaving out undefined (NaN) ones.
  - Refuses inputs it can't handle: an empty table, or a case with two k values.
and that clopper_pearson() matches scipy's exact binomial interval.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

from src.detection.grouped_eval import case_bootstrap, clopper_pearson

CASES_PER_K = {1: 4, 2: 3, 3: 3}


def make_fields(seed=0):
    """
    A small made-up score table with CASES_PER_K cases. Case number i has i + 2
    fields, so a resample that split a case would show up in the row counts.

    Outputs:
        pd.DataFrame: Columns case_id, k, field, value.
    """
    rng = np.random.default_rng(seed)
    rows = []
    case_number = 0
    for k, n_cases in CASES_PER_K.items():
        for _ in range(n_cases):
            for field in range(case_number + 2):
                rows.append({"case_id": f"case_{case_number:02d}", "k": k, "field": field, "value": rng.normal()})
            case_number += 1
    return pd.DataFrame(rows)


def mean_value(fields):
    return fields["value"].mean()


def bootstrap_and_record(df, **kwargs):
    """
    Runs case_bootstrap with a stat_fn that keeps every DataFrame it is given.

    Outputs:
        (BootstrapCI, list[pd.DataFrame]): The result, and the resamples (without
        the first call, which is on the full data).
    """
    seen = []

    def stat_fn(fields):
        seen.append(fields)
        return mean_value(fields)

    return case_bootstrap(df, stat_fn, **kwargs), seen[1:]


def copies_per_case(resample, df):
    """How many times each case of df was drawn: its rows in the resample over its rows in df."""
    rows_in_df = df["case_id"].value_counts()
    rows_in_resample = resample["case_id"].value_counts().reindex(rows_in_df.index, fill_value=0)
    return rows_in_resample / rows_in_df


def test_cases_are_never_split():
    df = make_fields()
    _, resamples = bootstrap_and_record(df, n_boot=50)

    assert len(resamples) == 50
    for resample in resamples:
        copies = copies_per_case(resample, df)
        assert (copies == copies.round()).all()
        # Each copy brings all of the case's fields, unchanged
        for case_id, rows in resample.groupby("case_id"):
            original = df[df["case_id"] == case_id]
            n_copies = int(copies[case_id])
            assert (sorted(rows[["field", "value"]].itertuples(index=False))
                    == sorted(list(original[["field", "value"]].itertuples(index=False)) * n_copies))


def test_cases_per_k_are_kept():
    df = make_fields()
    k_of_case = df.groupby("case_id")["k"].first()
    _, resamples = bootstrap_and_record(df, n_boot=200)

    for resample in resamples:
        assert copies_per_case(resample, df).groupby(k_of_case).sum().to_dict() == CASES_PER_K
    # Some resample draws a case more than once, so the test above has duplicates to count
    assert any((copies_per_case(resample, df) > 1).any() for resample in resamples)


def test_without_strata_only_the_total_is_kept():
    df = make_fields()
    k_of_case = df.groupby("case_id")["k"].first()
    _, resamples = bootstrap_and_record(df, n_boot=200, strata=None)

    per_k = [copies_per_case(resample, df).groupby(k_of_case).sum().to_dict() for resample in resamples]
    assert all(sum(counts.values()) == sum(CASES_PER_K.values()) for counts in per_k)
    assert any(counts != CASES_PER_K for counts in per_k)


def test_same_seed_same_result():
    df = make_fields()
    first = case_bootstrap(df, mean_value, n_boot=300, seed=7)

    assert case_bootstrap(df, mean_value, n_boot=300, seed=7) == first
    # The cases drawn depend on the seed only, not on the order of the rows
    shuffled = df.sample(frac=1, random_state=1)
    assert case_bootstrap(shuffled, mean_value, n_boot=300, seed=7) == pytest.approx(first)
    assert case_bootstrap(df, mean_value, n_boot=300, seed=8) != first


def test_a_case_counts_once_however_many_fields_it_has():
    """
    Repeating every case's field 31 times adds no information, so the range must
    not shrink. (A bootstrap over single fields would make it about sqrt(31) times narrower.)
    """
    rng = np.random.default_rng(0)
    one_field = pd.DataFrame({
        "case_id": [f"case_{i:02d}" for i in range(10)],
        "k": [1] * 4 + [2] * 3 + [3] * 3,
        "value": rng.normal(size=10),
    })
    many_fields = one_field.loc[one_field.index.repeat(31)]

    assert (case_bootstrap(many_fields, mean_value, n_boot=500, seed=3)
            == pytest.approx(case_bootstrap(one_field, mean_value, n_boot=500, seed=3)))


def test_range_is_the_percentiles_of_the_resamples():
    df = make_fields()
    result, resamples = bootstrap_and_record(df, n_boot=400, alpha=0.1)
    values = [mean_value(resample) for resample in resamples]

    assert result.estimate == mean_value(df)
    assert (result.low, result.high) == tuple(np.percentile(values, [5, 95]))
    assert result.n_boot == 400


def test_undefined_resamples_are_left_out():
    df = make_fields()

    def mean_if_case_00_drawn(fields):
        return mean_value(fields) if (fields["case_id"] == "case_00").any() else np.nan

    _, resamples = bootstrap_and_record(df, n_boot=500)
    values = np.array([mean_if_case_00_drawn(resample) for resample in resamples])
    defined = values[~np.isnan(values)]
    # case_00 is one of 4 cases with k=1, so a resample misses it with probability (3/4)^4, about 32%
    assert 0 < len(defined) < 500

    result = case_bootstrap(df, mean_if_case_00_drawn, n_boot=500)
    assert result.n_boot == len(defined)
    assert (result.low, result.high) == tuple(np.percentile(defined, [2.5, 97.5]))

    never_defined = case_bootstrap(df, lambda fields: np.nan, n_boot=50)
    assert never_defined.n_boot == 0
    assert np.isnan(never_defined.low) and np.isnan(never_defined.high)


def test_an_empty_table_is_refused():
    # A filter that leaves no rows should fail clearly, not deep inside numpy
    with pytest.raises(ValueError, match="no rows"):
        case_bootstrap(make_fields().iloc[0:0], mean_value, n_boot=10)


def test_a_case_with_two_k_values_is_refused():
    df = make_fields()
    df.loc[df.index[0], "k"] = 3  # case_00 now has fields with k=1 and k=3

    with pytest.raises(ValueError, match="case_00"):
        case_bootstrap(df, mean_value, n_boot=10)


@pytest.mark.parametrize("k, n", [(0, 30), (1, 30), (2, 30), (15, 30), (29, 30), (30, 30),
                                  (0, 10), (3, 10), (10, 10), (0, 1), (1, 1)])
@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_clopper_pearson_matches_scipy(k, n, alpha):
    expected = binomtest(k, n).proportion_ci(confidence_level=1 - alpha, method="exact")

    assert clopper_pearson(k, n, alpha) == pytest.approx((expected.low, expected.high), rel=1e-9, abs=1e-12)


def test_clopper_pearson_known_value():
    # With no false alarm out of 30, the upper end solves (1 - p)^30 = 0.025
    assert clopper_pearson(0, 30) == pytest.approx((0.0, 1 - 0.025 ** (1 / 30)))


@pytest.mark.parametrize("k, n", [(-1, 30), (31, 30), (0, 0), (2.5, 30)])
def test_clopper_pearson_refuses_impossible_counts(k, n):
    with pytest.raises(ValueError):
        clopper_pearson(k, n)
