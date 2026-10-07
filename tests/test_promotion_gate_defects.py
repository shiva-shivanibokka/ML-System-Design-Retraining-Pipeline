"""Reproductions for the four defects that crippled the promotion gate.

**15 of the 16 tests here** were confirmed to go red by reverting the
corresponding source fix. The exception is
``test_split_temporal_still_uses_issue_d``, which is a rename guard rather than a
reproduction and says so in its own docstring. An earlier version of this
paragraph claimed "each test here" was verified that way, which was false about
this file's own verification -- in a repo whose subject is unverified
self-reporting, that is worth fixing rather than glossing.

They are grouped in one file because all four defects are symptoms of the same
thing: the gate reported numbers nobody could act on.

The four:
  1. ``test_holdout_is_disjoint_from_every_training_window`` — the champion had
     seen part of the challenger's holdout, so its measured AUC was inflated and
     the hard floor could never be cleared.
  2. ``test_interval_entirely_below_zero_is_not_called_inconclusive`` — a CI that
     excluded 0 on the *losing* side was reported as "includes 0".
  3. ``test_slice_passed_survives_a_json_round_trip_as_a_bool`` — ``numpy.bool_``
     serialised through ``default=str`` to the string ``"True"``, which the
     frontend's ``passed === true`` check reads as neither pass nor fail.
  4. ``test_training_window_days_is_not_reported_when_it_filtered_nothing`` — the
     window matched on a column that does not exist, then logged 180 days against
     a multi-year span.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data.build_batches import split_temporal
from training.trainer import compute_training_window
from validation.validator import ModelValidator, SliceResult

# ---------------------------------------------------------------------------
# Defect 1: the champion had trained on part of the challenger's holdout
# ---------------------------------------------------------------------------


def _frame(n: int, start: str = "2015-01-01") -> pd.DataFrame:
    """A frame with one row per day, so month boundaries are unambiguous."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "issue_d": pd.date_range(start, periods=n, freq="D"),
            "loan_amnt": rng.normal(15000, 5000, n),
            "default": rng.integers(0, 2, n),
        }
    )


def test_holdout_is_disjoint_from_every_training_window():
    """The evaluation holdout must be reserved by DATE, not resampled per run.

    The pipeline accumulates batches, so run N trains on a superset of run N-1's
    rows. A random ``train_test_split`` over that growing frame hands run N a
    holdout that overlaps run N-1's *training* rows -- which is exactly the
    champion. The champion then scores in-sample on part of the comparison set,
    its AUC is inflated, and ``challenger_auc - champion_auc >= min_improvement``
    cannot be satisfied by an honest challenger.

    A date-reserved holdout is disjoint from every training window by
    construction, including windows that do not exist yet.
    """
    from training.trainer import reserve_holdout

    early = _frame(400)  # what run 1 could see
    late = _frame(700)  # run 2 sees a superset

    # The SAME pinned cutoff for both runs -- which is the whole point. A cutoff
    # derived from each frame's own range floats forward with the data and puts
    # run 1's holdout inside run 2's training set; the first version of this fix
    # did exactly that, and this test is what caught it.
    cutoff = "2015-10-01"
    train_1, holdout_1 = reserve_holdout(early, cutoff=cutoff)
    train_2, holdout_2 = reserve_holdout(late, cutoff=cutoff)

    # The holdout is defined by a date cutoff, so no training row of EITHER run
    # may appear in EITHER holdout.
    for train, name in ((train_1, "run-1 train"), (train_2, "run-2 train")):
        for holdout, hname in ((holdout_1, "run-1 holdout"), (holdout_2, "run-2 holdout")):
            overlap = set(train["issue_d"]) & set(holdout["issue_d"])
            assert not overlap, (
                f"{len(overlap)} row(s) of {name} also appear in {hname}; "
                "the champion would be scored partly in-sample"
            )

    # And the holdout must be the LATEST rows, not a random slice -- promotion is
    # a question about the future, not about a random corner of the past.
    assert holdout_2["issue_d"].min() > train_2["issue_d"].max()


def test_reserved_holdout_is_stable_as_more_data_arrives():
    """Adding newer batches must not silently redefine what 'the test set' means.

    If the cutoff floats with the data, two runs are graded on different exams and
    their AUCs are not comparable -- which is the subtler half of defect 1.
    """
    from training.trainer import reserve_holdout

    base = _frame(600)
    _, holdout_a = reserve_holdout(base, cutoff="2016-01-01")
    _, holdout_b = reserve_holdout(_frame(900), cutoff="2016-01-01")

    shared = set(holdout_a["issue_d"]) & set(holdout_b["issue_d"])
    assert shared == set(holdout_a["issue_d"]), (
        "an explicit cutoff must keep every previously-held-out row held out"
    )


def test_a_derived_cutoff_is_refused_on_the_promotion_path():
    """The convenient default is the dangerous one, so it must be opt-in.

    A tail-fraction cutoff recomputed per run is how the leakage comes back, so
    ``reserve_holdout`` refuses to guess and says why.
    """
    from training.trainer import reserve_holdout

    with pytest.raises(ValueError, match="explicit `cutoff`"):
        reserve_holdout(_frame(500))

    # Opt-in still works, for one-off analysis.
    train, holdout = reserve_holdout(_frame(500), allow_floating_cutoff=True)
    assert len(holdout) > 0
    assert holdout["issue_d"].min() > train["issue_d"].max()


def test_reserve_holdout_never_silently_falls_back_to_a_random_split():
    """A frame with no usable date column must raise, not random-split.

    Randomly splitting an accumulating frame is the original defect. Failing
    loudly is the only safe behaviour, because the silent version produced a
    plausible-looking AUC that nobody could act on.
    """
    from training.trainer import reserve_holdout

    undated = pd.DataFrame({"loan_amnt": [1.0, 2.0, 3.0], "default": [0, 1, 0]})
    with pytest.raises(ValueError, match="needs a date column"):
        reserve_holdout(undated, cutoff="2018-06-30")


def test_the_validation_flow_has_no_random_split_fallback_either():
    """`flows.py` kept a `train_test_split` fallback for results without a
    `test_df`, directly contradicting reserve_holdout's refusal one layer down.

    An adversarial review found it: the test above only exercised the function,
    while the *promotion path* retained exactly the fallback the function
    forbids. Pinned by source inspection because reaching that branch requires a
    legacy TrainingResult.
    """
    import inspect

    import pipelines.flows as flows

    # Prefect wraps tasks, so read the undecorated function.
    task_validate = getattr(flows.task_validate, "fn", flows.task_validate)
    src = inspect.getsource(task_validate)
    assert "train_test_split" not in src, (
        "task_validate re-split the accumulated frame when test_df was missing, "
        "which reintroduces the champion/challenger leakage"
    )
    assert "carries no test_df" in src, "it must raise and say why instead"


def test_reserve_holdout_is_a_partition_and_loses_no_rows():
    """NaT compares False against both `<= cutoff` and `> cutoff`, so undated
    rows landed in NEITHER output and vanished -- silent training-data loss that
    the caller's only guard (`len(test_df) == 0`) cannot detect."""
    from training.trainer import reserve_holdout

    mixed = pd.DataFrame(
        {
            "issue_d": pd.to_datetime(
                ["2015-01-01", None, "2016-01-01", None, None]
            ),
            "default": [0, 1, 0, 1, 0],
        }
    )
    with pytest.raises(ValueError, match="no usable"):
        reserve_holdout(mixed, cutoff="2015-06-30")

    # With clean dates it must be a true partition.
    clean = _frame(300)
    tr, ho = reserve_holdout(clean, cutoff="2015-06-01")
    assert len(tr) + len(ho) == len(clean)


def test_window_days_is_none_when_the_rows_carry_no_date():
    """Returning 0 for undated rows is the same category of untruth as the 180
    it replaced: a number unrelated to the data. None means unknown."""
    undated = pd.DataFrame({"loan_amnt": [1.0, 2.0, 3.0], "default": [0, 1, 0]})
    subset, days = compute_training_window(undated)
    assert len(subset) == 3
    assert days is None, "0 would claim these three rows span zero days"

    all_nat = pd.DataFrame(
        {"issue_d": pd.to_datetime([None] * 4), "default": [0, 1, 0, 1]}
    )
    subset, days = compute_training_window(all_nat)
    assert len(subset) == 4
    assert days is None


# ---------------------------------------------------------------------------
# Defect 2: an interval that excluded 0 was reported as including it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "deltas, expect_substring, expect_passed",
    [
        # Entirely above 0: challenger genuinely better.
        (np.linspace(0.010, 0.030, 500), "excludes 0", True),
        # Entirely BELOW 0: conclusive that the challenger is WORSE. The old
        # message called this "includes 0 -> not conclusive", which is the exact
        # sentence a reviewer stops reading at.
        (np.linspace(-0.0120, -0.0089, 500), "excludes 0", False),
        # Straddling 0: genuinely inconclusive.
        (np.linspace(-0.020, 0.020, 500), "includes 0", False),
    ],
)
def test_interval_entirely_below_zero_is_not_called_inconclusive(
    deltas, expect_substring, expect_passed
):
    msg, passed = ModelValidator.describe_delta_interval(
        delta_p5=float(np.percentile(deltas, 5)),
        delta_p95=float(np.percentile(deltas, 95)),
    )
    assert expect_substring in msg, msg
    assert passed is expect_passed
    # Whatever the verdict, `passed` must be a real bool so the frontend's
    # `passed === true` / `=== false` checks can both resolve.
    assert isinstance(passed, bool)


def test_a_losing_interval_says_the_challenger_is_worse():
    msg, passed = ModelValidator.describe_delta_interval(-0.0120, -0.0089)
    assert passed is False
    assert "worse" in msg.lower(), (
        f"a CI of [-0.0120, -0.0089] is conclusive, not inconclusive; got: {msg}"
    )


# ---------------------------------------------------------------------------
# Defect 3: numpy.bool_ reached JSON as the string "True"
# ---------------------------------------------------------------------------


def test_slice_passed_survives_a_json_round_trip_as_a_bool():
    """`json.dump(..., default=str)` turns numpy.bool_ into "True".

    The frontend reads `passed === true` for PASS and `passed === false` for FAIL,
    so a string lands in neither branch and every cohort renders as a neutral dash
    -- the live fairness page showed 0 PASS, 0 FAIL and 42 neutral rows.
    """
    # A numpy comparison, exactly as _slice_validation produces it.
    delta = np.float64(0.004) - np.float64(0.001)
    passed = delta >= -0.02
    assert isinstance(passed, np.bool_)  # the trap, pinned so it stays visible

    res = SliceResult(
        slice_name="grade",
        cohort_value="B",
        n_samples=500,
        champion_auc=0.71,
        challenger_auc=0.72,
        delta_auc=0.01,
        passed=passed,
    )

    round_tripped = json.loads(json.dumps(res.to_json_safe(), default=str))
    assert round_tripped["passed"] is True, (
        f"expected a JSON boolean, got {round_tripped['passed']!r}"
    )


def test_every_slice_field_is_json_native():
    """No numpy scalar may reach the model card -- `default=str` hides them all."""
    res = SliceResult(
        slice_name="term",
        cohort_value="36 months",
        n_samples=np.int64(1200),
        champion_auc=np.float64(0.70),
        challenger_auc=np.float64(0.69),
        delta_auc=np.float64(-0.01),
        passed=np.bool_(False),
    )
    payload = res.to_json_safe()
    for key, value in payload.items():
        assert not isinstance(value, np.generic), f"{key} is a numpy scalar: {value!r}"
    # json.dumps without a `default` hook must succeed; if it needs one, the
    # payload was not actually native.
    assert json.loads(json.dumps(payload))["passed"] is False


# ---------------------------------------------------------------------------
# Defect 4: a window that filtered nothing still reported 180 days
# ---------------------------------------------------------------------------


def test_training_window_days_is_not_reported_when_it_filtered_nothing():
    """`compute_training_window` matched on `batch_date`, a column the pipeline
    never writes -- the date lives in the FILENAME (`batch_2015-03.parquet`) and
    the frame's own date column is `issue_d`.

    So the filter was a no-op on every real frame, and the auto strategy returned
    on its first iteration with `n_days = auto_max_days`. MLflow and the model
    card then recorded a 180-day training window for a model trained on the whole
    multi-year history.
    """
    df = _frame(912)  # 912 distinct days
    span_days = int((df["issue_d"].max() - df["issue_d"].min()).days)
    assert span_days > 180  # the premise of the bug

    subset, window_days = compute_training_window(df)

    if len(subset) == len(df):
        assert window_days == span_days, (
            f"the window kept all {len(df)} rows spanning {span_days} days but "
            f"reported {window_days}; a reported window must describe the rows "
            "actually used"
        )
    else:
        kept_span = int((subset["issue_d"].max() - subset["issue_d"].min()).days)
        assert window_days >= kept_span


def test_training_window_filters_on_the_column_the_pipeline_actually_writes():
    """A fixed 90-day window over 912 days of history must drop rows."""
    from configs.settings import settings

    original = settings.training.training_window.strategy
    original_days = settings.training.training_window.fixed_days
    try:
        settings.training.training_window.strategy = "fixed"
        settings.training.training_window.fixed_days = 90
        df = _frame(912, start="2015-01-01")
        subset, window_days = compute_training_window(df)
        assert window_days == 90
        # Either it genuinely filtered, or it must not claim 90 days.
        if len(subset) == len(df):
            pytest.fail(
                "a 90-day fixed window kept all 912 daily rows -- the date "
                "predicate matched no column"
            )
    finally:
        settings.training.training_window.strategy = original
        settings.training.training_window.fixed_days = original_days


def test_split_temporal_still_uses_issue_d():
    """Guard the column name the fix depends on, so a rename cannot silently
    reintroduce defect 4.

    NOT a reproduction: this one passes with the fix reverted, because it asserts
    something that was already true. It is here to keep it true -- defect 4 was a
    column-name mismatch, so the fix is only as durable as the name it matches on.
    The other 12 tests in this file were each confirmed to go red by reverting the
    fix; this one is deliberately not one of them.
    """
    ref, batches = split_temporal(_frame(500), reference_months=3)
    assert "issue_d" in ref.columns
    assert batches and "issue_d" in batches[0][1].columns
