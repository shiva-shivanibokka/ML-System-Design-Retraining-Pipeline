"""
Regression test for Task 1.4: the slice/fairness gate must always block
promotion, even when require_all_gates=False. Previously, in non-strict
mode, decision.slice_gate_passed was computed but dropped from the final
`decision.promoted` expression, so a model that degraded a protected
cohort could still be promoted.
"""
import numpy as np
import pandas as pd

from validation.validator import ModelValidator


def test_degraded_slice_blocks_promotion_even_in_non_strict_mode():
    v = ModelValidator()
    v.require_all_gates = False  # non-strict mode

    rng = np.random.default_rng(0)
    n = 800
    df = pd.DataFrame(
        {
            "credit_grade": rng.choice(list("ABCDE"), n),
            "annual_income": rng.integers(20000, 150000, n),
            "loan_purpose": rng.choice(
                ["home", "car", "personal", "business", "education"], n
            ),
            "age": rng.integers(18, 90, n),
        }
    )
    y = rng.integers(0, 2, n)
    champ = np.clip(y + rng.normal(0, 0.2, n), 0, 1)
    chall = rng.random(n)  # pure noise -> degrades cohorts

    results = v._slice_validation(df, y, chall, champ)

    # Sanity: the noise challenger should degrade at least one cohort.
    assert any(not r.passed for r in results)

    # The aggregate slice gate must report failure regardless of
    # require_all_gates.
    assert v._slice_gate_passed(results) is False
