# RESULTS: what this pipeline actually did, and what it only appeared to do

Branch `sop-eval`, based on `main` at `8624ed7`. Measured on this machine on
**2026-10-07**.

**Environment.** Python **3.12.3** in a clean venv from the repo's own pinned
`requirements.txt` + `requirements-dev.txt` (mlflow 2.13.0, pytest 8.2.2, ruff
0.5.0, pandas 2.2.2, numpy 1.26.4). Building that venv is itself part of the
findings — see §6.

**Data caveat, stated up front.** The 36 batch parquets under `data/processed/`
are **not committed** — `data/.gitignore` excludes them and git tracks only the
`.dvc` pointers. Every number below is reproducible from a script, but only after
a successful `dvc pull` against a remote whose availability is not verified here.
A reader who cannot fetch the data cannot re-run these figures.

**Headline.** The promotion gate scored the champion on rows it had trained on,
and the inflation that produced (**+0.0181 AUC**) is **3.6× the improvement floor
a challenger has to clear** — so an honestly better challenger could be rejected
by arithmetic alone. The drift trigger fires on 100% of batches. And three
numbers on the live dashboard were wrong in ways a reader could not have caught
from outside.

| # | Defect | Status | §|
|---|---|---|---|
| 1 | Promotion gate scored the champion partly in-sample | **Fixed, inflation measured** | §1 |
| 2 | A conclusive bootstrap CI reported as "not conclusive" | **Fixed** | §2 |
| 3 | `numpy.bool_` serialised as `"True"` | **Fixed** | §3 |
| 4 | Training window filtered on a column that does not exist | **Fixed — with a side effect, §4** | §4 |
| 5 | Drift trigger fires on every batch | **Measured, not retuned** | §5 |
| 6 | The documented install does not resolve | **Fixed** | §6 |
| 7 | The documented run command crashes | **Fixed** | §7 |
| 8 | `import lightgbm` fails in the root Docker image | **Fixed (not built here)** | §8 |
| 9 | README described slice cohorts that do not exist | **Fixed** | §9 |

Tests: **110 → 130 passed**, `ruff check .` clean. 20 of the 23 tests across the
three files this branch touches were confirmed red by reverting the source fix —
method, exact counts, and the three that pass either way, in §10.

> **§11 records what an independent adversarial review found wrong in the first
> version of this document.** Five claims in it were overstated or unsupported,
> including the central measurement. They are corrected in place below; §11 says
> what they were, because a write-up about unverified self-reporting that quietly
> edits its own errors is not worth much.

---

## 1. The gate scored the champion on rows it had trained on

**Mechanism.** The pipeline accumulates batches: run *N* trains on a superset of
run *N−1*'s rows. The test set was drawn by
`train_test_split(test_size=0.20, random_state=42)` **per run**, over that growing
frame. A fixed seed does not give a fixed partition when the input size changes —
it gives a different one. So run *N*'s holdout contained rows that had been in run
*N−1*'s *training* set, and run *N−1*'s model is the champion.

### Measured: how much overlap

Over the 30 consecutive (champion, challenger) run pairs the pipeline actually
produces:

| | mean | median | min | max |
|---|---|---|---|---|
| Random split per run (**as shipped**) | **26.10%** | 23.91% | 14.99% | 45.05% |

`python scripts/measure_leakage.py`.

### Measured: whether that overlap actually mattered

An overlap is a necessary condition for unfairness, not a sufficient one — it
matters only if the champion scores better on the rows it memorised. Measured on
the **lowest-leakage pair in the set** (14.99% overlap, run 2018-06 → 2018-07),
LightGBM trained on the champion's split and scored on the challenger's holdout:

| champion AUC on… | |
|---|---|
| the full contaminated holdout | **0.7343** |
| rows it **had** seen | 0.7887 |
| rows it had **not** seen (honest) | **0.7162** |

**Inflation = +0.0181**, against `min_improvement = +0.0050`. The inflation alone
is **3.6× the floor**, on the most favourable pair available. A challenger better
by exactly the required margin would still be rejected.

`python scripts/measure_champion_inflation.py`.

### Fix

`training.trainer.reserve_holdout()` splits at a **date**, pinned in
`configs/config.yaml` as `training.holdout_cutoff: "2018-06-30"`. A date boundary
is disjoint from every training window by construction — including windows that
do not exist yet — and makes promotion a question about the *future*.

**There is no "0.00% after the fix" row above, deliberately.** With a date cutoff
the overlap is 0 *by construction*: `reserve_holdout` partitions on the same
column with the same literal cutoff in both calls, so
`{d > c} ∩ {d ≤ c} = ∅` is an identity, not a measurement. The first version of
this document printed it in the same table as the empirical 26.1%, which
overstated what had been learned. `measure_leakage.py --check-identity` asserts
it instead, which is all it is worth.

### A second bug inside the first fix, caught by its own test

The initial version derived the cutoff from each frame's own date range ("the
last 20%"). That boundary *floats forward* as data accumulates, so run *N*'s
training set swallowed run *N−1*'s holdout and the leakage came straight back —
`test_holdout_is_disjoint_from_every_training_window` failed with
`80 row(s) of run-2 train also appear in run-1 holdout`. `reserve_holdout` now
**requires** an explicit cutoff and raises if asked to guess.

### What this costs, which is not nothing

⚠️ **The reserved holdout is one month, not six.** The config comment describes
"the last 6 months, about 17% of the timeline". That is true of the raw date
range and **false of the frame the pipeline trains on**.
`pipelines.flows._load_all_processed_data` loads processed batches only (2015 is
the reference frame and is never in `data/processed/`) and drops batches below
`MATURE_POS_RATE_FLOOR = 0.10`. Measured:

```
training frame      236,547 rows   2016-01-01 -> 2018-07-01
trainable           232,245
holdout               4,302   (1.82% of rows)
holdout months      ['2018-07']
holdout default rate   0.1230      trainable  0.2262
```

So the gate's entire holdout is the single batch `2018-07`: 4,302 rows at a 12.3%
default rate against a 22.6% mature baseline — **the least label-mature month
that cleared the floor**. The README argues at length that immature batches are
untrustworthy; this fix puts the whole promotion decision on the most immature one
available.

Three consequences, none of them resolved here:

1. **Statistical power falls sharply.** The holdout goes from roughly 20% of
   ~247k rows (≈49k) to 4,302. The 1,000-replicate bootstrap on `delta_p5 > 0`
   gets much wider, so the fix may make the gate *harder* to clear. No power
   analysis was done. "The gate is now capable of promoting" is a statement about
   correctness, not about power.
2. **Both models are scored on censored labels.**
3. **The holdout is not stable as the data ages.** As `2018-08`…`2018-12` cross
   the maturity floor they silently join it. A different mechanism from the
   floating cutoff, and untested.

A defensible configuration would pin the cutoff *earlier* — far enough back that
the holdout is both several months long and fully matured — at the cost of
training on less recent data. That is a modelling trade-off for the repo's owner.

---

## 2. A bootstrap CI that excluded zero was reported as "not conclusive"

`validation/validator.py` computed `passed = delta_p5 > 0` and then printed
`includes 0 → not conclusive` whenever `passed` was false. There are **three**
outcomes, not two: an interval entirely *below* zero excludes zero and says the
challenger is worse.

**This shipped.** `reports/model_card_b21eb63a.json`, generated 2026-07-03 before
this branch, records:

```json
"message": "Bootstrap CI [-0.0018, -0.0003] includes 0 → not conclusive"
```

That interval excludes 0 on the losing side. (The first version of this document
cited `[-0.0120, -0.0089]` and described it as "the exact interval that shipped".
No artifact in the repo contains those numbers — it was a constructed example
presented as an observation, while a genuine one sat in the tree unused.)

**Fix.** `ModelValidator.describe_delta_interval()` handles all three cases and
returns a real `bool`:

| Interval | Message | `passed` |
|---|---|---|
| entirely > 0 | `excludes 0 → challenger better` | `True` |
| entirely < 0 | `excludes 0 → challenger worse` | `False` |
| straddles 0 | `includes 0 → not conclusive` | `False` |

Promotion semantics are unchanged. Pinned by 4 tests.

---

## 3. `numpy.bool_` reached JSON as the string `"True"`

`_slice_validation` computed `passed = delta >= -max_degrade` where `delta` is a
`numpy.float64`, so `passed` was a `numpy.bool_`. The model card is written with
`json.dump(card, f, indent=2, default=str)`, and `json` cannot serialise
`numpy.bool_` — so rather than raising, the `default=str` hook wrote the string
`"True"`.

**Reproduced** against the pre-fix code: running `main`'s `_slice_validation` on
synthetic cohorts yields `passed` of type `numpy.bool_`, and
`json.dumps(..., default=str)` emits `{"passed": "True"}`.

The frontend (`frontend/app/fairness/page.tsx:27-28`) tests `passed === true` then
`passed === false`; a string satisfies neither, so every affected cohort renders
as a neutral dash.

**What is NOT claimed.** The first version of this document said the live page
showed "0 PASS, 0 FAIL, 42 neutral". That number is **not supportable and cannot
be right**: the config defines **21** cohorts and the page renders one row per
entry, so 42 rows are unreachable. Worse, the only pre-fix card in the tree has 21
entries with genuine JSON `true`/`false`, not strings — so the one artifact that
exists contradicts the claimed observation in both count and value. The mechanism
is real and reproducible; the specific dashboard state was repeated from an
upstream review without checking it against the config or the artifact. That is
exactly the failure mode this repo is about, committed while documenting it.

**Fix.** `bool(...)` at the point of comparison, plus `SliceResult.to_json_safe()`
returning JSON-native scalars, which `_generate_model_card` now uses. The general
lesson stands: `default=str` turns a serialisation *error* into silently wrong
data.

---

## 4. The training window filtered on a column no frame here has

`compute_training_window` filtered on `batch_date`. The pipeline writes the batch
date into the **filename** (`batch_2015-03.parquet`); the frame's date column is
`issue_d`. The predicate was always false, the filter a no-op, and the auto
strategy returned on its first iteration with `n_days = auto_max_days`.

**Evidenced, not inferred:** `reports/model_card_b21eb63a.json` records
`training.window_days = 180` with `n_rows = 247,527` spanning 2015–2018.

**Fix, and its side effect.** The window now looks for `batch_date` then
`issue_d`, anchors on the latest date *in the data* (a wall-clock anchor selects
the empty set on a 2018 dataset), and reports the span of the rows it returned —
or `None` when the rows carry no usable date, because returning `0` for undated
rows is the same category of untruth as the 180 it replaced.

⚠️ **Once the predicate matches, `auto_max_days: 180` becomes binding for the
first time:**

```
compute_training_window(trainable) -> 40,245 rows, window_days 151,
                                      span 2018-01-01 -> 2018-06-01
window default rate 0.1888   (full trainable frame: 0.2262)
```

The model goes from **247,527 rows over four years** to **40,245 over 151 days**,
a ~6× reduction. That changes the model and therefore every promotion decision.
The first version of this document framed defect 4 as a reporting fix only and did
not mention it. Whether 180 days is the right window is a modelling decision; what
is not defensible is changing it by six-fold as a side effect of a logging fix and
not saying so.

---

## 5. The drift trigger fires on every batch (measured, deliberately not retuned)

**36 of 36 batches trigger a retrain — 100%.** KS flagged drift in 7 to 11 of the
11 numeric features on every batch, against a configured threshold of `>= 2`.

**The first explanation I reached for was wrong, and the sweep disproved it.** I
expected KS's sensitivity at large *n* to be the cause — and it is real: on batch
`2016-03` KS "detects drift" at effect sizes of 2–3% (`loan_amount` *D* = 0.0196,
p = 6.9e-03), where PSI flags **zero** features. But an effect-size floor does not
fix the rate:

| trigger rule | fires | rate |
|---|---|---|
| **As shipped:** KS-significant count ≥ 2 | 36/36 | **100.0%** |
| KS-significant **and** *D* ≥ 0.05, count ≥ 2 | 36/36 | **100.0%** |
| PSI critical in ≥ 1 feature | 26/36 | 72.2% |
| KS-significant **and** *D* ≥ 0.10, count ≥ 2 | 23/36 | 63.9% |

`python scripts/trigger_sweep.py`.

**The likely dominant cause is the reference set.** The reference frame is the
earliest 12 months (2015-01…2015-12, 96,000 rows) and never moves, so by 2018 the
pipeline is asking "does this month differ from 2015?" — and the answer is yes,
increasingly. Drift magnitude **rises with a clear trend** across the series,
though *not* monotonically (`max_D` falls 2016-01 → 2016-04 and 2016-09 → 2016-10;
`max_psi` falls 2018-06 → 2018-07):

| batch | max KS *D* | max PSI | PSI-critical |
|---|---|---|---|
| 2016-01 | 0.115 | 0.126 | 0 |
| 2017-01 | 0.187 | 0.209 | 1 |
| 2018-01 | 0.228 | 0.305 | 2 |
| 2018-12 | 0.299 | 0.495 | 2 |

(Four points from a 36-row series; the full series is in the script's output and
is not monotone. An earlier version of this section said "monotonically" and
showed only these four rows, which made a non-monotone series look monotone.)

**A second mechanism is not ruled out.** The batches with the largest drift are
also the most label-censored — 2018-12 has 1,243 rows at a 1.05% default rate
versus ~20% for a matured month. That tail is a survivorship-selected subset
(only loans already resolved at snapshot time), which the README's own
`MATURE_POS_RATE_FLOOR` section describes, and composition shift is an
independent driver of rising feature drift. Measuring drift against a **rolling**
12-month reference would separate the two; nothing here does, so "the reference
set is the cause" is the leading hypothesis, not a result.

**On what a threshold could do.** An earlier version said "no threshold on that
comparison can distinguish…", which the table above refutes — PSI-critical at
72.2% and KS∧*D*≥0.10 at 63.9% are both conditional. The defensible statement is
that **none of the thresholds tested is usable**: 72% is a trigger that fires
most months, which is not meaningfully different from a schedule.

**Why this is reported rather than retuned.** The fix is not a threshold change
but a change to what the reference set *is*, which alters the pipeline's
architecture and retraining cost. That belongs to the repo's owner. Picking a
number that produced a pleasing firing rate would be result-shopping.

---

## 6. The documented install does not resolve

`pip install -r requirements.txt -r requirements-dev.txt` fails with
`ResolutionImpossible`: `google-genai==1.46.0` requires `httpx>=0.28.1`, while
`requirements-dev.txt` pinned `httpx==0.27.0` for the FastAPI `TestClient`.

A second blocker: `setuptools>=81` removed `pkg_resources`, which mlflow 2.13.0
imports at module scope, so **14 test modules fail to collect** on a fresh venv.

**Fix.** `httpx==0.28.1` (verified: all **30** `TestClient`-backed tests pass on
it) and an explicit `setuptools<81`. `pip install --dry-run` now resolves, and the
repo's pre-existing suite goes from *uninstallable* to **110 passed**.

---

## 7. The documented run command crashes

```
$ python pipelines/flows.py --flow full
ModuleNotFoundError: No module named 'alerting'
```

Running a file puts its directory on `sys.path`, not the repo root. The module
form works. The CI workflow already used `-m` with a comment explaining why, so
the correct invocation was known in the repo and never reached the README, the
module docstring, or `docker-compose.yml`.

`docker-compose.yml` was **not broken** — both images set `PYTHONPATH=/app`, so
the path form works inside the container. It was changed for consistency, not to
fix a bug.

---

## 8. `import lightgbm` fails in the root Docker image

`serving/Dockerfile:10` installs `libgomp1`, with a comment noting LightGBM's
native library needs the GNU OpenMP runtime absent from `python:*-slim`. The root
`Dockerfile` — the one `docker-compose.yml` builds for the pipeline service — did
not. **Not verified by building the image**, which was not done here; the missing
package is confirmed by inspection and the failure mode is the documented one.

---

## 9. The README described slice cohorts that do not exist

Claimed "4 cohort dimensions (16 slices total)" and listed credit grades A–E (the
config has A–G), a loan-purpose list that appears nowhere in the config, and an
**age group** dimension. Actual: 4 dimensions, **21 cohorts**. Only the income
bracket was right.

The age claim is the one worth dwelling on: Lending Club does not publish borrower
age, so the dimension could not have existed — and a credit model documented as
slicing on age invites a fair-lending question this project does not answer.

**Fix.** The section now matches the config, pinned by
`tests/test_readme_slice_claims.py` (4 tests, all red against the old README).

---

## 10. How the tests were verified

Each was checked by **reverting the source fix and re-running** — never by
reasoning about what it would do. The check runs in a throwaway git worktree with
`training/`, `validation/`, `configs/`, `pipelines/` and `README.md` at `main`
and the tests at `sop-eval`:

```
20 of the 23 tests across the three touched files go red.
  tests/test_promotion_gate_defects.py   16 tests, 15 red
  tests/test_readme_slice_claims.py       4 tests,  4 red
  tests/test_feature_prep.py              (1 changed test, red)
```

Three pass either way, and none is counted as a reproduction:

- `test_split_temporal_still_uses_issue_d` — a rename guard. It asserts something
  already true, to keep it true, because defect 4 was a column-name mismatch. It
  says so in its own docstring.
- `test_prepare_features_encodes_categoricals` and
  `test_prepare_features_handles_unseen_category` — pre-existing tests in a file
  this branch touched, unrelated to these fixes.

(An earlier version of this section reported "12 of 13" and the test module's own
docstring claimed *every* test in it had been revert-verified, which was false
about that file's own verification. Both are corrected; the count rose because
four tests were added after the adversarial review.)

**Known weakness, disclosed rather than fixed:** the two tests for defect 3
construct a `SliceResult` by hand and call `to_json_safe()`. Neither exercises the
`bool(...)` in `_slice_validation` nor the `_generate_model_card` change that uses
it — revert the card writer alone and both still pass while the live page breaks.
They go red on `main` only because the method does not exist there, which is a
weaker guarantee than "the fix is pinned".

---

## 11. What an adversarial review found wrong in the first version of this file

Run after the work was complete, per the project's never-self-certify rule. Five
findings were upheld on re-measurement and are corrected above:

1. **The headline leakage figure was measured on the wrong frame.** The script
   seeded every run with the 96,000-row 2015 reference and used all 36 batches;
   the training path uses neither. The constant block diluted every pair. Claimed
   19.86% over 35 pairs; **actual 26.10% over 30**. The first version had
   explicitly overridden an upstream estimate of 24.6% with "the figures above are
   mine" — and the override was the less accurate number.
2. **"0 PASS, 0 FAIL, 42 neutral" is impossible.** 21 cohorts exist; the page
   renders one row each. Repeated from an upstream review without checking.
3. **The holdout is one censored month, not "the last 6 months, ~17%".** 4,302
   rows, 1.82%, 12.3% default rate. The statistical-power regression was
   undisclosed. (§1)
4. **Defect 4's fix cuts the training set ~6×** (247,527 → 40,245 rows), which was
   presented as a reporting fix only. (§4)
5. **"The exact interval that shipped"** cited numbers no artifact contains, while
   a genuine shipped instance sat in `reports/model_card_b21eb63a.json`. (§2)

Also corrected: "monotonically" (the series is not), "no threshold can
distinguish" (two do, just not usably), a claim that the gate "could not promote"
that was asserted rather than measured — now measured at +0.0181 against a
+0.0050 floor (§1) — Python 3.11 → 3.12.3, mlflow 2.14.x → 2.13.0, "31 TestClient
tests" → 30, and a test-module docstring claiming all of its tests were
revert-verified when 12 of 13 were.

Pattern worth recording: **everything mechanically re-derivable from a JSON file
was exact; everything that required counting or re-running was not.** The
restatements were written from the analysis rather than re-derived from the
artefacts.

---

## 12. What remains open

- **No post-fix end-to-end run**, so no corrected promotion decision is reported.
- **The holdout configuration needs a decision** (§1): one immature month is
  correct but weak. An earlier, fully-matured cutoff is the likely answer.
- **The 180-day training window is now binding** (§4) and nobody chose it.
- **The drift reference set** (§5) is an architectural decision, not a threshold.
- **The live dashboard still shows the old strings** for cards generated before
  these fixes; the card is a stored artifact and a fresh run regenerates it.
- **`prediction_psi` was `None`** throughout the sweep because no prediction
  scores were passed; whether the deployed flow populates it is unverified.
- **`scripts/measure_drift_rate.py` prints an always-empty `max_psi` column** —
  `DriftReport.to_dict()` has no such key.
