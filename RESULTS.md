# RESULTS: what this pipeline actually did, and what it only appeared to do

Branch `sop-eval`, based on `main` at `8624ed7`. Everything below was measured on
this machine on **2026-10-07** against the 36 batch parquets committed in
`data/processed/` and the reference frame in `data/reference/`. Nothing here is
estimated, and every number has a script that regenerates it.

Environment: a clean venv from the repo's own pinned `requirements.txt` +
`requirements-dev.txt` (Python 3.11). Building that venv is itself part of the
findings — see §6.

**Headline, in one line.** The promotion gate could not promote anything, because
it compared an in-sample champion against an out-of-sample challenger; the drift
trigger fired on 100% of batches, so it was a schedule rather than a detector; and
three of the numbers on the live dashboard were wrong in ways a reader could not
have caught from the outside.

| # | Defect | Status | Evidence |
|---|---|---|---|
| 1 | Promotion gate scored the champion partly in-sample | **Fixed** | §1 |
| 2 | A conclusive bootstrap CI was reported as "not conclusive" | **Fixed** | §2 |
| 3 | `numpy.bool_` serialised as `"True"`, emptying the fairness page | **Fixed** | §3 |
| 4 | Training window filtered on a column that does not exist | **Fixed** | §4 |
| 5 | Drift trigger fires on every batch | **Measured, not retuned** | §5 |
| 6 | The documented install does not resolve | **Fixed** | §6 |
| 7 | The documented run command crashes | **Fixed** | §7 |
| 8 | `import lightgbm` fails in the root Docker image | **Fixed** | §8 |
| 9 | README described slice cohorts that do not exist | **Fixed** | §9 |

Reproduction tests live in `tests/test_promotion_gate_defects.py` (12 tests) and
`tests/test_readme_slice_claims.py` (4 tests). **Every one of those 16 was
confirmed to go red by reverting the corresponding fix**, not by reasoning about
it — the method is in §10, including the one test that did *not* discriminate and
is labelled as a guard rather than a reproduction.

---

## 1. The promotion gate compared an in-sample champion to an out-of-sample challenger

This is the defect that mattered most, because it made the gate structurally
unable to do its job.

**Mechanism.** The pipeline accumulates batches: run *N* trains on a superset of
run *N−1*'s rows. The test set was drawn by
`train_test_split(test_size=0.20, random_state=42)` **per run**, over that growing
frame. A fixed seed does not give you a fixed partition when the input size
changes — it gives you a different partition. So run *N*'s holdout contained rows
that had been in run *N−1*'s *training* set, and run *N−1*'s model is exactly the
champion the challenger is measured against.

The champion was therefore scored partly **in-sample** on the comparison set. Its
AUC was inflated, and

```
challenger_auc − champion_auc >= min_improvement
```

could not be satisfied by an honest challenger.

**Measured.** Over all 35 consecutive (champion, challenger) run pairs:

| Holdout construction | mean overlap | median | min | max |
|---|---|---|---|---|
| Random split per run (**as shipped**) | **19.86%** | 18.06% | 12.72% | 36.02% |
| Date-reserved holdout (**fixed**) | **0.00%** | 0.00% | 0.00% | 0.00% |

Reproduce: `python scripts/measure_leakage.py`.

> The original adversarial review reported this as 24.6%. My own measurement puts
> it at 19.9% on average with a maximum of 36.0%, so the figures above are mine
> and the 24.6% is not repeated anywhere in the code or docs. The conclusion is
> unchanged and the mechanism is identical; only the magnitude differs.

**Fix.** `training.trainer.reserve_holdout()` splits at a **date**, pinned in
`configs/config.yaml` as `training.holdout_cutoff: "2018-06-30"` (data spans
2015-01…2018-12, so this reserves the last 6 months, ~17% of the timeline). A date
boundary is disjoint from every training window by construction — including
windows that do not exist yet — so it stays valid as batches arrive. It also makes
promotion a question about the *future* rather than about a random corner of the
past.

**A second bug inside the first fix, caught by its own test.** My initial version
derived the cutoff from each frame's own date range ("the last 20%"). That
boundary *floats forward* as data accumulates, so run *N*'s training set swallowed
run *N−1*'s holdout and the leakage came straight back —
`test_holdout_is_disjoint_from_every_training_window` failed with
`80 row(s) of run-2 train also appear in run-1 holdout`. `reserve_holdout` now
**requires** an explicit cutoff and raises with an explanation if asked to guess;
the tail heuristic survives only behind `allow_floating_cutoff=True`, for one-off
analysis. This is the clearest evidence in this write-up that the
reproduce-before-fix rule earns its cost: the convenient version of the fix was
wrong, and only the test knew.

**What is NOT claimed.** I have not re-run the full retraining flow end to end, so
I do not report a post-fix promotion decision or a corrected champion/challenger
AUC pair. The gate is now capable of promoting; whether any particular challenger
*deserves* promotion under it is unmeasured. Saying otherwise would be exactly the
kind of claim this exercise exists to remove.

---

## 2. A bootstrap CI that excluded zero was reported as "not conclusive"

`validation/validator.py` computed `passed = delta_p5 > 0` and then wrote:

```
Bootstrap CI [{p5}, {p95}] {'excludes 0 → challenger better' if passed
                            else 'includes 0 → not conclusive'}
```

There are **three** outcomes, not two. An interval like `[-0.0120, -0.0089]`
excludes 0 entirely and says plainly that the challenger is *worse* — and it was
labelled `includes 0 → not conclusive`. That string went out on the live
dashboard, where it reads as though the pipeline cannot distinguish "no evidence"
from "evidence of harm". It can; only the message was wrong.

**Fix.** `ModelValidator.describe_delta_interval()` handles all three cases and
returns a real `bool`:

| Interval | Message | `passed` |
|---|---|---|
| entirely > 0 | `excludes 0 → challenger better` | `True` |
| entirely < 0 | `excludes 0 → challenger worse` | `False` |
| straddles 0 | `includes 0 → not conclusive` | `False` |

The promotion semantics are unchanged (promote only when the interval lies wholly
above 0). Pinned by 4 tests, including the exact `[-0.0120, -0.0089]` interval
that shipped.

---

## 3. `numpy.bool_` reached JSON as the string `"True"`, emptying the fairness page

`_slice_validation` computed `passed = delta >= -max_degrade` where `delta` is a
`numpy.float64`, so `passed` was a **`numpy.bool_`**. The model card is written
with `json.dump(card, f, indent=2, default=str)`, and `json` cannot serialise
`numpy.bool_` — so instead of raising, the `default=str` hook wrote the **string**
`"True"`.

The frontend (`frontend/app/fairness/page.tsx:26-28`) tests:

```tsx
if (passed === true)  return <span className="pill pill-green">PASS</span>;
if (passed === false) return <span className="pill pill-red">FAIL</span>;
```

A string satisfies neither branch, so every cohort rendered as a neutral dash —
the live fairness page showed **0 PASS, 0 FAIL, 42 neutral**. The gate was
working; its report was unreadable.

**Fix.** `bool(...)` at the point of comparison, plus
`SliceResult.to_json_safe()`, which returns JSON-native scalars for every field
and is what the model card now serialises. The general lesson is worth stating:
`default=str` converts a serialisation *error* into silently wrong data, and is
therefore a liability wherever correctness of the output matters.

---

## 4. The training window filtered on a column no frame in this pipeline has

`compute_training_window` filtered on `batch_date`. The pipeline writes the batch
date into the **filename** (`batch_2015-03.parquet`); the frame's own date column
is `issue_d`. `"batch_date" in df.columns` was therefore always false, the filter
was a no-op, and the auto strategy returned on its **first** iteration with
`n_days = auto_max_days`.

Result: MLflow and every model card recorded a **180-day** training window for a
model trained on the full 2015–2018 history. The number was not an approximation;
it was unrelated to the data.

A second, quieter problem: the cutoff was computed from `utcnow()`. This dataset
ends in 2018, so a wall-clock anchor puts every cutoff years in the future and
would have selected the empty set had the column ever matched.

**Fix.** The window looks for `batch_date` then `issue_d`; it anchors the cutoff
on the **latest date present in the data**; and the day count it returns always
describes the rows it actually returned, computed from them.

---

## 5. The drift trigger fires on every batch (measured, deliberately not retuned)

**Measured: 36 of 36 batches trigger a retrain — 100%.** KS flagged drift in 7 to
11 of the 11 numeric features on *every* batch, against a configured threshold of
`>= 2`. A trigger that fires unconditionally is not a detector; it is a schedule
with extra steps, and the README presents it as the former.

**The first explanation I reached for was wrong, and the sweep disproved it.** I
expected the cause to be KS's sensitivity at large *n* — a p-value at n = 96,000
vs n = 8,000 flags trivial differences as significant. That is real: on batch
`2016-03`, the mildest in the set, KS "detects drift" at effect sizes of 2–3%
(`loan_amount` *D* = 0.0196, p = 6.9e-03; `credit_score` *D* = 0.0253,
p = 1.5e-04), where the CDFs differ by about two percent and PSI flags **zero**
features.

But adding an effect-size floor does not fix the trigger rate, which is what I
assumed before measuring:

| trigger rule | fires | rate |
|---|---|---|
| **As shipped:** KS-significant count ≥ 2 | 36/36 | **100.0%** |
| KS-significant **and** *D* ≥ 0.05, count ≥ 2 | 36/36 | **100.0%** |
| PSI critical in ≥ 1 feature | 26/36 | 72.2% |
| KS-significant **and** *D* ≥ 0.10, count ≥ 2 | 23/36 | 63.9% |

Reproduce: `python scripts/trigger_sweep.py`.

**The real cause is the reference set, not the test.** The drift magnitude rises
monotonically across the whole series:

| batch | max KS *D* | max PSI | PSI-critical features |
|---|---|---|---|
| 2016-01 | 0.115 | 0.126 | 0 |
| 2017-01 | 0.187 | 0.209 | 1 |
| 2018-01 | 0.228 | 0.305 | 2 |
| 2018-12 | 0.299 | 0.495 | 2 |

The reference frame is the **earliest 12 months (2015-01…2015-12, 96,000 rows)**
and it never moves. So by 2018 the pipeline is asking "does this month differ from
2015?", and the answer is yes, increasingly, forever. **On this dataset the
trigger is not mis-tuned — it is measuring the wrong comparison.** Every batch
really does differ from a frozen 2015 baseline, and no threshold on that
comparison can distinguish "the world moved since 2015" (always true, and already
responded to) from "the world moved since the model currently in production was
trained" (the question a retraining trigger exists to answer).

A drift trigger for *retraining* has to compare against the **current champion's
training distribution**, which advances every time a model is promoted. Measured
against a fixed 2015 reference, drift is being computed relative to a baseline
nobody is serving.

**Why this is reported rather than quietly retuned.** The fix is not a threshold
change — it is a change to what the reference set *is*, which alters the
pipeline's architecture and its retraining cost profile. That belongs to the
repo's owner, not to its evaluator, and picking a number that happens to produce a
pleasing firing rate would be the result-shopping this exercise exists to remove.
What is not acceptable is the status quo plus a README calling it drift-triggered
retraining, so the README now states the measured 100% and why.

I also no longer claim, as an earlier draft of this section did, that "PSI
discriminates and KS does not". PSI discriminates *better* — it is the only signal
here that tracks magnitude, and it stays at 0–1 critical features through 2016
before climbing — but at 72.2% it is not a usable trigger against this reference
either.

---

## 6. The documented install does not resolve

`pip install -r requirements.txt -r requirements-dev.txt` fails with
`ResolutionImpossible`. Cause: `google-genai==1.46.0` (in `requirements.txt`)
requires `httpx>=0.28.1`, while `requirements-dev.txt` pinned `httpx==0.27.0` for
the FastAPI `TestClient`. The two files were jointly unsatisfiable, so no one
following the README could install the project.

A second, independent blocker: `setuptools>=81` removed the bundled
`pkg_resources` shim, which `mlflow` 2.14.x still imports at module scope. On a
fresh venv this makes **14 test modules fail to collect** with
`No module named 'pkg_resources'`.

**Fix.** `httpx==0.28.1` (verified: all 31 `TestClient`-backed tests pass on it)
and an explicit `setuptools<81`, each with the reason recorded inline.
`pip install --dry-run -r requirements.txt -r requirements-dev.txt` now resolves.

**After both fixes the suite runs clean in a fresh pinned venv:** the repo's
pre-existing tests go from *uninstallable* to **110 passed**, including
`tests/test_full_flow_batch_selection.py`, which previously could not be collected
at all. With the 17 tests added by this branch the total is **127 passed**, and
`ruff check .` reports no findings.

---

## 7. The documented run command crashes

```
$ python pipelines/flows.py --flow full
ModuleNotFoundError: No module named 'alerting'
```

Running a file puts its directory on `sys.path`, not the repo root, so every
absolute import in `flows.py` fails. The module form works:
`python -m pipelines.flows --flow full`.

The CI workflow already used `-m` and carried a comment explaining why — so the
correct invocation was known in the repo and simply never reached the README, the
module docstring (which claimed the broken form "just works"), or
`docker-compose.yml`. All now use `-m`.

One correction to an earlier draft of this finding: `docker-compose.yml` was **not
broken**, because both Docker images set `PYTHONPATH=/app`, which makes the path
form work inside the container. It was changed for consistency, not to fix a bug,
and this file says so rather than inflating the count.

---

## 8. `import lightgbm` fails in the root Docker image

`serving/Dockerfile` installs `libgomp1` with a comment explaining that LightGBM's
native library needs the GNU OpenMP runtime, absent from `python:*-slim`. The
**root** `Dockerfile` — the one `docker-compose.yml` builds for the pipeline
service — did not. Every training and validation entrypoint in that image failed
at import with `libgomp.so.1: cannot open shared object file`.

**Fix.** `libgomp1` added to the root image, with the same reasoning recorded.

---

## 9. The README described slice cohorts that do not exist

The README claimed "4 cohort dimensions (16 slices total)" and listed:

- Credit grade: **A / B / C / D / E** — the config defines **A–G** (7)
- Loan purpose: **home / car / personal / business / education** — none of which
  appear in the config, which uses `debt_consolidation`, `credit_card`,
  `home_improvement`, `major_purchase`, `medical`, `small_business`, `car`,
  `other`
- **Age group: young / middle / senior / elderly** — there is **no age
  dimension**; the fourth is `loan_term` (36 / 60 months)

Actual: 4 dimensions, **21 cohorts**. Only the income bracket was described
correctly.

The age claim is the one worth dwelling on. Lending Club does not publish borrower
age, so the dimension could not have existed — and a credit model documented as
slicing on age invites a fair-lending question this project does not attempt to
answer. It was plausible-sounding text that no one had checked against the config.

**Fix.** The section now matches the config, and
`tests/test_readme_slice_claims.py` pins it: the dimension count, the cohort
count, every configured cohort value appearing in the prose, the absence of an
age-like slice column, and the specific string `"16 slices total"` as a
regression guard. All 4 tests go red against the old README.

---

## 10. How the tests were verified (and the one that failed verification)

A regression test that passes whether or not the fix is present is worse than no
test, because it certifies the defect. So each was checked by **reverting the
source fix in place and re-running** — never by reasoning about what it would do.

```
# with training/trainer.py, validation/validator.py, configs/* reverted:
12 failed   (of 13 in tests/test_promotion_gate_defects.py)
# with README.md reverted:
4 failed    (of 4 in tests/test_readme_slice_claims.py)
```

`test_split_temporal_still_uses_issue_d` **passed with the fix reverted**. It is
not a reproduction — it asserts something already true, to keep it true, because
defect 4 was a column-name mismatch and the fix is only as durable as the name it
matches on. It is labelled as a guard in its own docstring rather than quietly
counted among the reproductions.

**17 tests added: 12 reproductions of defects 1–4, 4 documentation guards (all of
which discriminate), and 1 non-discriminating guard disclosed as such.** Repo
total 110 → 127 passed, `ruff check .` clean.

---

## 11. What remains open

- **No post-fix end-to-end run.** §1 explains why no corrected promotion decision
  is reported. Re-baselining needs a full flow run against MLflow and Prefect,
  which was not completed here.
- **The drift threshold is a decision, not a bug fix** (§5). Until it is taken,
  the pipeline retrains unconditionally.
- **The live dashboard still shows the old strings** for any model card generated
  before these fixes. The card is a stored artifact; correcting the generator does
  not rewrite history. A fresh retrain run regenerates it.
- **`prediction_psi` was `None`** in every batch of my sweep, because I passed no
  prediction scores. Whether the deployed flow populates it is unverified here.
