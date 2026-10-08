"""Pin the README's slice-gate description to the actual config.

The README claimed "4 cohort dimensions (16 slices total)" and listed an **age
group** dimension (young / middle / senior / elderly). The config has 4
dimensions and **21** cohorts, and none of them is age -- the fourth is loan
term. The credit-grade range (A-E vs A-G) and the entire loan-purpose list were
wrong too.

Documentation drift is the cheapest kind of false claim to make and one of the
more expensive to be caught making, so this is a test rather than a comment.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
CONFIG = REPO_ROOT / "configs" / "config.yaml"


def _slices() -> dict:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return cfg["dataset"]["validation_slices"]


def _cohorts(slice_def: dict) -> list:
    """The cohort values of one dimension, however the config spells them."""
    values = slice_def.get("labels")
    if values is None:
        values = slice_def.get("values")
    assert values is not None, f"slice has neither labels nor values: {slice_def}"
    return list(values)


def test_readme_states_the_real_dimension_and_cohort_counts():
    slices = _slices()
    n_dimensions = len(slices)
    n_cohorts = sum(len(_cohorts(d)) for d in slices.values())

    text = README.read_text(encoding="utf-8")

    assert f"{n_dimensions} cohort dimensions" in text, (
        f"README should say '{n_dimensions} cohort dimensions'"
    )
    assert f"**{n_cohorts}** cohorts" in text or f"{n_cohorts} cohorts" in text, (
        f"README should state {n_cohorts} cohorts; the config defines that many"
    )

    # The specific wrong number that shipped. Guard it by value, so a future
    # edit that reintroduces it fails here rather than on someone else's screen.
    assert "16 slices total" not in text, (
        "README is back to claiming 16 slices; the config defines "
        f"{n_cohorts} cohorts"
    )


def test_readme_does_not_claim_a_cohort_dimension_that_does_not_exist():
    """Specifically: there is no age dimension, and claiming one on a credit
    model invites a fair-lending question the project does not answer."""
    columns = {d["column"] for d in _slices().values()}
    assert not any("age" in c.lower() for c in columns), (
        "config now has an age-like slice column; update the README text and "
        "this test together, and consider the fair-lending implications"
    )

    text = README.read_text(encoding="utf-8")
    assert "Age group:" not in text
    assert "purpose, age)" not in text


def test_every_configured_dimension_is_named_in_the_readme():
    """Each dimension's config key or column must appear, so a newly added
    slice cannot stay undocumented."""
    text = README.read_text(encoding="utf-8").lower()
    missing = []
    for name, definition in _slices().items():
        pretty = name.replace("_", " ")
        if pretty not in text and name not in text and definition["column"] not in text:
            missing.append(name)
    assert not missing, f"slice dimensions absent from the README: {missing}"


def test_readme_cohort_lists_match_the_config_values():
    """The listed cohort values must be the configured ones.

    The old text listed loan purposes (home / personal / business / education)
    that do not occur anywhere in the config -- they read plausibly, which is
    exactly what made them hard to notice.
    """
    text = README.read_text(encoding="utf-8")
    # Pull the bullet list under the slice-validation heading.
    section = re.search(
        r"\*\*Gate 2 — Slice Validation\*\*(.+?)If ANY", text, re.S
    )
    assert section, "could not locate the Gate 2 section in the README"
    body = section.group(1)

    for name, definition in _slices().items():
        for cohort in _cohorts(definition):
            assert str(cohort) in body, (
                f"cohort {cohort!r} of dimension {name!r} is configured but not "
                "listed in the README's Gate 2 section"
            )
