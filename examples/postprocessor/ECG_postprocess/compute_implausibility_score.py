"""
compute_implausibility_score.py

Computes the QRS implausibility score (scalar total + per-criterion
contribution vector) comparing a real PyAnsys Heart simulation against a
hand-edited clinical reference (typically adapted from a PTB-XL record via
`ptbxl_reference.py` / `test_ptbxl_biomarker_detection.py`, then copied and
adjusted by hand below).

Usage
-----
1. Edit the SIMULATION CONFIGURATION block below to point at your own
   simulation results (same pattern as `ecg_metrics_usage_example.py`).
2. Edit the REFERENCE CONFIGURATION block below with the biomarker values
   you want to compare against (copy them from a
   `test_ptbxl_biomarker_detection.py` run's printed summary, or type them
   in by hand).
3. Run this file directly (VS Code: "Run Python File", or
   `python compute_implausibility_score.py`).

The script prints the scalar `total_score`, the full per-criterion
`penalty_vector` (raw, unweighted), the `weights` actually used, and any
diagnostic `flags` (wave-presence mismatches, axis ambiguity), sorted by
contribution (`weight * penalty`) so the largest contributors to the score
are immediately visible.
"""

from __future__ import annotations

import numpy as np

from ansys.health.heart.examples import get_preprocessed_fullheart
from ansys.health.heart.post.dpf_utils import EPpostprocessor
import ansys.health.heart.models as models
import os
from pathlib import Path

from ecg_metrics import EcgMetrics, ReferenceMetrics, LEAD_INDEX, _NOTCH_LEADS

os.environ["ANSYS_DPF_ACCEPT_LA"] = "Y"


# =============================================================================
# SIMULATION CONFIGURATION — edit to point at your own simulation results
# (same pattern as ecg_metrics_usage_example.py).
# =============================================================================

WORKDIR = Path.home() / "test_pri_mesh_2mm" / "Rodero2021" / "01" / "FullHeart"
SIMULATION_FOLDER_NAME = "test_example3"

PROMINENCE_FRACTION = None  # None -> use EcgMetrics' own default (_DEFAULT_PROMINENCE_FRACTION)


def _load_simulated_metrics() -> EcgMetrics:
    """
    Load a real PyAnsys Heart simulation result and return a ready-to-use
    `EcgMetrics` instance (`compute_qrs_waves()` already called).
    """
    path_to_model, path_to_partinfo, _ = get_preprocessed_fullheart(resolution="2.0mm")

    data_path = (
        WORKDIR / SIMULATION_FOLDER_NAME / "main_ep_reaction_eikonal" / "d3plot"
    )
    if not data_path.is_file():
        raise FileNotFoundError(f"File not found: {data_path}")

    post = EPpostprocessor(data_path)

    model: models.FullHeart = models.HeartModel.load_model(
        path_to_model, path_to_partinfo, working_directory=WORKDIR
    )

    path_to_ecg = (
        WORKDIR / SIMULATION_FOLDER_NAME / "main_ep_reaction_eikonal" / "em_EKG_001.dat"
    )
    ecgs, times_ecg = post.read_ECGs(path_to_ecg)
    ecg_12lead = post.compute_12_lead_ECGs(ecgs, times_ecg, plot=False)

    metrics = EcgMetrics(model=model, post=post, ecg_12lead=ecg_12lead, times=times_ecg)

    if PROMINENCE_FRACTION is not None:
        metrics.compute_qrs_waves(prominence_fraction=PROMINENCE_FRACTION)
    else:
        metrics.compute_qrs_waves()

    return metrics


# =============================================================================
# REFERENCE CONFIGURATION — edit by hand with the clinical biomarker values
# to compare against (e.g. copied from a test_ptbxl_biomarker_detection.py
# run's printed summary for a chosen PTB-XL record).
# =============================================================================

# Helper: a dict defaulting every lead to a "no notch" (0.0) value, with
# only the leads you actually want to set to something else overridden.
# Saves having to type all 12 leads by hand for a clean reference where
# notches are expected to be absent almost everywhere.
def _notch_dict(wave: str, overrides: dict = None) -> dict:
    d = {lead: 0.0 for lead in _NOTCH_LEADS[wave]}
    if overrides:
        d.update(overrides)
    return d


REFERENCE = ReferenceMetrics(
    # --- QRS duration (ms) ---
    qrs_duration=85.0,

    # --- Per-lead Q-wave duration (ms), leads: I, aVL, V5, V6 ---
    # np.nan means Q is absent on that lead in the reference.
    q_duration={
        "I": np.nan,
        "aVL": np.nan,
        "V5": np.nan,
        "V6": 18.0,
    },

    # --- Per-lead |Q|/R amplitude ratio, leads: I, aVL, V5, V6 ---
    q_over_r={
        "I": 0.0,
        "aVL": 0.0,
        "V5": 0.00,
        "V6": 0.0262,
    },

    # --- Per-lead QRS-onset-to-R-peak duration (ms), leads: V5, V6 ---
    onset_to_peak={
        "V5": 29.0,
        "V6": 34.0,
    },

    # --- R/S monotonicity penalty across V1-V4 (0.0 = normal progression) ---
    r_progression_monotony=0.0,

    # --- Per-lead R/|S| amplitude ratio, leads: V1, V5 ---
    r_over_s={
        "V1": 0.08,
        "V5": 9.78,
    },

    # --- Normalized Q-wave amplitude in V1 (0.0 = Q absent, normal) ---
    q_amplitude_v1=0.0,

    # --- Notch depth / interval per wave (Q, R, S) and lead (all 12 leads
    # each) — defaults to 0.0 (no notch) everywhere; override specific
    # leads here if you want to test a non-trivial notch scenario. ---
    notch_depth={
        "Q": _notch_dict("Q"),
        "R": _notch_dict("R"),
        "S": _notch_dict("S"),
    },
    notch_interval={
        "Q": _notch_dict("Q"),
        "R": _notch_dict("R"),
        "S": _notch_dict("S"),
    },

    # --- Mean QRS electrical axis in the frontal plane (degrees) ---
    electrical_axis=60.0,

    # --- Free-text provenance identifier, for traceability only ---
    source="hand-edited reference, adapted from PTB-XL record 05552_hr, 75 year old healthy patient",

    # --- Per-lead SIGNED sum of late-extrema normalized values (extrema
    # detected after the S-complex, see QrsWaves.extra_extrema) — defaults
    # to 0.0 (no late extrema) on all 12 leads; override specific leads
    # here if you want to test a non-trivial late-extrema scenario. ---
    extra_extrema={lead: 0.0 for lead in LEAD_INDEX},
)

# Optional: override specific weights (defaults to ecg_metrics.DEFAULT_WEIGHTS
# for any key not given here). Leave as None to use all defaults.
WEIGHTS_OVERRIDE = None


# Minimum contribution (weight * penalty) for a criterion to be shown in
# the main table by default — most of the 12-lead notch_depth/notch_interval
# criteria are legitimately 0.0 for a clean reference, and listing all 84
# of them (12 leads x 3 waves x 2 metrics) would bury the criteria that
# actually matter. Set to 0.0 to show every criterion.
_MIN_CONTRIBUTION_TO_SHOW = 1e-6


def _print_implausibility_result(result):
    """Print the scalar score and the contribution vector, sorted by
    contribution (largest first). Near-zero contributions (below
    `_MIN_CONTRIBUTION_TO_SHOW`) are counted but not listed individually,
    to keep the table readable — see that constant to show everything."""
    print(f"\n=== Implausibility score ===")
    print(f"total_score = {result.total_score:.4f}\n")

    contributions = {
        key: result.weights[key] * value
        for key, value in result.penalty_vector.items()
        if not np.isnan(value)
    }
    nan_keys = [key for key, value in result.penalty_vector.items() if np.isnan(value)]

    shown = {k: v for k, v in contributions.items() if v >= _MIN_CONTRIBUTION_TO_SHOW}
    hidden_count = len(contributions) - len(shown)

    print(f"{'criterion':35s} {'penalty':>10s} {'weight':>10s} {'contribution':>14s}")
    print("-" * 71)
    for key, contribution in sorted(shown.items(), key=lambda kv: -kv[1]):
        penalty = result.penalty_vector[key]
        weight = result.weights[key]
        print(f"{key:35s} {penalty:>10.4f} {weight:>10.4f} {contribution:>14.4f}")

    if hidden_count:
        print(
            f"\n({hidden_count} criteria with contribution < "
            f"{_MIN_CONTRIBUTION_TO_SHOW} not listed. Set"
            f"_MIN_CONTRIBUTION_TO_SHOW = 0.0 to show everything.)"
        )

    if nan_keys:
        print(f"\nExcluded from total_score (NaN on both sides): {nan_keys}")

    if result.flags:
        print("\n=== Flags ===")
        for key, message in result.flags.items():
            print(f"  {key}: {message}")


if __name__ == "__main__":
    metrics = _load_simulated_metrics()
    result = metrics.compute_implausibility(REFERENCE, weights=WEIGHTS_OVERRIDE)
    _print_implausibility_result(result)
