"""
test_ptbxl_biomarker_detection.py

End-to-end test/visualization script for the PTB-XL clinical biomarker
extraction pipeline (`ptbxl_reference.py`): loads a real PTB-XL record,
runs the full procedure (multi-beat detection -> first-beat selection ->
isoelectric baseline correction -> Q/R/S biomarker extraction), and opens
two interactive matplotlib windows:

  Figure 1 — Context: the full 10 s, 12-lead recording in the background,
  with the selected single beat highlighted (shaded window) on every lead,
  so you can check the beat-selection step against the real multi-beat
  signal.

  Figure 2 — Detail: the isolated, baseline-corrected single beat only, on
  a 12-lead clinical grid, with the detected Q/R/S peaks marked on each
  lead (colored markers) and the extracted biomarker values printed to the
  console.

Usage
-----
Edit the RECORD_PATH / MANUAL_WINDOW / ANN_PATH variables in the
"CONFIGURATION" block near the bottom of this file, then run it directly
(VS Code: "Run Python File", or `python test_ptbxl_biomarker_detection.py`
from a terminal with the project's venv active).

Automatic multi-beat detection (default): leave `MANUAL_WINDOW = None`.

Manual window (when automatic detection picks the wrong window — e.g.
captures P or T instead of QRS, since PTB-XL ships no wave-delineation
annotation file and automatic detection has no P/QRS/T selectivity, see
project chat): set `MANUAL_WINDOW = (t_start_ms, t_end_ms)`, e.g.
`MANUAL_WINDOW = (260.0, 400.0)`.

Manual-window mode mirrors `ptbxl_reference.load_ptbxl_reference_manual_window`:
a deliberately wide window is narrowed down via Q/R/S detection (onset =
Q-onset else R-onset; offset = S-offset else R-offset; earliest/latest
across leads, excluding aVR/III — see `_BOUNDS_EXCLUDED_LEADS` in
`ptbxl_reference.py`). The window should still be tight enough to visibly
exclude the P and T waves (a first run with `MANUAL_WINDOW = None`, reading
the approximate QRS location off Figure 1, is the recommended way to pick
it) — `_identify_qrs_waves` defines R as the first positive local maximum
in the supplied window, so an overly wide window risks detecting P instead
of the true R.

`RECORD_PATH` (and `ANN_PATH` if set) are paths WITHOUT extension, exactly
as expected by `wfdb.rdrecord` / `wfdb.rdann`. `ANN_PATH` is ignored when
`MANUAL_WINDOW` is set.

This file can also be imported and driven programmatically, e.g. from
another script or a Jupyter cell::

    from test_ptbxl_biomarker_detection import run_test
    run_test("path/to/00001_hr", manual_window=(260.0, 400.0))

Notes
-----
This script reuses the internal functions of `ptbxl_reference.py`
directly (not just the public `load_ptbxl_reference` /
`load_ptbxl_reference_manual_window` entry points) so it can show the
intermediate state at each step (full signal, detected beats or first-pass
wave detection, selected/narrowed beat, corrected signal) rather than only
the final `ReferenceMetrics` result.

This script has not been run against a real downloaded PTB-XL file (no
network access in the environment it was written in) — only logic- and
syntax-checked with a synthetic signal of the same shape. Run it on one of
your downloaded records and inspect both figures before trusting the
detected biomarkers on real data.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import ptbxl_reference as ptbxl
from ecg_metrics import LEAD_INDEX

# Clinical 12-lead grid layout: 4 rows x 3 columns, standard reading order.
# Note: `_rename_ptbxl_leads` (called before any plotting here) already
# renames PTB-XL's raw AVR/AVL/AVF to the LEAD_INDEX convention aVR/aVL/aVF
# — so `leads_full` and `metrics.qrs_waves` use the SAME lead-name
# convention, and a single grid order works for both figures.
_GRID_ORDER = [
    "I", "aVR", "V1", "V4",
    "II", "aVL", "V2", "V5",
    "III", "aVF", "V3", "V6",
]

# Marker style per wave, used in Figure 2.
_WAVE_MARKER_STYLE = {
    "Q": {"marker": "v", "color": "tab:blue", "label": "Q"},
    "R": {"marker": "^", "color": "tab:red", "label": "R"},
    "S": {"marker": "v", "color": "tab:green", "label": "S"},
}


def _plot_figure1_context(
    leads_full: dict, times_full: np.ndarray, t_qrs_start: float, t_qrs_end: float,
    record_path: str,
):
    """
    Figure 1: full 10 s recording per lead, with the selected beat window
    shaded, on a 12-lead clinical grid.
    """
    crop_start = t_qrs_start - ptbxl._BASELINE_WINDOW_START_MS
    crop_end = t_qrs_end + ptbxl._POST_QRS_MARGIN_MS

    fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Figure 1 — Full recording with selected beat highlighted\n{record_path}",
        fontsize=12,
    )

    for ax, lead_name in zip(axes.flat, _GRID_ORDER):
        ax.plot(times_full, leads_full[lead_name], color="black", linewidth=0.6)
        ax.axvspan(crop_start, crop_end, color="orange", alpha=0.25)
        ax.axvline(t_qrs_start, color="tab:red", linewidth=0.8, linestyle="--")
        ax.axvline(t_qrs_end, color="tab:red", linewidth=0.8, linestyle="--")
        ax.set_title(lead_name, fontsize=10, loc="left")
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Amplitude (mV)")

    fig.tight_layout(rect=[0, 0, 1, 0.94])


def _plot_figure2_detail(metrics, record_path: str):
    """
    Figure 2: isolated, baseline-corrected single beat per lead, on a
    12-lead clinical grid, with detected Q/R/S peaks marked.

    Parameters
    ----------
    metrics : EcgMetrics
        Instance after `compute_qrs_waves()` has been called (so
        `metrics.qrs_waves` and `metrics.qrs_peak_amplitudes` are
        populated).
    """
    fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"Figure 2 — Isolated, baseline-corrected beat with detected Q/R/S\n{record_path}",
        fontsize=12,
    )

    legend_handles = {}

    for ax, lead_name in zip(axes.flat, _GRID_ORDER):
        signal = metrics._ecg_12lead[LEAD_INDEX[lead_name]]
        times = metrics._times
        ax.plot(times, signal, color="black", linewidth=0.9)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.6)

        waves = metrics.qrs_waves[lead_name]
        norm = metrics.qrs_peak_amplitudes[lead_name]
        denorm = norm if not np.isnan(norm) else 1.0

        for wave_name in ("Q", "R", "S"):
            wave = getattr(waves, wave_name)
            if wave.n_peaks == 0:
                continue
            style = _WAVE_MARKER_STYLE[wave_name]
            peak_times = wave.peak_times
            peak_values_mv = wave.peak_values * denorm
            handle = ax.scatter(
                peak_times, peak_values_mv,
                marker=style["marker"], color=style["color"],
                s=60, zorder=5, label=style["label"],
            )
            legend_handles[wave_name] = handle

        ax.set_title(lead_name, fontsize=10, loc="left")
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Amplitude (mV)")

    if legend_handles:
        fig.legend(
            legend_handles.values(), legend_handles.keys(),
            loc="upper right", ncol=3, fontsize=10,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.92])


def _print_biomarker_summary(metrics, reference):
    """Print the extracted ReferenceMetrics biomarkers to the console."""
    print("\n=== Extracted biomarkers (this record) ===")
    print(f"QRS duration: {reference.qrs_duration:.1f} ms")
    print(f"Electrical axis: {reference.electrical_axis}")
    print()
    print("Per-lead Q duration (ms):")
    for lead, val in reference.q_duration.items():
        print(f"  {lead:5s}: {val}")
    print()
    print("Per-lead Q/R ratio:")
    for lead, val in reference.q_over_r.items():
        print(f"  {lead:5s}: {val}")
    print()
    print("Per-lead onset-to-peak (ms):")
    for lead, val in reference.onset_to_peak.items():
        print(f"  {lead:5s}: {val}")
    print()
    print(f"R-progression monotony penalty: {reference.r_progression_monotony}")
    print()
    print("Per-lead R/|S| ratio:")
    for lead, val in reference.r_over_s.items():
        print(f"  {lead:5s}: {val}")
    print()
    print(f"Q amplitude in V1: {reference.q_amplitude_v1}")
    print()
    print("Notch depth (deepest notch per wave/lead, non-zero only):")
    for wave_name, lead_dict in reference.notch_depth.items():
        for lead, val in lead_dict.items():
            if val > 0:
                interval = reference.notch_interval[wave_name][lead]
                print(f"  {wave_name}/{lead:5s}: depth={val:.3f}, interval={interval:.1f} ms")
    print()
    print("Per-lead late-extrema signed sum (non-zero only):")
    for lead, val in reference.extra_extrema.items():
        if val != 0.0:
            print(f"  {lead:5s}: {val:+.3f}")


def run_test(record_path: str, manual_window: tuple = None, ann_path: str = None):
    """
    Run the full PTB-XL biomarker detection pipeline on one record and open
    the two diagnostic figures.

    Parameters
    ----------
    record_path : str
        Path to the record WITHOUT extension (e.g.
        ``"path/to/records500/00000/00001_hr"``), as expected by
        `wfdb.rdrecord`.
    manual_window : tuple(float, float), optional
        `(t_window_start_ms, t_window_end_ms)`. If supplied, bypasses
        automatic multi-beat detection and instead narrows this
        deliberately wide window down to the real QRS bounds — use this
        when automatic detection picks up the wrong window (e.g. captures
        the P or T wave instead of QRS, since PTB-XL ships no
        wave-delineation annotation file and automatic detection has no
        morphological selectivity between P/QRS/T — see project chat).
        Read the window off Figure 1 from a first run with
        `manual_window=None`, tightened to visibly exclude P and T (a few
        tens of ms of margin around the true QRS is enough — too wide a
        window can cause `_identify_qrs_waves` to mis-detect P as R, since
        R is defined as the first positive local maximum in the supplied
        window). Default is `None` (automatic detection).
    ann_path : str, optional
        Path (without extension) to a wfdb annotation file, used only in
        automatic-detection mode (ignored if `manual_window` is supplied).
        Default is `None`.
    """
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError(
            "The 'wfdb' package is required. Install it with: pip install wfdb"
        ) from exc

    record = wfdb.rdrecord(record_path)
    fs = float(record.fs)
    n_samples = record.p_signal.shape[0]
    times_full = np.arange(n_samples) / fs * 1000.0

    leads_full = ptbxl._rename_ptbxl_leads(record.p_signal)

    if manual_window is not None:
        # --- Manual-window mode: mirrors load_ptbxl_reference_manual_window
        # step by step, keeping every intermediate result for plotting. ---
        t_window_start, t_window_end = manual_window
        window_mask = (times_full >= t_window_start) & (times_full <= t_window_end)
        if not np.any(window_mask):
            raise RuntimeError(
                f"manual_window ({t_window_start}, {t_window_end}) ms "
                f"contains no samples — check it falls within the record's "
                f"duration (0 to {times_full[-1]:.1f} ms)."
            )

        times_window = times_full[window_mask]
        leads_window = {name: sig[window_mask] for name, sig in leads_full.items()}

        # Pass 1: detect waves on the raw, wide, user-supplied window.
        metrics_pass1 = ptbxl._make_clinical_ecg_metrics(
            leads_window, times_window, t_window_start, t_window_end
        )
        metrics_pass1.compute_qrs_waves()

        t_qrs_start, t_qrs_end = ptbxl._compute_real_qrs_bounds_from_first_pass(metrics_pass1)
        print(
            f"Manual window {manual_window} narrowed to real "
            f"QRS bounds [{t_qrs_start:.1f}, {t_qrs_end:.1f}] ms."
        )
    else:
        # --- Automatic mode ---
        all_windows = []
        if ann_path is not None:
            try:
                ann = wfdb.rdann(ann_path, extension="atr")
                all_windows = ptbxl._detect_all_qrs_windows_from_annotations(ann, fs)
            except Exception as exc:  # noqa: BLE001
                print(f"Could not read annotations ({exc}); falling back to threshold detection.")
                all_windows = []

        if not all_windows:
            all_windows = ptbxl._detect_all_qrs_windows_from_threshold(leads_full, times_full)

        print(f"Detected {len(all_windows)} beat(s) in the full recording.")

        t_qrs_start, t_qrs_end = ptbxl._select_first_beat_window(all_windows, times_full)
        print(f"Selected first complete beat: [{t_qrs_start:.1f}, {t_qrs_end:.1f}] ms")

    # --- From here on, both modes converge: baseline correction + pass 2,
    # exactly as in load_ptbxl_reference / load_ptbxl_reference_manual_window. ---
    leads_corrected = ptbxl._correct_isoelectric_baseline(leads_full, times_full, t_qrs_start)

    crop_start = t_qrs_start - ptbxl._BASELINE_WINDOW_START_MS
    crop_end = t_qrs_end + ptbxl._POST_QRS_MARGIN_MS
    crop_mask = (times_full >= crop_start) & (times_full <= crop_end)
    times_cropped = times_full[crop_mask]
    leads_cropped = {name: sig[crop_mask] for name, sig in leads_corrected.items()}

    metrics = ptbxl._make_clinical_ecg_metrics(
        leads_cropped, times_cropped, t_qrs_start, t_qrs_end
    )
    reference = ptbxl._extract_reference_biomarkers(metrics, source=record_path)

    _print_biomarker_summary(metrics, reference)

    _plot_figure1_context(leads_full, times_full, t_qrs_start, t_qrs_end, record_path)
    _plot_figure2_detail(metrics, record_path)

    plt.show()


# =============================================================================
# CONFIGURATION — edit the parameters below, then run this file directly
# (VS Code: "Run Python File", or `python test_ptbxl_biomarker_detection.py`).
# =============================================================================

# Path to the PTB-XL record, WITHOUT extension (wfdb looks for
# "<RECORD_PATH>.dat" and "<RECORD_PATH>.hea" next to each other).
RECORD_PATH = r"C:\Users\Raphael\pyansys-heart_update_1\examples\postprocessor\ptbxl_signals\records500\05000\05552_hr"

# Set to None for automatic multi-beat detection, or to (t_start_ms,
# t_end_ms) to use a manually-supplied window instead — see `run_test`'s
# docstring above for how to pick this window (read it off Figure 1 from a
# first run with MANUAL_WINDOW = None).
# Example: MANUAL_WINDOW = (260.0, 400.0)
MANUAL_WINDOW = None

# Optional path (without extension) to a wfdb annotation file. Only used
# in automatic-detection mode (ignored if MANUAL_WINDOW is set). PTB-XL's
# base dataset ships no such file for any record, so this is rarely
# useful in practice — see project chat.
ANN_PATH = None


if __name__ == "__main__":
    run_test(RECORD_PATH, manual_window=MANUAL_WINDOW, ann_path=ANN_PATH)

