"""
plot_implausibility_overlay.py

Combines `compute_implausibility_score.py` (real PyAnsys Heart simulation
vs. a hand-edited clinical reference) and `test_ptbxl_biomarker_detection.py`
(PTB-XL signal processing pipeline) into a single diagnostic view:

1. Loads the simulated ECG (`EcgMetrics`)
2. Loads and processes a real clinical PTB-XL ECG (multi-beat detection or
   manual window, isoelectric baseline correction — same pipeline as
   `test_ptbxl_biomarker_detection.py`), producing a second `EcgMetrics`
   instance for the single isolated clinical beat.
3. Computes the implausibility score of the simulated ECG against the
   **hand-edited** `ReferenceMetrics` (not directly against the clinical
   signal's own derived biomarkers — the score uses whatever you typed into
   the REFERENCE block below, exactly as in `compute_implausibility_score.py`).
4. Plots both 12-lead ECGs overlaid, time-aligned on each signal's own R
   peak in lead I, restricted to the time range where both signals have
   data (intersection of their respective windows around that shared
   t=0), amplitude-normalized per lead by each signal's own
   `qrs_peak_amplitudes` (so the biggest deflection is at +-1 on both signals).
5. Annotates each subplot with the relevant penalty-vector contributions:
   global criteria (qrs_duration, electrical_axis, axis_ambiguous) are
   spread onto the frontal leads (I, II, III, aVR, aVL, aVF);
   r_progression_monotony is spread onto V1-V4; everything else
   (q_duration, q_over_r, onset_to_peak, r_over_s, q_amplitude_v1, notches)
   is shown only on its own specific lead, both as a text box and as a
   highlighted time span over the relevant wave when that span is known.

Usage
-----
Edit the three CONFIGURATION blocks near the bottom of this file (
SIMULATION CONFIGURATION, CLINICAL CONFIGURATION, REFERENCE CONFIGURATION),
then run this file directly (VS Code: "Run Python File", or
`python plot_implausibility_overlay.py`).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ansys.health.heart.examples import get_preprocessed_fullheart
from ansys.health.heart.post.dpf_utils import EPpostprocessor
import ansys.health.heart.models as models
import os
from pathlib import Path

from ecg_metrics import (
    EcgMetrics,
    ReferenceMetrics,
    LEAD_INDEX,
    _Q_WAVE_LEADS,
    _ONSET_TO_PEAK_LEADS,
    _R_OVER_S_LEADS,
    _R_PROGRESSION_LEADS,
    _NOTCH_LEADS,
    _HEXAXIAL,
)
import ptbxl_reference as ptbxl

os.environ["ANSYS_DPF_ACCEPT_LA"] = "Y"

# Same 3 rows x 4 columns layout as compute_12_lead_ECGs_interactive in
# ecg_metrics_usage_example.py (limb leads in column 0-1, precordial leads
# in columns 2-3), NOT the 4x3 clinical grid used in
# test_ptbxl_biomarker_detection.py — project decision for this script.
_GRID_LAYOUT = [
    ("I", 0, 0), ("II", 1, 0), ("III", 2, 0),
    ("aVR", 0, 1), ("aVL", 1, 1), ("aVF", 2, 1),
    ("V1", 0, 2), ("V2", 1, 2), ("V3", 2, 2),
    ("V4", 0, 3), ("V5", 1, 3), ("V6", 2, 3),
]

_FRONTAL_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF")

_SIM_COLOR = "tab:blue"
_CLIN_COLOR = "black"

# Minimum |contribution| (weight * penalty) for a criterion to be shown in
# a subplot's annotation box / highlighted as a time span — most notch_depth
# / notch_interval criteria are legitimately 0.0 for a clean reference, and
# listing all of them per lead would make the annotation boxes unreadable.
# Set to 0.0 to show everything.
_MIN_CONTRIBUTION_TO_SHOW = 1e-6

# ---------------------------------------------------------------------------
# Criterion color palette — one fixed color per family of criteria.
# The same color is used for both the text label in each subplot's annotation
# box and the vertical boundary lines marking the relevant time span.
# Families group criteria that characterise the same morphological feature.
# ---------------------------------------------------------------------------
_CRITERION_COLORS = {
    "qrs_duration":         "tab:purple",
    "electrical_axis":      "tab:orange",
    "axis_ambiguous":       "tab:orange",
    "r_progression_monotony": "tab:brown",
    "q_duration":           "tab:blue",
    "q_over_r":             "tab:cyan",
    "onset_to_peak":        "tab:green",
    "r_over_s":             "tab:olive",
    "q_amplitude_v1":       "tab:pink",
    "notch_depth":          "tab:red",
    "notch_interval":       "crimson",
    "extra_extrema":        "tab:gray",
}

# Vertical boundary line style for time-span highlighting.
# Two thin axvlines (one at span start, one at span end)
_SPAN_LINEWIDTH = 1.5
_SPAN_LINESTYLE = "--"
_SPAN_ALPHA = 0.85


def _criterion_color(key: str) -> str:
    """
    Return the display color for a penalty-vector key by matching its family
    prefix against ``_CRITERION_COLORS``.

    Falls back to ``"tab:gray"`` for unrecognised keys rather than raising,
    so new criteria added to ``ecg_metrics.py`` degrade gracefully to a
    neutral color instead of crashing the plot.
    """
    for prefix, color in _CRITERION_COLORS.items():
        if key == prefix or key.startswith(prefix + "_"):
            return color
    return "tab:gray"


def _closest_frontal_lead_to_axis(axis_deg: float) -> str:
    """
    Return the name of the frontal lead whose hexaxial direction is closest
    (minimum circular angular distance) to ``axis_deg``.

    Uses ``_HEXAXIAL`` from ``ecg_metrics`` directly — O(6) comparison, no
    iterative computation. If ``axis_deg`` is NaN (ambiguous axis), falls
    back to ``"I"`` as a safe default so the annotation still appears
    somewhere rather than disappearing silently.

    Parameters
    ----------
    axis_deg : float
        Mean QRS electrical axis in degrees, as returned by
        ``EcgMetrics.electrical_axis``.
    """
    if np.isnan(axis_deg):
        return "I"

    def _circular_dist(a: float, b: float) -> float:
        diff = abs(a - b) % 360
        return diff if diff <= 180 else 360 - diff

    return min(
        _HEXAXIAL.keys(),
        key=lambda lead: _circular_dist(float(_HEXAXIAL[lead]["angle"]), axis_deg),
    )



# =============================================================================
# Step 1 — load the simulated ECG (real PyAnsys model)
# =============================================================================

def _load_simulated_metrics(workdir: Path, simulation_folder_name: str,
                             prominence_fraction: float = None) -> EcgMetrics:
    """
    Load a real PyAnsys Heart simulation result and return a ready-to-use
    `EcgMetrics` instance (`compute_qrs_waves()` already called).
    """
    path_to_model, path_to_partinfo, _ = get_preprocessed_fullheart(resolution="2.0mm")

    data_path = workdir / simulation_folder_name / "main_ep_reaction_eikonal" / "d3plot"
    if not data_path.is_file():
        raise FileNotFoundError(f"File not found: {data_path}")

    post = EPpostprocessor(data_path)

    model: models.FullHeart = models.HeartModel.load_model(
        path_to_model, path_to_partinfo, working_directory=workdir
    )

    path_to_ecg = workdir / simulation_folder_name / "main_ep_reaction_eikonal" / "em_EKG_001.dat"
    ecgs, times_ecg = post.read_ECGs(path_to_ecg)
    ecg_12lead = post.compute_12_lead_ECGs(ecgs, times_ecg, plot=False)

    metrics = EcgMetrics(model=model, post=post, ecg_12lead=ecg_12lead, times=times_ecg)

    if prominence_fraction is not None:
        metrics.compute_qrs_waves(prominence_fraction=prominence_fraction)
    else:
        metrics.compute_qrs_waves()

    return metrics


# =============================================================================
# Step 2 — load and process the clinical PTB-XL ECG
# =============================================================================

def _load_clinical_metrics(record_path: str, manual_window: tuple = None,
                            ann_path: str = None) -> EcgMetrics:
    """
    Load a PTB-XL record and run the same pipeline as
    `test_ptbxl_biomarker_detection.run_test` (multi-beat detection or
    manual window, isoelectric baseline correction), returning the final
    `EcgMetrics` instance for the single isolated, corrected beat
    (`compute_qrs_waves()` already called).

    Parameters mirror `test_ptbxl_biomarker_detection.run_test` exactly —
    see that function's docstring for the full explanation of
    `manual_window` / `ann_path`.
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

        metrics_pass1 = ptbxl._make_clinical_ecg_metrics(
            leads_window, times_window, t_window_start, t_window_end
        )
        metrics_pass1.compute_qrs_waves()
        t_qrs_start, t_qrs_end = ptbxl._compute_real_qrs_bounds_from_first_pass(metrics_pass1)
        print(
            f"Clinical manual window {manual_window} narrowed to real "
            f"QRS bounds [{t_qrs_start:.1f}, {t_qrs_end:.1f}] ms."
        )
    else:
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
        print(f"Detected {len(all_windows)} beat(s) in the clinical recording.")

        t_qrs_start, t_qrs_end = ptbxl._select_first_beat_window(all_windows, times_full)
        print(f"Selected first complete clinical beat: [{t_qrs_start:.1f}, {t_qrs_end:.1f}] ms")

    leads_corrected = ptbxl._correct_isoelectric_baseline(leads_full, times_full, t_qrs_start)

    crop_start = t_qrs_start - ptbxl._BASELINE_WINDOW_START_MS
    crop_end = t_qrs_end + ptbxl._POST_QRS_MARGIN_MS
    crop_mask = (times_full >= crop_start) & (times_full <= crop_end)
    times_cropped = times_full[crop_mask]
    leads_cropped = {name: sig[crop_mask] for name, sig in leads_corrected.items()}

    metrics = ptbxl._make_clinical_ecg_metrics(
        leads_cropped, times_cropped, t_qrs_start, t_qrs_end
    )
    metrics.compute_qrs_waves()
    return metrics


# =============================================================================
# Step 3 — time alignment + amplitude normalization helpers
# =============================================================================

def _normalized_signal_aligned_on_lead_i_r_peak(metrics: EcgMetrics) -> tuple:
    """
    Return (times_aligned, normalized_signals) for one `EcgMetrics`
    instance: every lead's signal divided by its own `qrs_peak_amplitudes`
    value, and the shared time axis shifted
    so lead I's detected R peak sits at t=0.

    Returns
    -------
    times_aligned : numpy.ndarray
    normalized_signals : dict[str, numpy.ndarray]

    Raises
    ------
    RuntimeError
        If lead I has no detected R peak (cannot anchor the alignment).
    """
    r_wave_i = metrics.qrs_waves["I"].R
    if r_wave_i.n_peaks == 0:
        raise RuntimeError(
            "Lead I has no detected R peak — cannot use it as the timing "
            "anchor for overlay alignment. Inspect this signal's QRS "
            "detection before using this script."
        )
    t_r_peak_i = float(r_wave_i.time)

    times_aligned = metrics._times - t_r_peak_i

    normalized_signals = {}
    for lead in LEAD_INDEX:
        norm = metrics.qrs_peak_amplitudes[lead]
        denom = norm if (not np.isnan(norm) and norm != 0.0) else 1.0
        normalized_signals[lead] = metrics._ecg_12lead[LEAD_INDEX[lead]] / denom

    return times_aligned, normalized_signals


def _intersect_time_range(times_a: np.ndarray, times_b: np.ndarray) -> tuple:
    """Return (t_min, t_max), the intersection of two time ranges already
    expressed relative to their own shared t=0 anchor."""
    t_min = max(times_a[0], times_b[0])
    t_max = min(times_a[-1], times_b[-1])
    if t_min >= t_max:
        raise RuntimeError(
            f"The two signals' time windows do not overlap after "
            f"R-peak alignment: signal A spans [{times_a[0]:.1f}, "
            f"{times_a[-1]:.1f}] ms, signal B spans [{times_b[0]:.1f}, "
            f"{times_b[-1]:.1f}] ms relative to their own R peak in lead I."
        )
    return t_min, t_max


# =============================================================================
# Step 4 — annotation routing: which lead(s) does each penalty-vector key
# concern, and what time span (if any, in the SIMULATED metrics' own
# original-time coordinates) does it correspond to?
# =============================================================================

def _build_annotation_routing(sim_metrics: EcgMetrics, result) -> dict:
    """
    Build `{lead_name: [ (criterion_key, contribution, span_or_None), ... ]}`
    for every lead that has at least one relevant criterion, where
    `span_or_None` is `(t_start, t_end)` in the same time coordinates as
    `sim_metrics._times` (i.e. NOT yet aligned on the R peak — the caller
    must apply the same shift used for the simulated signal), or `None` if
    no specific time span applies (e.g. global criteria).

    Parameters
    ----------
    sim_metrics : EcgMetrics
        The simulated signal's `EcgMetrics` instance (used to look up wave
        timings for the time-span highlighting).
    result : ImplausibilityResult
        Output of `EcgMetrics.compute_implausibility`.

    Notes
    -----
    Routing rules:
    - `qrs_duration`, `electrical_axis`, `axis_ambiguous`: routed to the
      single frontal lead whose hexaxial direction is closest to the
      simulated electrical axis (minimum circular angular distance in
      ``_HEXAXIAL``), rather than spread onto all 6 frontal leads.
      Falls back to lead ``"I"`` if the axis is NaN (ambiguous).
    - `r_progression_monotony`: spread onto `_R_PROGRESSION_LEADS` (V1-V4).
    - `q_duration_{lead}`, `q_over_r_{lead}`: their own lead in
      `_Q_WAVE_LEADS`, time span = that lead's detected Q wave duration.
    - `onset_to_peak_{lead}`: its own lead in `_ONSET_TO_PEAK_LEADS`, time
      span = [QRS onset, R peak] for that lead.
    - `r_over_s_{lead}`: its own lead in `_R_OVER_S_LEADS`, time span =
      [R peak, S peak] for that lead (covers the R-to-S transition the
      ratio characterises).
    - `q_amplitude_v1`: V1, time span = that lead's Q wave (if present).
    - `notch_depth_{wave}_{lead}` / `notch_interval_{wave}_{lead}`: that
      lead, time span = the deepest notch's two peak times for that wave
      (if at least 2 peaks were detected on that wave/lead).
    - `extra_extrema_{lead}`: its own lead, for every lead in `LEAD_INDEX`,
      time span = [earliest, latest] among all detected late-extrema times
      (maxima and minima combined) on that lead, if any were detected.
    """
    routing: dict = {lead: [] for lead in LEAD_INDEX}
    pv = result.penalty_vector
    weights = result.weights

    def add(lead, key, span=None):
        if key in pv and not np.isnan(pv[key]):
            routing[lead].append((key, weights[key] * pv[key], span))

    # --- Global criteria: route to the single frontal lead whose hexaxial
    # direction is closest to the simulated electrical axis, rather than
    # spreading onto all 6 frontal leads (project decision, see chat).
    # r_progression_monotony is spread onto V1-V4 (its natural substrate). ---
    axis_lead = _closest_frontal_lead_to_axis(sim_metrics.electrical_axis)
    add(axis_lead, "qrs_duration")
    add(axis_lead, "electrical_axis")
    add(axis_lead, "axis_ambiguous")

    for lead in _R_PROGRESSION_LEADS:
        add(lead, "r_progression_monotony")

    for lead in _Q_WAVE_LEADS:
        q_wave = sim_metrics.qrs_waves[lead].Q
        span = None
        if q_wave.n_peaks > 0 and not np.isnan(q_wave.wave_duration):
            span = (float(q_wave.time), float(q_wave.time) + float(q_wave.wave_duration))
        add(lead, f"q_duration_{lead}", span)
        add(lead, f"q_over_r_{lead}", span)

    for lead in _ONSET_TO_PEAK_LEADS:
        r_wave = sim_metrics.qrs_waves[lead].R
        span = None
        if r_wave.n_peaks > 0:
            min_lv = sim_metrics._get_min_activation_time_for_part(sim_metrics._model.left_ventricle)
            min_rv = sim_metrics._get_min_activation_time_for_part(sim_metrics._model.right_ventricle)
            min_septum = sim_metrics._get_min_activation_time_for_part(sim_metrics._model.septum)
            t_qrs_start = min(min_lv, min_rv, min_septum)
            span = (float(t_qrs_start), float(r_wave.time))
        add(lead, f"onset_to_peak_{lead}", span)

    for lead in _R_OVER_S_LEADS:
        waves = sim_metrics.qrs_waves[lead]
        span = None
        if waves.R.n_peaks > 0 and waves.S.n_peaks > 0:
            span = (float(waves.R.time), float(waves.S.time))
        add(lead, f"r_over_s_{lead}", span)

    q_v1 = sim_metrics.qrs_waves["V1"].Q
    span_v1 = None
    if q_v1.n_peaks > 0 and not np.isnan(q_v1.wave_duration):
        span_v1 = (q_v1.time, q_v1.time + q_v1.wave_duration)
    add("V1", "q_amplitude_v1", span_v1)

    for wave_name, leads in _NOTCH_LEADS.items():
        for lead in leads:
            wave = getattr(sim_metrics.qrs_waves[lead], wave_name)
            span = None
            if wave.n_peaks >= 2 and len(wave.notch_depths) > 0:
                idx = int(np.argmax(wave.notch_depths))
                span = (float(wave.peak_times[idx]), float(wave.peak_times[idx + 1]))
            add(lead, f"notch_depth_{wave_name}_{lead}", span)
            add(lead, f"notch_interval_{wave_name}_{lead}", span)

    for lead in LEAD_INDEX:
        extrema = sim_metrics.qrs_waves[lead].extra_extrema
        all_times = np.concatenate(
            [extrema["maxima"]["times"], extrema["minima"]["times"]]
        )
        span = None
        if all_times.size > 0:
            span = (float(np.min(all_times)), float(np.max(all_times)))
        add(lead, f"extra_extrema_{lead}", span)

    return routing


# =============================================================================
# Step 5 — the combined plot
# =============================================================================

def plot_overlay(sim_metrics: EcgMetrics, clin_metrics: EcgMetrics, result):
    """
    Plot the simulated and clinical 12-lead ECGs overlaid, time-aligned on
    each signal's own R peak in lead I, amplitude-normalized per lead by
    each signal's own `qrs_peak_amplitudes`, restricted to the intersection
    of both signals' time ranges around the shared t=0. Annotates each
    subplot with the relevant implausibility penalty-vector contributions.

    Parameters
    ----------
    sim_metrics : EcgMetrics
        Simulated signal, `compute_qrs_waves()` already called.
    clin_metrics : EcgMetrics
        Clinical signal (single isolated, baseline-corrected beat),
        `compute_qrs_waves()` already called.
    result : ImplausibilityResult
        Output of `sim_metrics.compute_implausibility(reference)`.
    """
    t_r_peak_sim = float(sim_metrics.qrs_waves["I"].R.time)

    times_sim_aligned, signals_sim = _normalized_signal_aligned_on_lead_i_r_peak(sim_metrics)
    times_clin_aligned, signals_clin = _normalized_signal_aligned_on_lead_i_r_peak(clin_metrics)

    t_min, t_max = _intersect_time_range(times_sim_aligned, times_clin_aligned)

    routing = _build_annotation_routing(sim_metrics, result)

    fig, axes = plt.subplots(3, 4, figsize=(18, 10), sharex=True)
    fig.suptitle(
        f"Simulated vs clinical ECG overlay — total_score = {result.total_score:.4f}\n"
        f"(aligned on lead I R peak; amplitudes normalized per lead by "
        f"qrs_peak_amplitudes)",
        fontsize=13,
    )

    sim_mask = (times_sim_aligned >= t_min) & (times_sim_aligned <= t_max)
    clin_mask = (times_clin_aligned >= t_min) & (times_clin_aligned <= t_max)

    for lead, row, col in _GRID_LAYOUT:
        ax = axes[row, col]
        ax.plot(
            times_sim_aligned[sim_mask], signals_sim[lead][sim_mask],
            color=_SIM_COLOR, linewidth=1.1, label="Simulated",
        )
        ax.plot(
            times_clin_aligned[clin_mask], signals_clin[lead][clin_mask],
            color=_CLIN_COLOR, linewidth=1.1, label="Clinical", alpha=0.8,
        )
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
        ax.set_xlim(t_min, t_max)
        ax.set_title(lead, fontsize=10, loc="left")
        ax.grid(True, alpha=0.25)

        entries = routing.get(lead, [])
        shown_entries = [e for e in entries if abs(e[1]) >= _MIN_CONTRIBUTION_TO_SHOW]
        hidden_count = len(entries) - len(shown_entries)

        if shown_entries:
            # Draw each criterion label in its own color, stacking them
            # vertically from the top-right corner.  We use individual
            # ax.text() calls so each line can carry its own color, rather
            # than a single monochrome text block.
            line_height = 0.115   # fraction of axes height per line
            y_top = 0.97
            for idx, (key, contribution, _) in enumerate(shown_entries):
                color = _criterion_color(key)
                y = y_top - idx * line_height
                ax.text(
                    0.98, y, f"{key}: {contribution:+.3f}",
                    transform=ax.transAxes, fontsize=6.5, color=color,
                    ha="right", va="top", fontweight="bold",
                )
            if hidden_count:
                y = y_top - len(shown_entries) * line_height
                ax.text(
                    0.98, y, f"(+{hidden_count} near-zero)",
                    transform=ax.transAxes, fontsize=6, color="gray",
                    ha="right", va="top",
                )

            # Draw two thin vertical lines at the span boundaries (start and
            # end) instead of a filled axvspan, to avoid color confusion when
            # multiple criteria share overlapping time ranges.
            drawn_spans: set = set()
            for key, _, span in shown_entries:
                if span is None:
                    continue
                # Force plain Python floats — span values can be DPFArray
                # scalars in a real PyAnsys context, which don't support
                # round() or hash() as-is.
                span_start_aligned = float(span[0]) - t_r_peak_sim
                span_end_aligned = float(span[1]) - t_r_peak_sim
                span_key = (round(span_start_aligned, 3), round(span_end_aligned, 3), key)
                if span_key in drawn_spans:
                    continue
                drawn_spans.add(span_key)
                color = _criterion_color(key)
                ax.axvline(
                    span_start_aligned, color=color,
                    linewidth=_SPAN_LINEWIDTH, linestyle=_SPAN_LINESTYLE,
                    alpha=_SPAN_ALPHA,
                )
                ax.axvline(
                    span_end_aligned, color=color,
                    linewidth=_SPAN_LINEWIDTH, linestyle=_SPAN_LINESTYLE,
                    alpha=_SPAN_ALPHA,
                )

    axes[0, 0].legend(loc="upper left", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time relative to lead I R peak (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized amplitude")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


# =============================================================================
# SIMULATION CONFIGURATION — edit to point at your own simulation results.
# =============================================================================

WORKDIR = Path.home() / "pyansys-heart"/"downloads" / "Rodero2021" / "01" / "FullHeart"
SIMULATION_FOLDER_NAME = "set_LHS_02"
PROMINENCE_FRACTION = 0.1 # None -> EcgMetrics' own default

# =============================================================================
# CLINICAL CONFIGURATION — edit to point at your own PTB-XL record.
# =============================================================================

RECORD_PATH = r"C:\Users\Raphael\pyansys-heart_update_1\examples\postprocessor\ptbxl_signals\records500\05000\05552_hr"
MANUAL_WINDOW = (4610.0, 4710)  # or (t_start_ms, t_end_ms) — see test_ptbxl_biomarker_detection.py
ANN_PATH = None

# =============================================================================
# REFERENCE CONFIGURATION — hand-edited clinical biomarker values to score
# the SIMULATED ecg against (same structure as compute_implausibility_score.py).
# =============================================================================


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
REFERENCE = None
WEIGHTS_OVERRIDE = None


if __name__ == "__main__":
    sim_metrics = _load_simulated_metrics(WORKDIR, SIMULATION_FOLDER_NAME, PROMINENCE_FRACTION)
    clin_metrics = _load_clinical_metrics(RECORD_PATH, MANUAL_WINDOW, ANN_PATH)
    result = sim_metrics.compute_implausibility(REFERENCE, weights=WEIGHTS_OVERRIDE)
    print(f"\ntotal_score = {result.total_score:.4f}")

    plot_overlay(sim_metrics, clin_metrics, result)