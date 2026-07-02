"""
ptbxl_reference.py

Loads a clinical 12-lead ECG record from PTB-XL (physionet.org/content/ptb-xl)
and extracts the QRS biomarker catalogue defined for the implausibility score,
producing a `ReferenceMetrics` instance independent of any PyAnsys model/post
object.

Design choice — algorithmic identity with the simulated pipeline
------------------------------------------------------------------
This module drives `EcgMetrics` itself on the clinical signal, using
a minimal stand-in object in place of the real PyAnsys `HeartModel` /
`EPpostprocessor`. The only information `_identify_qrs_waves` needs from
`model`/`activation_times` is two scalars: the QRS window start
(`t_min_ventricles`) and end bound (`t_max_ventricles`, only used for the
S-wave duration formula). `_ClinicalQrsWindowModel` below encodes exactly
that, nothing else — see `_make_clinical_ecg_metrics`.

This guarantees that any future change to the simulated-side wave detection
(thresholds, notch logic, onset detection) automatically and identically
affects the clinical reference extraction, with zero duplicated logic.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ecg_metrics import (
    EcgMetrics,
    ReferenceMetrics,
    LEAD_INDEX,
    _Q_WAVE_LEADS,
    _ONSET_TO_PEAK_LEADS,
    _R_OVER_S_LEADS,
    _R_PROGRESSION_LEADS,
    _NOTCH_LEADS,
    _DEFAULT_PROMINENCE_FRACTION,
    _Q_ONSET_THRESHOLD_FRACTION,
)

try:
    from ansys.health.heart import LOG as LOGGER
except ImportError:  # pragma: no cover - allows standalone use/testing
    import logging
    LOGGER = logging.getLogger("ptbxl_reference")
    if not LOGGER.handlers:
        LOGGER.addHandler(logging.NullHandler())


# PTB-XL lead order in the raw signal arrays returned by wfdb.rdrecord /
# wfdb.rdsamp, before renaming to match LEAD_INDEX (AVR/AVL/AVF -> aVR/aVL/aVF).
_PTBXL_LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"]

_PTBXL_RENAME = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}

# Fallback global-amplitude threshold fraction for QRS onset/offset detection
# when no wfdb annotation file is available. Applied to the global signal
# (max |amplitude| across all 12 leads), per project decision. The clinical
# PTB-XL records500 signals are already bandpass-filtered 0.05-150 Hz, so no
# additional filtering is performed here.
_GLOBAL_THRESHOLD_FRACTION = 0.05

# wfdb annotation symbol conventions (MIT-BIH style) used when wave-delineation
# annotations are available for a PTB-XL record: '(' = wave onset, ')' = wave
# offset, and the QRS complex onset/offset are bracketed around the 'N'
# (normal beat) fiducial point closest to record center.
# However, the annotations are not available for the tested PTB-XL records
# The delimitations defined by these symbols are not guaranteed to 
# work, the annotations should be checked before using this method
_QRS_ONSET_SYMBOLS = ("(",)
_QRS_OFFSET_SYMBOLS = (")",)

# Isoelectric baseline correction window, relative to the QRS onset of the
# selected beat: average the signal over [onset - 40ms, onset - 10ms] per
# lead and subtract it from the whole signal for that lead, so the
# pre-QRS baseline sits at 0 mV before any biomarker is measured (project
# decision — PTB-XL signals are clean of major artifacts by the upstream
# NORM/quality filtering, but baseline_drift can still shift the
# zero-level even on "clean" records).
_BASELINE_WINDOW_START_MS = 40.0
_BASELINE_WINDOW_END_MS = 10.0

# Minimum gap (ms) enforced between two consecutive detected QRS complexes
# in the threshold-based multi-beat detector, to avoid splitting a single
# wide/notched QRS into two spurious detections. Set comfortably below a
# typical RR interval at high heart rate (~140 bpm -> ~430 ms) while well
# above the widest physiologically expected QRS duration in this project's
# search-space range (150 ms, see DEFAULT_WEIGHTS in ecg_metrics.py).
_MIN_QRS_SEPARATION_MS = 200.0

# Margin (ms) kept after the detected QRS offset when cropping the signal
# window handed to EcgMetrics._identify_qrs_waves for the selected beat,
# so the S-wave-related tail detection has enough samples to work with
# without reaching into the next beat.
_POST_QRS_MARGIN_MS = 150.0


_R_ONSET_THRESHOLD_FRACTION = 0.05

# Leads excluded from the real-QRS-bounds calculation in
# _compute_real_qrs_bounds_from_first_pass (project decision).
# because these leads can sometimes be almost isoelectric
# and cause the QRS detection to trigger on noise 
_BOUNDS_EXCLUDED_LEADS = ("aVR", "III")


def _rename_ptbxl_leads(signal: np.ndarray) -> dict:
    """
    Map a PTB-XL raw signal array (samples x 12 leads, PTB-XL lead order)
    to a dict keyed by `LEAD_INDEX` lead names.

    Parameters
    ----------
    signal : numpy.ndarray
        Shape (n_samples, 12), columns in `_PTBXL_LEAD_ORDER` order, as
        returned by `wfdb.rdrecord(...).p_signal`.

    Returns
    -------
    dict
        Mapping `LEAD_INDEX` lead name -> 1D numpy.ndarray (n_samples,).
    """
    if signal.shape[1] != 12:
        raise ValueError(
            f"Expected a 12-lead signal, got shape {signal.shape}."
        )

    renamed = {}
    for col_idx, ptbxl_name in enumerate(_PTBXL_LEAD_ORDER):
        lead_name = _PTBXL_RENAME.get(ptbxl_name, ptbxl_name)
        if lead_name not in LEAD_INDEX:
            raise ValueError(
                f"PTB-XL lead '{ptbxl_name}' renamed to '{lead_name}' "
                f"does not match any key in LEAD_INDEX."
            )
        renamed[lead_name] = signal[:, col_idx]
    return renamed


def _detect_all_qrs_windows_from_annotations(ann, fs: float) -> list:
    """
    Derive ALL QRS windows [t_start_ms, t_end_ms] from wfdb wave-delineation
    annotations, sorted in chronological order.

    Parameters
    ----------
    ann : wfdb.Annotation
        Annotation object returned by `wfdb.rdann(record_path, "...")`.
        Expected to contain onset ('(') / offset (')') bracket symbols
        around QRS fiducial points, as produced by PTB-XL's wave-delineation
        annotators where available.
    fs : float
        Sampling frequency (Hz), used to convert annotation sample indices
        to milliseconds.

    Returns
    -------
    list of tuple(float, float)
        `(t_start_ms, t_end_ms)` for every valid onset/offset bracket pair
        found, in chronological order. Empty list if no usable pair is
        found (caller should fall back to threshold-based detection).

    Notes
    -----
    PTB-XL wave-delineation annotations were not available for the tested records.
    The whole method including annotations should be checked before being used
    """
    if ann is None or len(ann.sample) == 0:
        return []

    symbols = ann.symbol
    samples = ann.sample

    onset_indices = [i for i, s in enumerate(symbols) if s in _QRS_ONSET_SYMBOLS]
    offset_indices = [i for i, s in enumerate(symbols) if s in _QRS_OFFSET_SYMBOLS]

    if not onset_indices or not offset_indices:
        return []

    # Pair each onset with the nearest following offset.
    pairs = []
    for onset_i in onset_indices:
        following_offsets = [j for j in offset_indices if j > onset_i]
        if following_offsets:
            offset_i = min(following_offsets)
            pairs.append((onset_i, offset_i))

    windows = []
    for onset_i, offset_i in pairs:
        onset_sample, offset_sample = samples[onset_i], samples[offset_i]
        t_start_ms = float(onset_sample) / fs * 1000.0
        t_end_ms = float(offset_sample) / fs * 1000.0
        if t_end_ms <= t_start_ms:
            LOGGER.warning(
                "Skipping a non-positive annotation-derived QRS window "
                f"(start={t_start_ms:.1f} ms, end={t_end_ms:.1f} ms)."
            )
            continue
        windows.append((t_start_ms, t_end_ms))

    windows.sort(key=lambda w: w[0])
    return windows


def _detect_all_qrs_windows_from_threshold(
    leads: dict, times: np.ndarray, threshold_fraction: float = _GLOBAL_THRESHOLD_FRACTION
) -> list:
    """
    Fallback multi-beat QRS detection: groups every contiguous run of
    samples where the composite signal (max |amplitude| across all 12
    leads) exceeds `threshold_fraction * global max |amplitude|` into a
    separate QRS window, one per detected beat.

    Parameters
    ----------
    leads : dict
        Mapping lead name -> 1D numpy.ndarray (n_samples,), already renamed
        to `LEAD_INDEX` keys.
    times : numpy.ndarray
        Timestamps in ms, shape (n_samples,).
    threshold_fraction : float, optional
        Fraction of the global absolute maximum used as the crossing
        threshold. Default is `_GLOBAL_THRESHOLD_FRACTION` (5 %).

    Returns
    -------
    list of tuple(float, float)
        `(t_start_ms, t_end_ms)` for every detected QRS complex, in
        chronological order.

    Raises
    ------
    RuntimeError
        If the signal never exceeds the threshold (record likely
        flat-lined, mis-scaled, or corrupted).

    Notes
    -----
    This is a coarse, global, multi-lead method intentionally distinct from
    `EcgMetrics._find_threshold_crossing` (which is single-lead and used for
    per-wave onset/offset within an already-known window). It exists only to
    bootstrap the per-beat windows when no annotation is available.

    A 10-second PTB-XL record typically contains ~8-15 beats: this
    function is the one responsible for not collapsing them into a single
    spurious "QRS window" spanning the whole recording.
    """
    global_max_abs = max(float(np.max(np.abs(sig))) for sig in leads.values())
    threshold = threshold_fraction * global_max_abs

    composite = np.max(np.abs(np.vstack(list(leads.values()))), axis=0)
    above = np.where(composite >= threshold)[0]

    if len(above) == 0:
        raise RuntimeError(
            "No QRS window could be detected: signal never exceeds "
            f"{threshold_fraction * 100:.0f}% of the global max amplitude. "
            "Check that the record is not flat-lined or mis-scaled."
        )

    # Group contiguous (or near-contiguous, within _MIN_QRS_SEPARATION_MS)
    # above-threshold indices into separate beat windows.
    sample_period_ms = float(times[1] - times[0]) if len(times) > 1 else 1.0
    min_gap_samples = max(1, int(round(_MIN_QRS_SEPARATION_MS / sample_period_ms)))

    windows = []
    seg_start = above[0]
    seg_end = above[0]
    for idx in above[1:]:
        if idx - seg_end <= min_gap_samples:
            seg_end = idx
        else:
            windows.append((float(times[seg_start]), float(times[seg_end])))
            seg_start = idx
            seg_end = idx
    windows.append((float(times[seg_start]), float(times[seg_end])))

    return windows


def _select_first_beat_window(
    all_windows: list, times: np.ndarray
) -> tuple:
    """
    Select the first complete QRS beat from a list of detected windows,
    ensuring enough samples exist before its onset for the isoelectric
    baseline correction window (project decision: always use the first
    complete beat, never the one closest to the record center).

    Parameters
    ----------
    all_windows : list of tuple(float, float)
        `(t_start_ms, t_end_ms)` per detected beat, in chronological order
        (output of `_detect_all_qrs_windows_from_annotations` or
        `_detect_all_qrs_windows_from_threshold`).
    times : numpy.ndarray
        Timestamps in ms for the full record, used to check how much
        signal is available before the first candidate's onset.

    Returns
    -------
    tuple(float, float)
        `(t_start_ms, t_end_ms)` of the selected beat.

    Raises
    ------
    RuntimeError
        If no detected beat has at least `_BASELINE_WINDOW_START_MS` of
        signal available before its onset (e.g. every detected beat is too
        close to the very start of the recording — pathological edge case,
        not expected on a normal 10 s PTB-XL record with several beats).

    Notes
    -----
    "First complete beat" means: the first detected window that still has
    a usable baseline window before it, not necessarily literally the very
    first window in `all_windows` — if that one is too close to t=0 to fit
    the baseline window, the next one is used instead.
    """
    t_record_start = float(times[0])

    for t_start, t_end in all_windows:
        if t_start - _BASELINE_WINDOW_START_MS >= t_record_start:
            return t_start, t_end

    raise RuntimeError(
        f"No detected beat has at least {_BASELINE_WINDOW_START_MS:.0f} ms "
        f"of signal before its onset (needed for isoelectric baseline "
        f"correction). Detected beat onsets: "
        f"{[round(w[0], 1) for w in all_windows]} ms; "
        f"record starts at {t_record_start:.1f} ms."
    )


def _correct_isoelectric_baseline(
    leads: dict, times: np.ndarray, t_qrs_start: float
) -> dict:
    """
    Subtract, per lead, the mean signal level over
    `[t_qrs_start - _BASELINE_WINDOW_START_MS, t_qrs_start - _BASELINE_WINDOW_END_MS]`
    from the entire lead signal, so the pre-QRS baseline sits at 0 mV.

    Parameters
    ----------
    leads : dict
        Mapping lead name -> 1D numpy.ndarray (n_samples,), already renamed
        to `LEAD_INDEX` keys.
    times : numpy.ndarray
        Timestamps in ms, shape (n_samples,), shared across all leads.
    t_qrs_start : float
        Onset (ms) of the QRS complex used as the baseline-window anchor
        (the selected first beat's onset).

    Returns
    -------
    dict
        Mapping lead name -> baseline-corrected 1D numpy.ndarray, same
        shape as input. A new dict/arrays are returned; `leads` is not
        modified in place.

    Notes
    -----
    The correction is computed and applied per lead independently, since
    baseline drift is not necessarily shared across leads (each lead has
    its own electrode and its own potential drift). The reference window
    is `[onset - 40ms, onset - 10ms]` (project decision), deliberately
    stopping 10 ms before the onset rather than running right up to it, to
    avoid bleeding into the QRS upstroke itself if the detected onset is
    slightly early.
    """
    window_start = t_qrs_start - _BASELINE_WINDOW_START_MS
    window_end = t_qrs_start - _BASELINE_WINDOW_END_MS

    mask = (times >= window_start) & (times <= window_end)
    if not np.any(mask):
        raise RuntimeError(
            f"Baseline correction window [{window_start:.1f}, "
            f"{window_end:.1f}] ms contains no samples — check the "
            f"sampling rate and the selected beat's onset "
            f"({t_qrs_start:.1f} ms)."
        )

    corrected = {}
    for lead_name, signal in leads.items():
        baseline_level = float(np.mean(signal[mask]))
        corrected[lead_name] = signal - baseline_level

    return corrected


def _compute_real_qrs_bounds_from_first_pass(metrics: EcgMetrics) -> tuple:
    """
    Derive the real QRS onset/offset from a first-pass `EcgMetrics` instance
    driven on a user-supplied, deliberately wide window (project decision:
    manual window input).

    Per-lead onset/offset rule (project decision):
    - Onset: Q-wave onset if Q is detected on that lead, otherwise R-wave
      onset (first sample dropping below `_R_ONSET_THRESHOLD_FRACTION` of
      the raw R peak amplitude, scanning backward from the R peak).
    - Offset: S-wave offset (zero-crossing after S) if S is detected,
      otherwise R-wave offset (zero-crossing after R).
    The real QRS bounds are then the EARLIEST onset and LATEST offset
    across the leads in `LEAD_INDEX` EXCLUDING `_BOUNDS_EXCLUDED_LEADS`
    (aVR, III — see that constant's docstring: aVR's structurally
    negative-dominant QRS can cause Q/R mis-assignment, widening the
    computed bounds; III is excluded as a precaution by analogy, not an
    independently confirmed fix). This otherwise mirrors the clinical
    definition of the QRS complex as a global event, and the same
    min/max-across-leads convention already used on the simulated side via
    ventricular activation times.

    Parameters
    ----------
    metrics : EcgMetrics
        Instance after `compute_qrs_waves()` has been called on the
        wide, user-supplied window (first pass — NOT baseline-corrected
        yet).

    Returns
    -------
    tuple(float, float)
        `(t_qrs_start_real, t_qrs_end_real)` in ms.

    Raises
    ------
    RuntimeError
        If not a single lead has a detectable R wave (nothing to anchor
        onset/offset on at all).

    Notes
    -----
    This duplicates a small amount of threshold-crossing logic that exists
    in `EcgMetrics._identify_qrs_waves` (Q-wave onset, zero-crossings) by
    calling `EcgMetrics._find_threshold_crossing` directly rather than
    re-deriving wave_duration (which, in `_identify_qrs_waves`, is computed
    relative to the supplied window bounds, not as a standalone onset/offset). This is a deliberate, isolated duplication local
    to the clinical pipeline, so the validated simulated-side wave-detection
    code in `ecg_metrics.py` is not modified for this manual-window
    workflow.

    `_R_ONSET_THRESHOLD_FRACTION` (5 %) is set HIGHER than
    `_Q_ONSET_THRESHOLD_FRACTION` (2 %) — not symmetric — after testing
    showed 2 % unreliable on low-amplitude R leads (see that constant's
    docstring). Treat 5 % as a free parameter to revisit if real
    recordings show R-onset detection still too early/late.
    """
    onsets = []
    offsets = []

    for lead_name in LEAD_INDEX:
        if lead_name in _BOUNDS_EXCLUDED_LEADS:
            continue
        waves = metrics.qrs_waves[lead_name]
        signal = metrics._ecg_12lead[LEAD_INDEX[lead_name]]
        times = metrics._times
        norm = metrics.qrs_peak_amplitudes[lead_name]
        norm_value = norm if not np.isnan(norm) else None

        # --- Onset: Q onset if Q present, else R onset ---
        if waves.Q.n_peaks > 0:
            t_q = waves.Q.time
            q_value_raw = waves.Q.value * norm_value if norm_value is not None else waves.Q.value
            threshold_q_onset = _Q_ONSET_THRESHOLD_FRACTION * q_value_raw  # negative
            t_onset = metrics._find_threshold_crossing(
                signal, times,
                t_start=times[0], t_end=t_q,
                threshold=threshold_q_onset, direction="forward", crossing="below",
            )
        elif waves.R.n_peaks > 0:
            t_r = waves.R.time
            r_value_raw = waves.R.value * norm_value if norm_value is not None else waves.R.value
            threshold_r_onset = _R_ONSET_THRESHOLD_FRACTION * r_value_raw  # positive
            t_onset = metrics._find_threshold_crossing(
                signal, times,
                t_start=times[0], t_end=t_r,
                threshold=threshold_r_onset, direction="forward", crossing="above",
            )
        else:
            t_onset = None  # no R, no Q on this lead -> cannot anchor onset here

        # --- Offset: S offset (zero-crossing after S) if S present, else R offset ---
        if waves.S.n_peaks > 0:
            t_s = waves.S.time
            t_offset = metrics._find_threshold_crossing(
                signal, times,
                t_start=t_s, t_end=times[-1],
                threshold=0.0, direction="forward", crossing="above",
            )
        elif waves.R.n_peaks > 0:
            t_r = waves.R.time
            t_offset = metrics._find_threshold_crossing(
                signal, times,
                t_start=t_r, t_end=times[-1],
                threshold=0.0, direction="forward", crossing="above",
            )
        else:
            t_offset = None

        if t_onset is not None:
            onsets.append(t_onset)
        if t_offset is not None:
            offsets.append(t_offset)

    if not onsets or not offsets:
        raise RuntimeError(
            "Could not anchor a QRS onset/offset on ANY of the leads "
            f"considered (all leads except {_BOUNDS_EXCLUDED_LEADS}, "
            "see _BOUNDS_EXCLUDED_LEADS) — no R wave detected anywhere in "
            "the supplied window. Check that the manually-supplied window "
            "actually contains a QRS complex, or widen it / lower the "
            "prominence_fraction used in compute_qrs_waves()."
        )

    t_qrs_start_real = min(onsets)
    t_qrs_end_real = max(offsets)

    LOGGER.info(
        f"Real QRS bounds from first pass: [{t_qrs_start_real:.1f}, "
        f"{t_qrs_end_real:.1f}] ms (earliest onset / latest offset across "
        f"{len(onsets)}/{len(offsets)} leads with a detectable anchor)."
    )

    return t_qrs_start_real, t_qrs_end_real


@dataclass
class _ClinicalQrsWindowPart:
    """Stand-in for a PyAnsys anatomical `Part`, exposing only `get_node_ids`."""
    node_ids: np.ndarray

    def get_node_ids(self, mesh):
        return self.node_ids


@dataclass
class _ClinicalQrsWindowModel:
    """
    Minimal stand-in for `ansys.health.heart.models.HeartModel`.

    Encodes only the two scalars `EcgMetrics._identify_qrs_waves` actually
    reads through `model.{left_ventricle,right_ventricle,septum}` and
    `activation_times.data`: the QRS window start (`t_start`) and end bound
    (`t_end`). All three "parts" share the same two synthetic node IDs
    (0 -> t_start, 1 -> t_end), so `min`/`max` activation-time lookups
    resolve to the clinical QRS window bounds for every part, exactly as a
    real ventricular activation-time field would for a single QRS complex.
    """
    left_ventricle: _ClinicalQrsWindowPart
    right_ventricle: _ClinicalQrsWindowPart
    septum: _ClinicalQrsWindowPart
    mesh: object = None


class _ActivationTimesStub:
    """Stand-in for the `dpf.core.Field` returned by `model.get_activation_times()`."""
    def __init__(self, data: np.ndarray):
        self.data = data


def _make_clinical_ecg_metrics(
    leads: dict, times: np.ndarray, t_qrs_start: float, t_qrs_end: float
) -> EcgMetrics:
    """
    Build an `EcgMetrics` instance driven by a clinical 12-lead signal,
    bypassing the need for a real PyAnsys `model`/`post` object.

    Parameters
    ----------
    leads : dict
        Mapping lead name -> 1D numpy.ndarray, renamed to `LEAD_INDEX` keys.
        Expected to already be cropped to a single-beat window and
        isoelectric-baseline-corrected by the caller (see
        `_correct_isoelectric_baseline` and `_select_first_beat_window` in
        `load_ptbxl_reference`) — this function does not perform either
        step itself, it only wires the (already-prepared) signal into a
        stand-in `EcgMetrics` instance.
    times : numpy.ndarray
        Timestamps in ms, shared across all leads. Expected to cover only
        the cropped single-beat window, not the full multi-beat recording.
    t_qrs_start, t_qrs_end : float
        QRS window bounds in ms (from annotations or threshold fallback),
        for the single selected beat.

    Returns
    -------
    EcgMetrics
        Instance with `qrs_duration` and `qrs_peak_amplitudes` already
        resolvable, ready for `_identify_qrs_waves(lead)` calls per lead.

    Notes
    -----
    `EcgMetrics.__init__` is bypassed (`__new__`) because it requires a real
    `model`/`post` pair that does not exist for a clinical recording. Every
    private attribute it would normally set is initialised by hand below,
    mirroring `EcgMetrics.__init__` exactly so the rest of the class behaves
    identically (lazy properties, `compute_qrs_waves`, etc. all work as
    designed).
    """
    ecg_array = np.zeros((12, len(times)))
    for lead_name, idx in LEAD_INDEX.items():
        ecg_array[idx] = leads[lead_name]

    metrics = EcgMetrics.__new__(EcgMetrics)
    metrics._model = _ClinicalQrsWindowModel(
        left_ventricle=_ClinicalQrsWindowPart(np.array([0, 1])),
        right_ventricle=_ClinicalQrsWindowPart(np.array([0, 1])),
        septum=_ClinicalQrsWindowPart(np.array([0, 1])),
        mesh=None,
    )
    metrics._post = None
    metrics._activation_times = _ActivationTimesStub(
        data=np.array([t_qrs_start, t_qrs_end])
    )
    metrics._ecg_12lead = ecg_array
    metrics._times = times
    metrics._qrs_waves_prominence_fraction = _DEFAULT_PROMINENCE_FRACTION

    metrics._repolarization_times = None
    metrics._qrs_duration = float(t_qrs_end - t_qrs_start)
    metrics._qt_interval = None
    metrics._p_wave_duration = None
    metrics._pq_interval = None
    metrics._qrs_peak_amplitudes = None
    metrics.qrs_waves = None
    metrics._r_progression = None
    metrics._electrical_axis = None
    metrics._implausibility = None
    metrics._implausibility_reference = None

    return metrics


def _extract_reference_biomarkers(metrics: EcgMetrics, source: str) -> ReferenceMetrics:
    """
    Run the full QRS biomarker catalogue extraction on a clinical-driven
    `EcgMetrics` instance and package the result as `ReferenceMetrics`.

    Parameters
    ----------
    metrics : EcgMetrics
        Instance produced by `_make_clinical_ecg_metrics`, with
        `compute_qrs_waves()` not yet called.
    source : str
        Free-text provenance string stored in `ReferenceMetrics.source`.

    Returns
    -------
    ReferenceMetrics
    """
    metrics.compute_qrs_waves()

    q_duration = {}
    q_over_r = {}
    for lead in _Q_WAVE_LEADS:
        waves = metrics.qrs_waves[lead]
        q_duration[lead] = waves.Q.wave_duration if waves.Q.n_peaks > 0 else np.nan
        if waves.R.n_peaks > 0:
            q_min = float(np.min(waves.Q.peak_values)) if waves.Q.n_peaks > 0 else 0.0
            r_max = float(np.max(waves.R.peak_values))
            q_over_r[lead] = abs(q_min) / r_max
        else:
            q_over_r[lead] = np.nan

    onset_to_peak = {}
    t_qrs_start = metrics._activation_times.data[0]
    for lead in _ONSET_TO_PEAK_LEADS:
        r_wave = metrics.qrs_waves[lead].R
        onset_to_peak[lead] = (
            float(r_wave.time - t_qrs_start) if r_wave.n_peaks > 0 else np.nan
        )

    r_progression_monotony = metrics.r_progression[_R_PROGRESSION_LEADS[0]].r_over_s_monotony_penalty

    r_over_s = {}
    for lead in _R_OVER_S_LEADS:
        waves = metrics.qrs_waves[lead]
        r_present = waves.R.n_peaks > 0
        s_present = waves.S.n_peaks > 0
        if not r_present:
            r_over_s[lead] = 0.0
        elif not s_present:
            r_over_s[lead] = np.nan
        else:
            r_max = float(np.max(waves.R.peak_values))
            s_min = float(np.min(waves.S.peak_values))
            r_over_s[lead] = r_max / abs(s_min) if abs(s_min) > 1e-9 else np.nan

    q_wave_v1 = metrics.qrs_waves["V1"].Q
    q_amplitude_v1 = (
        float(abs(np.min(q_wave_v1.peak_values))) if q_wave_v1.n_peaks > 0 else 0.0
    )

    # Same convention as EcgMetrics._implausibility_notches: identify the
    # deepest notch per wave/lead, then read depth and interval at that same
    # index, so the clinical target characterises one identified notch on
    # both dimensions rather than mixing the deepest notch's depth with an
    # unrelated notch's width.
    notch_depth = {wave: {} for wave in _NOTCH_LEADS}
    notch_interval = {wave: {} for wave in _NOTCH_LEADS}
    for wave_name, lead_list in _NOTCH_LEADS.items():
        for lead in lead_list:
            wave = getattr(metrics.qrs_waves[lead], wave_name)
            if wave.notch_depths.size > 0:
                deepest_idx = int(np.argmax(wave.notch_depths))
                notch_depth[wave_name][lead] = float(wave.notch_depths[deepest_idx])
                notch_interval[wave_name][lead] = float(wave.peak_intervals[deepest_idx])
            else:
                notch_depth[wave_name][lead] = 0.0
                notch_interval[wave_name][lead] = 0.0

    electrical_axis = metrics.electrical_axis

    # Same signed-sum convention as EcgMetrics._implausibility_extra_extrema:
    # maxima contribute positively, minima negatively, an empty sum is 0.0
    # (no "absent" state distinct from "zero" for this criterion).
    extra_extrema = {}
    for lead in LEAD_INDEX:
        extrema = metrics.qrs_waves[lead].extra_extrema
        extra_extrema[lead] = float(np.sum(extrema["maxima"]["values"])) + float(
            np.sum(extrema["minima"]["values"])
        )

    LOGGER.info(
        f"PTB-XL reference extracted from '{source}': "
        f"qrs_duration={metrics.qrs_duration:.1f} ms, "
        f"electrical_axis={electrical_axis if np.isnan(electrical_axis) else f'{electrical_axis:.1f}'}°."
    )

    return ReferenceMetrics(
        qrs_duration=float(metrics.qrs_duration),
        q_duration=q_duration,
        q_over_r=q_over_r,
        onset_to_peak=onset_to_peak,
        r_progression_monotony=float(r_progression_monotony) if not np.isnan(r_progression_monotony) else np.nan,
        r_over_s=r_over_s,
        q_amplitude_v1=q_amplitude_v1,
        notch_depth=notch_depth,
        notch_interval=notch_interval,
        electrical_axis=float(electrical_axis) if not np.isnan(electrical_axis) else np.nan,
        extra_extrema=extra_extrema,
        source=source,
    )


def load_ptbxl_reference(record_path: str, ann_path: str = None) -> ReferenceMetrics:
    """
    Load a PTB-XL clinical 12-lead ECG record and extract the QRS biomarker
    catalogue as a `ReferenceMetrics` instance.

    Parameters
    ----------
    record_path : str
        Path to the PTB-XL record, without extension, as expected by
        `wfdb.rdrecord` (e.g. ``".../records500/00000/00001_hr"``).
        Use the `records500` (500 Hz) subdirectory to benefit from
        wave-delineation annotations where available (see `ann_path`).
    ann_path : str, optional
        Path (without extension) to a wfdb annotation file for this record,
        passed to `wfdb.rdann(ann_path, extension=...)`. PTB-XL ships
        wave-delineation annotations for only a subset of records; when
        `None` or when annotations cannot be read, this function falls back
        to global-amplitude threshold detection (5 % of the global max
        absolute amplitude across all 12 leads — the PTB-XL `records500`
        signals are already bandpass-filtered 0.05-150 Hz, so no additional
        filtering step is applied here).

    Returns
    -------
    ReferenceMetrics
        Pre-computed clinical biomarkers, ready to be passed to
        `EcgMetrics.compute_implausibility(reference)`.

    Raises
    ------
    ImportError
        If the `wfdb` package is not installed.
    RuntimeError
        If no QRS beat could be detected at all (record likely flat-lined,
        mis-scaled, or corrupted), or if no detected beat has enough signal
        before its onset for the isoelectric baseline correction window.
    ValueError
        If the record does not contain exactly 12 leads in the expected
        PTB-XL order.

    Notes
    -----
    A 10 s PTB-XL record contains several heartbeats (typically ~8-15).
    This function:

    1. Detects every QRS complex in the record (via wfdb annotations if
       available, otherwise a multi-beat threshold detector — see
       `_detect_all_qrs_windows_from_annotations` /
       `_detect_all_qrs_windows_from_threshold`).
    2. Selects the first complete beat (project decision — not the beat
       closest to the record center), skipping ahead only if the very
       first detected beat does not have enough signal before its onset
       for the baseline correction window (see
       `_select_first_beat_window`).
    3. Corrects the isoelectric baseline per lead, subtracting the mean
       signal level over `[onset - 40ms, onset - 10ms]` so the pre-QRS
       baseline sits at 0 mV before any biomarker is measured (see
       `_correct_isoelectric_baseline`).
    4. Crops the signal to a window around that single selected beat
       (from `onset - _BASELINE_WINDOW_START_MS` to
       `offset + _POST_QRS_MARGIN_MS`) before handing it to `EcgMetrics`,
       so later beats in the recording cannot interfere with wave
       detection for the selected one.

    This function is the clinical-side counterpart of `EcgMetrics`: it
    drives the *same* `_identify_qrs_waves` wave-detection algorithm used
    for simulated ECGs, so the resulting biomarkers
    are extracted with an identical methodology, differing only in the
    upstream beat-selection and baseline-correction steps above (which
    have no equivalent on the simulated side, since PyAnsys simulations used
    only produced one beat with an already-zeroed baseline).

    Filtering convention (project decision): select PTB-XL records where
    `scp_codes` contains `"NORM"` with `likelihood == 100.0` in
    `ptbxl_database.csv` *before* calling this function — this loader only
    reads and processes a single already-selected record; it does not query
    the PTB-XL database index.

    Examples
    --------
    >>> reference = load_ptbxl_reference(
    ...     "/data/ptbxl/records500/00000/00001_hr",
    ...     ann_path="/data/ptbxl/records500/00000/00001_hr",
    ... )
    >>> metrics.compute_qrs_waves()
    >>> result = metrics.compute_implausibility(reference)
    """
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError(
            "The 'wfdb' package is required to load PTB-XL records. "
            "Install it with: pip install wfdb"
        ) from exc

    record = wfdb.rdrecord(record_path)
    fs = float(record.fs)
    n_samples = record.p_signal.shape[0]
    times_full = np.arange(n_samples) / fs * 1000.0  # ms

    leads_full = _rename_ptbxl_leads(record.p_signal)

    all_windows = []
    if ann_path is not None:
        try:
            ann = wfdb.rdann(ann_path, extension="atr")
            all_windows = _detect_all_qrs_windows_from_annotations(ann, fs)
        except Exception as exc:  # noqa: BLE001 - any wfdb/annotation issue -> fallback
            LOGGER.warning(
                f"Could not read or parse wfdb annotations at '{ann_path}' "
                f"({exc}); falling back to threshold-based QRS detection."
            )
            all_windows = []

    if all_windows:
        LOGGER.info(
            f"{len(all_windows)} QRS beat(s) detected via wfdb annotations."
        )
    else:
        all_windows = _detect_all_qrs_windows_from_threshold(leads_full, times_full)
        LOGGER.info(
            f"{len(all_windows)} QRS beat(s) detected via "
            f"{_GLOBAL_THRESHOLD_FRACTION * 100:.0f}% global-amplitude threshold."
        )

    t_qrs_start, t_qrs_end = _select_first_beat_window(all_windows, times_full)
    LOGGER.info(
        f"Selected first complete beat: [{t_qrs_start:.1f}, {t_qrs_end:.1f}] ms."
    )

    leads_corrected = _correct_isoelectric_baseline(leads_full, times_full, t_qrs_start)

    crop_start = t_qrs_start - _BASELINE_WINDOW_START_MS
    crop_end = t_qrs_end + _POST_QRS_MARGIN_MS
    crop_mask = (times_full >= crop_start) & (times_full <= crop_end)

    times_cropped = times_full[crop_mask]
    leads_cropped = {name: sig[crop_mask] for name, sig in leads_corrected.items()}

    metrics = _make_clinical_ecg_metrics(leads_cropped, times_cropped, t_qrs_start, t_qrs_end)
    return _extract_reference_biomarkers(metrics, source=record_path)


def load_ptbxl_reference_manual_window(
    record_path: str, manual_window: tuple
) -> ReferenceMetrics:
    """
    Load a PTB-XL clinical 12-lead ECG record and extract the QRS biomarker
    catalogue, using a MANUALLY-SUPPLIED, deliberately wide time window to
    locate the beat instead of automatic multi-beat detection.

    Project rationale: PTB-XL ships no wave-delineation
    annotation file alongside its `.dat`/`.hea` records (confirmed against
    the official file listing — only a `.dat`/`.hea` pair per record
    exists in the base PTB-XL dataset), so
    `_detect_all_qrs_windows_from_annotations` never actually fires in
    practice; and the threshold-based fallback
    (`_detect_all_qrs_windows_from_threshold`) has no morphological
    selectivity between P, QRS, and T — it can capture P and/or T inside
    what it reports as a "QRS window" on records where these waves are not
    negligible in amplitude relative to the global threshold. For
    calibrating on a handful of hand-picked records under a tight
    deadline, manually supplying an approximate window (read off Figure 1
    of `test_ptbxl_biomarker_detection.py`, a few tens of ms wider than
    the true QRS on each side) and letting wave detection narrow it down
    is far more reliable than fixing the general-purpose automatic
    detector for this specific need.

    Parameters
    ----------
    record_path : str
        Path to the PTB-XL record, without extension, as expected by
        `wfdb.rdrecord`.
    manual_window : tuple(float, float)
        `(t_window_start_ms, t_window_end_ms)`, deliberately wider than
        the true QRS complex on this record (a few tens of ms of margin
        on each side is recommended) — read visually from the full-signal
        plot (e.g. Figure 1 in `test_ptbxl_biomarker_detection.py`).

    Returns
    -------
    ReferenceMetrics
        Pre-computed clinical biomarkers, ready to be passed to
        `EcgMetrics.compute_implausibility(reference)`.

    Raises
    ------
    ImportError
        If the `wfdb` package is not installed.
    RuntimeError
        If no R wave can be detected on any lead within `manual_window`
        (see `_compute_real_qrs_bounds_from_first_pass`), or if the
        isoelectric baseline window has no samples available before the
        real QRS onset (e.g. `manual_window` starts too close to the
        record's own t=0).
    ValueError
        If the record does not contain exactly 12 leads in the expected
        PTB-XL order.

    Notes
    -----
    Two-pass procedure (project decision):

    1. **First pass** — run `EcgMetrics._identify_qrs_waves` on
       `manual_window`, on the RAW (not yet baseline-corrected) signal, to
       locate Q/R/S on every lead.
    2. **Real QRS bounds** — from that first pass, derive
       `t_qrs_start_real` / `t_qrs_end_real` as the earliest onset / latest
       offset across all 12 leads (see
       `_compute_real_qrs_bounds_from_first_pass` for the exact per-lead
       onset/offset rule: Q-onset else R-onset; S-offset else R-offset).
    3. **Baseline correction** — subtract, per lead, the mean signal level
       over `[t_qrs_start_real - 40ms, t_qrs_start_real - 10ms]` from the
       whole signal (same convention as `load_ptbxl_reference`).
    4. **Second pass** — re-run `_identify_qrs_waves` on the
       baseline-corrected signal, cropped to
       `[t_qrs_start_real - _BASELINE_WINDOW_START_MS,
       t_qrs_end_real + _POST_QRS_MARGIN_MS]`, to get the final biomarkers.

    Unlike `load_ptbxl_reference`, this function does not attempt automatic
    multi-beat detection at all — `manual_window` is trusted to already
    contain exactly one beat. Pick a record/window where that is visibly
    true (e.g. via `test_ptbxl_biomarker_detection.py`'s Figure 1) before
    calling this function.

    Examples
    --------
    >>> reference = load_ptbxl_reference_manual_window(
    ...     "/data/ptbxl/records500/00000/00001_hr",
    ...     manual_window=(280.0, 420.0),
    ... )
    """
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError(
            "The 'wfdb' package is required to load PTB-XL records. "
            "Install it with: pip install wfdb"
        ) from exc

    record = wfdb.rdrecord(record_path)
    fs = float(record.fs)
    n_samples = record.p_signal.shape[0]
    times_full = np.arange(n_samples) / fs * 1000.0
    leads_full = _rename_ptbxl_leads(record.p_signal)

    t_window_start, t_window_end = manual_window
    window_mask = (times_full >= t_window_start) & (times_full <= t_window_end)
    if not np.any(window_mask):
        raise RuntimeError(
            f"manual_window=({t_window_start}, {t_window_end}) ms contains "
            f"no samples — check it falls within the record's duration "
            f"(0 to {times_full[-1]:.1f} ms)."
        )

    times_window = times_full[window_mask]
    leads_window = {name: sig[window_mask] for name, sig in leads_full.items()}

    # --- Pass 1: detect waves on the raw, wide, user-supplied window ---
    metrics_pass1 = _make_clinical_ecg_metrics(
        leads_window, times_window, t_window_start, t_window_end
    )
    metrics_pass1.compute_qrs_waves()

    t_qrs_start_real, t_qrs_end_real = _compute_real_qrs_bounds_from_first_pass(metrics_pass1)
    LOGGER.info(
        f"Manual window {manual_window} narrowed to real QRS bounds "
        f"[{t_qrs_start_real:.1f}, {t_qrs_end_real:.1f}] ms."
    )

    # --- Baseline correction, anchored on the real QRS onset ---
    leads_corrected = _correct_isoelectric_baseline(leads_full, times_full, t_qrs_start_real)

    # --- Pass 2: re-detect waves on the corrected, cropped signal ---
    crop_start = t_qrs_start_real - _BASELINE_WINDOW_START_MS
    crop_end = t_qrs_end_real + _POST_QRS_MARGIN_MS
    crop_mask = (times_full >= crop_start) & (times_full <= crop_end)
    times_cropped = times_full[crop_mask]
    leads_cropped = {name: sig[crop_mask] for name, sig in leads_corrected.items()}

    metrics_pass2 = _make_clinical_ecg_metrics(
        leads_cropped, times_cropped, t_qrs_start_real, t_qrs_end_real
    )
    return _extract_reference_biomarkers(metrics_pass2, source=record_path)

