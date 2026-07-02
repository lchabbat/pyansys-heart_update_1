"""
ecg_metrics.py
Module dedicated to the extraction of ECG metrics from electrophysiology simulations.
"""

import numpy as np
from ansys.health.heart import LOG as LOGGER
from dataclasses import dataclass
from typing import Optional

@dataclass
class WaveInfo:
    """Stores information about a single ECG wave (Q, R, or S).

    Attributes
    ----------
    time : float
        Timestamp of the primary peak in ms. ``np.nan`` if wave is absent.
    value : float
        Normalized amplitude of the primary peak. ``np.nan`` if wave is absent.
    n_peaks : int
        Total number of peaks of the same polarity detected for this wave.
        0 if wave is absent. > 1 indicates a notched wave.
    peak_times : numpy.ndarray
        Timestamps of all peaks (ms), shape (n_peaks,). Empty if wave is absent.
        All peaks share the same polarity (all maxima for R, all minima for Q/S).
    peak_values : numpy.ndarray
        Normalized amplitudes of all peaks, shape (n_peaks,).
        Empty if wave is absent. Same ordering as ``peak_times``.
    peak_intervals : numpy.ndarray
        Time intervals between consecutive peaks (ms), shape (n_peaks - 1,).
        Each interval is the duration from one peak to the next — equivalent
        to the temporal width of the notch between those two peaks.
        Empty if n_peaks <= 1 or wave is absent.
    notch_depths : numpy.ndarray 
        Relative depth of each notch between consecutive peaks, shape (n_peaks - 1,).
        Defined as ``abs(peak_value - notch_extremum) / abs(peak_value)`` where
        ``notch_extremum`` is the extremum of opposite polarity most distant from
        zero between ``peak_times[i]`` and ``peak_times[i+1]``.
        0.0 if no extremum is detected between two peaks.
        Empty if n_peaks <= 1 or wave is absent.
        Note: There are currently problems with notch detection and notch contribution to the implausibility score
        depths definition should be looked into in details
    wave_duration : float
        Duration of the wave in ms. ``np.nan`` if not calculable.
    """
    time: float
    value: float
    n_peaks: int
    peak_times: np.ndarray
    peak_values: np.ndarray
    peak_intervals: np.ndarray
    notch_depths: np.ndarray
    wave_duration: float


def _absent_wave() -> WaveInfo:
    """Return a WaveInfo sentinel representing an absent wave."""
    return WaveInfo(
        time=np.nan,
        value=np.nan,
        n_peaks=0,
        peak_times=np.array([]),
        peak_values=np.array([]),
        peak_intervals=np.array([]),
        notch_depths=np.array([]),
        wave_duration=np.nan,
    )


@dataclass
class QrsWaves:
    """Stores the identified waves of a QRS complex for a single lead."""
    Q: WaveInfo
    R: WaveInfo
    S: WaveInfo
    extra_extrema: dict

 

_FRONTAL_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF")
"""Tuple of frontal lead names."""

_PRECORDIAL_LEADS = ("V1", "V2", "V3", "V4", "V5", "V6")
"""Tuple of precordial lead names, in anatomical order V1→V6."""
 
_R_PROGRESSION_LEADS = ("V1", "V2", "V3", "V4")
"""Leads used for R-wave progression assessment (V1 to V4)."""
 
 
@dataclass
class RProgressionInfo:
    """
    Stores R-wave progression metrics for precordial leads V1 to V4.

    Per-lead fields (one value per lead in ``_R_PROGRESSION_LEADS``) and one
    global field summarising the monotonicity of the R/S sequence across leads.

    All amplitudes are **normalized** by ``qrs_peak_amplitudes[lead]``
    (= ``max(|signal|)`` in the QRS window), so they are dimensionless
    and comparable across leads with different absolute scales.

    Attributes
    ----------
    lead : str
        Lead name (e.g. ``"V1"``).
    r_amplitude : float
        Normalized amplitude of the R wave (primary peak).
        ``np.nan`` if R wave is absent.
    s_amplitude : float
        Normalized amplitude of the S wave (primary peak, expected ≤ 0).
        ``np.nan`` if S wave is absent.
    r_over_s : float
        Ratio R / |S|.  Positive by construction (|S| in denominator).
        ``np.nan`` if either wave is absent or S amplitude is zero.
    r_present : bool
        ``True`` if R wave was identified (``WaveInfo.n_peaks > 0``).
    s_present : bool
        ``True`` if S wave was identified (``WaveInfo.n_peaks > 0``).
    r_over_s_monotony_penalty : float
        Sum of R/S decreases between consecutive valid leads in V1→V4.
        Computed as ``sum(max(0, r_over_s[i] - r_over_s[i+1]))`` over all
        consecutive pairs where neither value is ``np.nan``.
        Zero indicates a perfectly non-decreasing (normal) progression.
        Larger values indicate stronger or more numerous monotonicity violations.
        ``np.nan`` if fewer than two valid R/S ratios are available across
        ``_R_PROGRESSION_LEADS``.

        This field has the **same value** in every ``RProgressionInfo`` instance
        of a given ``_compute_r_progression`` call — it summarises the whole
        V1–V4 sequence and is stored per-instance for convenience of access
        (``metrics.r_progression["V1"].r_over_s_monotony_penalty``).
    
    Notes
    ----------
    r_over_s could be 0 instead of np.nan if R amplitude is 0 but S amplitude
    isn't. More leads would be used to compute the r_over_s_monotony_penalty
    """

    lead: str
    r_amplitude: float
    s_amplitude: float
    r_over_s: float
    r_present: bool
    s_present: bool
    r_over_s_monotony_penalty: float = np.nan
 

_DEFAULT_MARGIN_MS = 0.0
_DEFAULT_PROMINENCE_FRACTION = 0.02
_Q_ONSET_THRESHOLD_FRACTION = 0.02
"""Fraction of the raw Q peak amplitude used as threshold for Q-wave onset detection.
The onset is defined as the first sample in the QRS window where the signal drops
below ``_Q_ONSET_THRESHOLD_FRACTION * q_peak_amplitude_raw``. Value could be tuned
to be insensitive to baseline noise while remaining below any
physiologically meaningful Q deflection."""

_NOTCH_PROMINENCE_FRACTION = 0.01
"""Very low prominence fraction used when searching for the notch extremum between
two consecutive peaks of the same wave.  A value of 0.01 (1 %) ensures that even
shallow notches are detected.  The deepest extremum is
then selected to maximise the reported notch depth."""

# Hexaxial reference system : lead → (angle_deg, perp_pos, perp_neg)
# perp_pos : list of leads closest to iso_angle+90° (may be tied)
# perp_neg : list of leads closest to iso_angle-90° (may be tied)
# When multiple leads are tied, their net amplitudes are summed to resolve the quadrant.
# Ties occur for I (perp_neg: aVR/aVL at 60°) and III (perp_neg: I/II at 30°).
_HEXAXIAL = {
    "I":   {"angle":    0, "perp_pos": ["aVF"],       "perp_neg": ["aVR", "aVL"]},
    "II":  {"angle":   60, "perp_pos": ["III"],        "perp_neg": ["aVL"]},
    "III": {"angle":  120, "perp_pos": ["aVR"],        "perp_neg": ["I", "II"]},
    "aVR": {"angle": -150, "perp_pos": ["aVL"],        "perp_neg": ["III"]},
    "aVL": {"angle":  -30, "perp_pos": ["II"],         "perp_neg": ["aVR"]},
    "aVF": {"angle":   90, "perp_pos": ["aVR"],        "perp_neg": ["I"]},
}

LEAD_INDEX = {
    "I": 0, "II": 1, "III": 2,
    "aVR": 3, "aVL": 4, "aVF": 5,
    "V1": 6, "V2": 7, "V3": 8,
    "V4": 9, "V5": 10, "V6": 11,
}

# Leads used for the Q-wave biomarkers (duration, Q/R ratio).
_Q_WAVE_LEADS = ("I", "aVL", "V5", "V6")

# Leads used for the QRS onset-to-R-peak biomarker.
_ONSET_TO_PEAK_LEADS = ("V5", "V6")

# Leads used for the R/S amplitude ratio biomarker.
_R_OVER_S_LEADS = ("V1", "V5")

# Leads on which notch depth/interval penalties are evaluated, per wave.
# Deliberately the FULL 12-lead set for all three waves (Q, R, S) — not a
# clinically-motivated subset. Rationale: unlike a real heart, an eikonal /
# monodomain simulation can generate notching on ANY lead if non-physiological 
# parameter combinations are used. Restricting to a
# "clinically plausible" subset would make such artifacts invisible to the
# cost function — and therefore invisible to the optimiser, which could then
# drift toward parameter sets that look fine on the watched leads while
# producing unphysiological notching elsewhere. Every lead must be checked.
_NOTCH_LEADS = {
    "Q": tuple(LEAD_INDEX.keys()),
    "R": tuple(LEAD_INDEX.keys()),
    "S": tuple(LEAD_INDEX.keys()),
}

# Conventional penalty multiplier applied (in units of sigma) when a wave is
# present on one side (simulated or reference) and absent on the other side
# for a biomarker that depends on that wave. See project decisions: "5 *
# sigma[k]" is used as the fixed punitive value rather than np.nan, so the
# total implausibility score remains finite and comparable across particles.
_NAN_PENALTY_SIGMA_MULTIPLE = 5.0

# ---------------------------------------------------------------------------
# Default weights for the implausibility score.
#
# Each weight is a normalisation factor, not a measurement-uncertainty
# estimate: weight = 1 / sigma_estimate, where sigma_estimate = (max - min)/4
# for a defined plausible range [min, max] of that criterion. This
# keeps every criterion's contribution to the sum on a comparable scale
# regardless of its native unit (ms, degrees, dimensionless ratio), without
# claiming any statistical meaning the underlying range estimate doesn't
# have.
#
# IMPORTANT - nature of the ranges used:
#   - Most ranges below are NOT clinical-normal ranges. They are numerically
#     expected ranges for the simulated ECG outputs over the full parameter
#     search space planned for the inverse-problem optimisation (Purkinje
#     conduction speed, fast-pathway conduction speed, earliest activation
#     sites, myocardial conduction speed) - i.e. "what
#     this eikonal solver could plausibly output", including
#     mildly-to-clearly pathological-looking QRS complexes that may result
#     from a poorly-tuned parameter set, not just healthy
#     variation. Several of these ranges are deliberately oversized
#     (project decision) as a safety margin against
#     underestimating the true achievable spread - see per-criterion notes.
#   - `qrs_duration` [45, 150] ms and `q_duration` [0, 60] ms: oversized
#     relative to clinical-normal (80-110 ms and 20-40 ms respectively) on
#     purpose, as a margin for off-nominal simulated outputs.
#   - `q_over_r`, `r_over_s_V1` [0, 5]: expert estimate, explicitly flagged
#     as uncertain - the modeller was not confident in these bounds.
#   - `r_over_s_V5` [1, 10]: applies ONLY to the residual case where S is
#     present on both sides. The expected case for a healthy V5
#     (S absent, R present) is intercepted upstream of this range entirely
#     and assigned a penalty of 0.0 by construction - see
#     `_implausibility_r_over_s`.
#   - `r_progression_monotony` [0, 15]: NOT an independently-estimated
#     range. `monotony_penalty = sum(max(0, r_curr - r_next))` over at most
#     3 consecutive-lead pairs (V1-V2, V2-V3, V3-V4) is mathematically
#     unbounded (R/S ratios are themselves unbounded). 15 is a worst-case
#     bound derived FROM the `r_over_s_V1`-leads range of 5 (3 pairs x a
#     maximal single-pair drop of 5), not a separately validated estimate.
#   - `notch_interval` [0, 100] ms: deliberately oversized - 100 ms is
#     comparable to a full QRS duration, far beyond what a true "notch"
#     (a local morphological artifact) should measure. Kept as a wide
#     safety margin.
#   - `extra_extrema` [-1, 1]: a true mathematical bound, like
#     `electrical_axis` — `extra_extrema_{lead}` is the SIGNED sum of every
#     late-extremum's normalized value on that lead (maxima positive,
#     minima negative, see `_implausibility_extra_extrema`), and each
#     individual normalized value is itself bounded in [-1, 1] by
#     construction (normalized by `qrs_peak_amplitudes[lead]`, the lead's
#     own dominant QRS peak). The signed sum across several extrema is NOT
#     separately re-bounded to [-1, 1] (it could in principle exceed this
#     range with multiple same-sign extrema) — [-1, 1] is used as the
#     plausible single-extremum-dominated range, per project decision.
#   - `electrical_axis` [0, 180] degrees: the maximum possible circular
#     distance, matching the bound already enforced by the circular-penalty
#     formula itself (`((diff + 180) % 360) - 180` is always in [-180, 180]).
#   - `axis_ambiguous` has no underlying range (binary flag): weight kept at
#     1.0 so the flag's penalty (0.0 or 1.0) contributes directly to the
#     score, unscaled.
#
# These weights remain explicit, named, and overridable on every call to
# compute_implausibility() so they can be revisited once better estimates
# (e.g. a General Sensitivity Analysis) become available.
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "qrs_duration": 0.038095,            # range [45, 150] ms
    "q_duration": 0.066667,              # range [0, 60] ms
    "q_over_r": 0.8,                     # range [0, 5] -- ratios are unbound, difficult to choose a plausible range
    "onset_to_peak": 0.072727,           # range [20, 75] ms
    "r_progression_monotony": 0.266667,  # range [0, 15] -- derived, see note above, ratios are unbound, difficult to choose a plausible range
    "r_over_s_V1": 0.8,                  # range [0, 5] -- ratios are unbound, difficult to choose a plausible range
    "r_over_s_V5": 0.444444,             # range [1, 10] -- residual case only, see note above
    "q_amplitude_v1": 4.0,               # range [0, 1] -- true mathematical bound
    "notch_depth": 4.0,                  # range [0, 1] -- true mathematical bound
    "notch_interval": 0.04,              # range [0, 100] ms -- deliberately oversized
    "extra_extrema": 2.0,                # range [-1, 1] -- true mathematical bound (signed sum of normalized late-extrema values, see project decision)
    "electrical_axis": 0.022222,         # range [0, 180] degrees -- true mathematical bound
    "axis_ambiguous": 1.0,               # binary flag, no range -- unscaled
}



@dataclass
class ReferenceMetrics:
    """
    Pre-computed ECG biomarkers from a clinical reference recording.

    This dataclass is intentionally independent from :class:`EcgMetrics`:
    a clinical 12-lead ECG (e.g. from PTB-XL) has no associated
    ``model``/``post`` PyAnsys objects, so it cannot be wrapped in an
    ``EcgMetrics`` instance. ``compute_implausibility`` therefore takes a
    ``ReferenceMetrics`` instance rather than another ``EcgMetrics``.

    All biomarkers mirror the catalogue computed by ``EcgMetrics`` for the
    simulated ECG, restricted to the QRS complex (project scope, first
    iteration). Per-lead biomarkers are stored as ``dict[str, float]`` keyed
    by lead name, using only the leads relevant to that biomarker (see
    ``_Q_WAVE_LEADS``, ``_ONSET_TO_PEAK_LEADS``, ``_R_OVER_S_LEADS``).

    Attributes
    ----------
    qrs_duration : float
        Total QRS duration (ms).
    q_duration : dict[str, float]
        Q-wave duration (ms) per lead in ``_Q_WAVE_LEADS``.
        ``np.nan`` if Q is absent on that lead in the reference.
    q_over_r : dict[str, float]
        Intra-lead ``|min(Q)| / max(R)`` ratio per lead in ``_Q_WAVE_LEADS``.
        ``np.nan`` if R is absent on that lead in the reference.
    onset_to_peak : dict[str, float]
        QRS-onset-to-R-peak duration (ms) per lead in ``_ONSET_TO_PEAK_LEADS``.
        ``np.nan`` if R is absent on that lead in the reference.
    r_progression_monotony : float
        R/S monotonicity penalty across V1-V4 (typically ``0.0`` for a
        normal reference recording).
    r_over_s : dict[str, float]
        ``max(R) / |min(S)|`` ratio per lead in ``_R_OVER_S_LEADS``.
    q_amplitude_v1 : float
        Normalized Q-wave amplitude in V1 (typically ``0.0``: Q is normally
        absent in V1).
    notch_depth : dict[str, dict[str, float]]
        ``notch_depth[wave][lead]`` = depth of the **deepest** notch for the
        given wave (``"Q"``, ``"R"``, or ``"S"``) and lead, restricted to
        ``_NOTCH_LEADS[wave]``. ``0.0`` if the wave has fewer than 2 peaks
        (no notch) or is absent.
    notch_interval : dict[str, dict[str, float]]
        ``notch_interval[wave][lead]`` = temporal width (ms) of that **same**
        deepest notch (i.e. ``peak_intervals`` at the index of
        ``argmax(notch_depths)``), same structure as ``notch_depth``.
        ``0.0`` if the wave has fewer than 2 peaks (no notch) or is absent.
        Depth and interval are always read at the same notch index so the
        two criteria characterise one identified notch, not two unrelated
        ones.
    electrical_axis : float
        Mean QRS electrical axis in the frontal plane (degrees).
        ``np.nan`` if ambiguous in the reference recording.
    extra_extrema : dict[str, float]
        ``extra_extrema[lead]`` = SIGNED sum of every late extremum's
        normalized value detected after the S-complex on that lead (see
        ``QrsWaves.extra_extrema`` / ``EcgMetrics._identify_qrs_waves``),
        across all 12 leads in ``LEAD_INDEX``. Maxima contribute positively,
        minima negatively; an empty sum (no late extrema detected) is
        ``0.0``, not ``np.nan`` — there is no "absent" state distinct from
        "zero" for this criterion. Typically ``0.0`` everywhere for a clean
        reference recording.
    source : str
        Free-text identifier of the reference recording's provenance
        (e.g. PTB-XL record path), for traceability/debugging only.
    """
    qrs_duration: float
    q_duration: dict
    q_over_r: dict
    onset_to_peak: dict
    r_progression_monotony: float
    r_over_s: dict
    q_amplitude_v1: float
    notch_depth: dict
    notch_interval: dict
    electrical_axis: float
    extra_extrema: dict
    source: str = ""


@dataclass
class ImplausibilityResult:
    """
    Result of comparing a simulated ECG to a :class:`ReferenceMetrics`.

    Attributes
    ----------
    total_score : float
        Weighted sum of all per-criterion penalties::

            total_score = sum(
                weights[k] * penalty_vector[k]
                for k in penalty_vector
                if not np.isnan(penalty_vector[k])
            )

        Lower is better (0.0 = perfect match on every criterion). Each
        weight is a normalisation factor (``weight = 1 / sigma_estimate``,
        with ``sigma_estimate`` derived from defined plausible
        range for that criterion over the optimisation parameter space),
        not a measurement-uncertainty estimate — see ``DEFAULT_WEIGHTS``.
    penalty_vector : dict[str, float]
        Raw (un-normalised, un-weighted) penalty value per criterion key.
        ``np.nan`` entries are excluded from ``total_score`` -- see
        ``flags`` for cases where a NaN was converted into a conventional
        finite penalty instead (wave-presence mismatches).
    weights : dict[str, float]
        Weight actually used for each criterion (after filling defaults
        from ``DEFAULT_WEIGHTS``).
    flags : dict[str, str]
        Free-text diagnostic notes per criterion key, e.g. wave-presence
        mismatches that triggered the conventional NaN penalty, or axis
        ambiguity. Empty dict entries are omitted (only flagged criteria
        appear here).
    """
    total_score: float
    penalty_vector: dict
    weights: dict
    flags: dict


class EcgMetrics:
    """
    Computes and stores in-silico ECG metrics from electrophysiology simulations.

    This class provides on-demand evaluation of standard ECG markers such as 
    QRS duration and QT intervals based on 3D tissue kinematics. It uses lazy loading 
    to ensure heavy computations (like repolarization) are only performed when explicitly requested.
    """
    
    def __init__(self, model, post, activation_times=None, ecg_12lead=None, times=None):
        """
        Initialize the EcgMetrics object.

        Parameters
        ----------
        model : ansys.health.heart.models.HeartModel
            The loaded full-heart model containing anatomical parts.
        post : ansys.health.heart.post.dpf_utils.EPpostprocessor
            The postprocessor object containing the simulation results.
        activation_times : dpf.core.Field, optional
            Pre-computed activation times. If None, they will be extracted 
            only when needed.
        """
        self._model = model
        self._post = post
        self._activation_times = activation_times
        self._ecg_12lead = ecg_12lead
        self._times = times
        self._qrs_waves_prominence_fraction = _DEFAULT_PROMINENCE_FRACTION


        self._repolarization_times = None
        self._qrs_duration = None
        self._qt_interval = None
        self._p_wave_duration = None
        self._pq_interval = None
        self._qrs_peak_amplitudes = None  
        self.qrs_waves = None
        self._r_progression = None
        self._electrical_axis = None
        self._implausibility = None
        self._implausibility_reference = None

    # --- Public Properties (Lazy Loading) ---

    @property
    def activation_times(self):
        """
        dpf.core.Field: The activation times field. 
        Extracted on-the-fly if not provided during initialization.
        """
        if self._activation_times is None:
            self._activation_times = self._post.get_activation_times()
        return self._activation_times
    
    @property
    def repolarization_times(self, method= "apd90"):
        """
        numpy.ndarray: The repolarization times for all nodes (default method is APD90).
        Computed only on the first call.
        """
        if self._repolarization_times is None:
            self._repolarization_times = self._compute_repolarization(method=method)
        return self._repolarization_times

    @property
    def qrs_duration(self):
        """
        float: The computed QRS duration in milliseconds (Ventricular activation time).

        Source for normal values 
        -------
        https://www.ahajournals.org/doi/10.1161/circulationaha.108.191095
        """
        if self._qrs_duration is None:
            self._qrs_duration = self._compute_qrs_duration()
        return self._qrs_duration

    @property
    def qt_interval(self):
        """
        float: The computed QT interval in milliseconds (Ventricular activation to repolarization).
        """
        if self._qt_interval is None:
            self._qt_interval = self._compute_qt_interval()
        return self._qt_interval
    
    @property
    def p_wave_duration(self):
        """
        float: The computed P-wave duration in milliseconds (Atrial activation time).
        """
        if self._p_wave_duration is None:
            self._p_wave_duration = self._compute_p_wave_duration()
        return self._p_wave_duration
    
    @property
    def pq_interval(self):
        """
        float: The computed PQ interval in milliseconds (Beginning of P-wave to beginning of QRS).
        """
        if self._pq_interval is None:
            self._pq_interval = self._compute_pq_interval()
        return self._pq_interval
    
    @property
    def qrs_peak_amplitudes(self) -> dict:
        """
        dict: Peak absolute amplitude of the QRS complex for each lead.
        Computed in a single pass over all 12 leads on first access.
        Keys are lead names (e.g. ``"V1"``), values are floats in arbitrary units.
        """
        if self._qrs_peak_amplitudes is None:
            self._qrs_peak_amplitudes = self._compute_qrs_peak_amplitudes()
        return self._qrs_peak_amplitudes
    
    @property
    def r_progression(self) -> dict:
        """
        dict: R-wave progression metrics for precordial leads V1 to V4.
    
        Computed in a single pass on first access.  Requires
        ``compute_qrs_waves()`` to have been called beforehand.
    
        Returns
        -------
        dict
            Mapping ``lead → RProgressionInfo`` for leads in
            ``_R_PROGRESSION_LEADS`` (``"V1"`` … ``"V4"``).
    
        Raises
        ------
        RuntimeError
            If ``compute_qrs_waves()`` has not been called yet.
    
        See Also
        --------
        is_r_progression_normal : Boolean criterion derived from this dict.
        compute_qrs_waves : Must be called first.
        """
        if self._r_progression is None:
            self._r_progression = self._compute_r_progression()
        return self._r_progression
    
    
    @property
    def is_r_progression_normal(self) -> Optional[bool]:
        """
        bool or None: Whether R-wave progression from V1 to V4 is normal.
    
        Criterion : the R/|S| ratio must be **monotonically non-decreasing**
        from V1 to V4, AND must cross 1.0 at some point in V1–V4 (transition
        zone present).
    
        Returns ``None`` when insufficient data prevents a verdict
        (e.g. more than one lead has ``np.nan`` R/S ratio).
    
        Returns
        -------
        bool or None
            ``True``  — progression normal (non-decreasing R/S, transition present).
            ``False`` — criterion violated.
            ``None``  — undecidable (too many NaN values).
    
        Raises
        ------
        RuntimeError
            If ``compute_qrs_waves()`` has not been called yet.
        
        Source
        ------
        https://gpnotebook.com/pages/cardiovascular-medicine/r-wave-progression
        """
        return self._evaluate_r_progression_criterion()

    @property
    def electrical_axis(self) -> float:
        """
        float: Mean QRS electrical axis in the frontal plane, in degrees.

        Computed on first access using the isoelectric lead method
        (see _compute_electrical_axis documentation) 
        Requires ``compute_qrs_waves()`` to have been called beforehand.

        The normal range is −30° to +90°.  Values outside this range
        indicate axis deviation (left < −30°, right > +90°).

        Returns
        -------
        float
            Axis angle in degrees, in [−180°, +180°].
            ``np.nan`` if ``compute_qrs_waves()`` has not been called or if
            the axis cannot be determined (all leads isoelectric).

        Raises
        ------
        RuntimeError
            If ``compute_qrs_waves()`` has not been called yet.

        See Also
        --------
        compute_qrs_waves : Must be called first.

        Source
        --------
        https://www.ncbi.nlm.nih.gov/books/NBK470532/
        """
        if self._electrical_axis is None:
            self._electrical_axis = self._compute_electrical_axis()
        return self._electrical_axis

    @property
    def implausibility(self) -> "ImplausibilityResult":
        """
        ImplausibilityResult: Cached result of the last ``compute_implausibility()``
        call.

        This is a *lazy cache*, not an independent computation: it does not
        know which reference to compare against on its own. Call
        ``compute_implausibility(reference, ...)`` at least once; subsequent
        reads of this property return the cached result without recomputing,
        until ``compute_implausibility`` is called again (e.g. with a new
        reference, or after ``compute_qrs_waves()`` is re-run with different
        settings).

        Returns
        -------
        ImplausibilityResult

        Raises
        ------
        RuntimeError
            If ``compute_implausibility()`` has not been called yet.
        """
        if self._implausibility is None:
            raise RuntimeError(
                "Implausibility has not been computed yet. "
                "Call compute_implausibility(reference) first."
            )
        return self._implausibility

    # --- Private Helper Methods ---

    def _get_min_activation_time_for_part(self, part):
        part_nodes = part.get_node_ids(self._model.mesh)
        part_act_times = self.activation_times.data[part_nodes]
        return np.nanmin(part_act_times)

    def _get_max_activation_time_for_part(self, part):
        part_nodes = part.get_node_ids(self._model.mesh)
        part_act_times = self.activation_times.data[part_nodes]
        return np.nanmax(part_act_times)

    def _get_min_repolarization_time_for_part(self, part):
        part_nodes = part.get_node_ids(self._model.mesh)
        part_repo_times = self.repolarization_times[part_nodes] 
        return np.nanmin(part_repo_times)

    def _get_max_repolarization_time_for_part(self, part):
        part_nodes = part.get_node_ids(self._model.mesh)
        part_repo_times = self.repolarization_times[part_nodes] 
        return np.nanmax(part_repo_times)

    def _compute_qrs_peak_amplitudes(self) -> dict:
        """
        Compute the peak absolute amplitude of the QRS complex for all 12 leads.

        Computed as ``max(|signal|)`` within the ventricular activation window
        for each lead, in a single pass.

        The calculated amplitudes can then be used to normalize the QRS signal,
        which is a necessary step to compare amplitudes between leads of the simulated signal
        or between simulated and clinical ECGs

        Returns
        -------
        dict
            Mapping of lead name to peak absolute amplitude (float).
            Keys follow ``LEAD_INDEX`` (e.g. ``"I"``, ``"V1"``).
            Values are in arbitrary units (simulation output, not physiologically
            calibrated).

        Raises
        ------
        RuntimeError
            If ``ecg_12lead`` and ``times`` were not provided at initialization.

        Notes
        -----
        The QRS window is derived from nodal activation times:
        ``t_start = min ventricular activation``,
        ``t_end = t_start + qrs_duration``. 
        This assumes a shared time origin between the ECG signal and the
        nodal activation times.

        Leads with a flat signal in the QRS window (peak = 0) are stored as
        ``numpy.nan`` with a warning, rather than raising, to avoid blocking
        the full computation.
        """
        self._check_ecg_signals()

        min_lv = self._get_min_activation_time_for_part(self._model.left_ventricle)
        min_rv = self._get_min_activation_time_for_part(self._model.right_ventricle)
        min_septum = self._get_min_activation_time_for_part(self._model.septum)
        t_start = min(min_lv, min_rv, min_septum)
        t_end = t_start + self.qrs_duration

        mask = (
            (self._times >= t_start - _DEFAULT_MARGIN_MS)
            & (self._times <= t_end + _DEFAULT_MARGIN_MS)
        )

        amplitudes = {}
        for lead, idx in LEAD_INDEX.items():
            signal_win = self._ecg_12lead[idx][mask]
            peak = float(np.max(np.abs(signal_win)))
            if peak == 0.0:
                LOGGER.warning(
                    f"QRS peak amplitude is zero for lead '{lead}': "
                    f"signal is flat in window [{t_start:.1f}, {t_end:.1f}] ms. "
                    f"Normalization for this lead will produce NaN."
                )
                amplitudes[lead] = np.nan
            else:
                amplitudes[lead] = peak

        LOGGER.info(
            f"QRS peak amplitudes computed for all leads over window "
            f"[{t_start:.1f}, {t_end:.1f}] ms (margin={_DEFAULT_MARGIN_MS} ms)."
        )
        return amplitudes
    
    def _check_ecg_signals(self) -> None:
        """Raise RuntimeError if compute_12_lead_ECGs() has not been called."""
        if self._ecg_12lead is None or self._times is None:
            raise RuntimeError(
                "ECG signals are not set. Call compute_12_lead_ECGs(ecg, times) first."
            )
    
    def _find_local_extrema(
        self,
        signal: np.ndarray,
        times: np.ndarray,
        t_start: float,
        t_end: float,
        kind: str = "max",
        margin_ms: float = _DEFAULT_MARGIN_MS,
        prominence_fraction: float = _DEFAULT_PROMINENCE_FRACTION,
        normalization_value: float = None,
    ) -> dict:
        """
        Detect local extrema within a time window on a 1D ECG signal.

        Parameters
        ----------
        signal : numpy.ndarray
            1D array of ECG amplitude values, shape (n_timesteps,).
            Baseline assumed to be 0. Units are arbitrary (simulation output).
        times : numpy.ndarray
            1D array of timestamps in milliseconds, shape (n_timesteps,).
            Must share the same time origin as the nodal activation times.
        t_start : float
            Start of the search window in milliseconds (inclusive).
            Typically derived from nodal activation times.
            A margin can be applied internally.
        t_end : float
            End of the search window in milliseconds (inclusive).
            Typically derived from nodal activation or repolarization times.
            A margin can be applied internally.
        kind : str, optional
            Type of extrema to detect. Either ``"max"`` for local maxima
            or ``"min"`` for local minima. Default is ``"max"``.
        margin_ms : float, optional
            Temporal margin in milliseconds added symmetrically around
            [t_start, t_end]. Default is 0.0 ms.
        prominence_fraction : float, optional
            Minimum prominence as a fraction of the signal range in the window.
            Default is 0.02 (2 %).
        normalization_value : float, optional
            If provided, returned ``"values"`` are divided by this value.
            Typically ``qrs_peak_amplitudes[lead]`` for the relevant lead.
            If ``None``, raw amplitudes are returned. Default is ``None``.

        Returns
        -------
        dict with keys:
            - ``"times"`` : numpy.ndarray of peak timestamps in ms
            - ``"values"`` : numpy.ndarray of peak amplitudes, normalized if
            ``normalization_value`` is provided
            - ``"indices"`` : numpy.ndarray of peak indices in the original signal

        Raises
        ------
        ValueError
            If ``kind`` is not ``"max"`` or ``"min"``.
            If the time window contains fewer than 2 samples.
            If ``normalization_value`` is 0.
        """
        from scipy.signal import find_peaks

        if kind not in ("max", "min"):
            raise ValueError(f"kind must be 'max' or 'min', got '{kind}'.")

        if normalization_value is not None and normalization_value == 0.0:
            raise ValueError("normalization_value must not be zero.")

        mask = (times >= t_start - margin_ms) & (times <= t_end + margin_ms)
        if mask.sum() < 2:
            raise ValueError(
                f"Time window [{t_start - margin_ms:.1f}, {t_end + margin_ms:.1f}] ms "
                f"contains fewer than 2 samples."
            )

        times_win = times[mask]
        signal_win = signal[mask]
        indices_win = np.where(mask)[0]

        search_signal = signal_win if kind == "max" else -signal_win # find_peaks looks for maxima only, changing signal sign to find minima

        signal_range = np.ptp(signal_win)
        prominence = prominence_fraction * signal_range if signal_range > 0 else 0.0

        peak_indices_win, _ = find_peaks(search_signal, prominence=prominence)

        if len(peak_indices_win) == 0:
            LOGGER.info(
                f"No local {kind}ima found in window "
                f"[{t_start:.1f}, {t_end:.1f}] ms (margin={margin_ms} ms)."
            )
            return {
                "times": np.array([]),
                "values": np.array([]),
                "indices": np.array([], dtype=int),
            }

        values = signal_win[peak_indices_win]
        if normalization_value is not None:
            values = values / normalization_value

        LOGGER.info(
            f"Found {len(peak_indices_win)} local {kind}ima in window "
            f"[{t_start:.1f}, {t_end:.1f}] ms"
            + (f", normalized by {normalization_value:.4f}." if normalization_value else ".")
        )

        return {
            "times": times_win[peak_indices_win],
            "values": values,
            "indices": indices_win[peak_indices_win],
        }

    def _find_threshold_crossing(
        self,
        signal: np.ndarray,
        times: np.ndarray,
        t_start: float,
        t_end: float,
        threshold: float = 0.0,
        direction: str = "forward",
        crossing: str = "above",
    ) -> float | None:
        """
        Find the first sample where the signal crosses a threshold within a time window.

        Parameters
        ----------
        signal : numpy.ndarray
            1D array of ECG amplitude values, shape (n_timesteps,).
            Units are arbitrary (raw simulation output, not normalised).
        times : numpy.ndarray
            1D array of timestamps in milliseconds, shape (n_timesteps,).
        t_start : float
            Search window start (ms, exclusive).  Samples at exactly ``t_start``
            are excluded so that the anchor peak itself is never returned.
        t_end : float
            Search window end (ms, inclusive).
        threshold : float, optional
            Amplitude threshold for the crossing.  Default is ``0.0`` (zero-crossing).
        direction : str, optional
            ``"forward"``  — return the **first** crossing from ``t_start`` toward
            ``t_end`` (chronological order).
            ``"backward"`` — return the **last** crossing scanning from ``t_end``
            back toward ``t_start`` (reverse chronological order, useful for
            wave-onset detection).
            Default is ``"forward"``.
        crossing : str, optional
            ``"above"`` — crossing detected when ``signal >= threshold``
            (signal rises above or equals the threshold).
            ``"below"`` — crossing detected when ``signal <= threshold``
            (signal falls below or equals the threshold).
            Default is ``"above"``.

        Returns
        -------
        float or None
            Timestamp (ms) of the first qualifying sample, or ``None`` if no
            crossing is found in the window.

        Raises
        ------
        ValueError
            If ``direction`` is not ``"forward"`` or ``"backward"``.
            If ``crossing`` is not ``"above"`` or ``"below"``.

        Examples
        --------
        Zero-crossing after the R peak :

        >>> t = self._find_threshold_crossing(
        ...     signal, times,
        ...     t_start=t_r, t_end=times[-1],
        ...     threshold=0.0, direction="forward", crossing="above",
        ... )

        Q-wave onset (first sample dropping below 2 % of the Q peak amplitude):

        >>> threshold_q = _Q_ONSET_THRESHOLD_FRACTION * q_peak_amplitude_raw  # negative value
        >>> t = self._find_threshold_crossing(
        ...     signal, times,
        ...     t_start=t_qrs_start, t_end=t_q,
        ...     threshold=threshold_q, direction="forward", crossing="below",
        ... )
        """
        if direction not in ("forward", "backward"):
            raise ValueError(
                f"direction must be 'forward' or 'backward', got '{direction}'."
            )
        if crossing not in ("above", "below"):
            raise ValueError(
                f"crossing must be 'above' or 'below', got '{crossing}'."
            )

        mask = (times > t_start) & (times <= t_end)
        if not mask.any():
            return None

        indices = np.where(mask)[0]

        if direction == "backward":
            indices = indices[::-1]

        signal_win = signal[indices]

        if crossing == "above":
            hit = np.where(signal_win >= threshold)[0]
        else:
            hit = np.where(signal_win <= threshold)[0]

        if len(hit) == 0:
            return None

        return float(times[indices[hit[0]]])


    def _build_wave_dict(
        self,
        anchor_time: float,
        anchor_value: float,
        extra_peaks: dict,
        wave_duration: float = np.nan,
        wave_kind: str = "max",
    ) -> WaveInfo:
        """
        Build a WaveInfo from an anchor peak and extra peaks, including notch depths.

        Parameters
        ----------
        anchor_time : float
            Timestamp of the primary wave peak (ms).
        anchor_value : float
            Normalized amplitude of the primary wave peak.
        extra_peaks : dict
            Result of ``_find_local_extrema`` for additional peaks of the same
            polarity after the anchor. Must contain keys ``"times"`` and ``"values"``.
        wave_duration : float, optional
            Duration of the wave in milliseconds. ``np.nan`` if not calculable.
            Default is ``np.nan``.
        wave_kind : str, optional
            Polarity of the wave peaks: ``"max"`` for R (positive peaks),
            ``"min"`` for Q and S (negative peaks). Used to determine the
            polarity of the notch extremum searched between consecutive peaks.
            Default is ``"max"``.

        Returns
        -------
        WaveInfo
            Dataclass with fields ``time``, ``value``, ``n_peaks``,
            ``peak_times``, ``peak_values``, ``peak_intervals``,
            ``notch_depths``, ``wave_duration``.

        Notes
        -----
        Notch depth between ``peak_times[i]`` and ``peak_times[i+1]`` :

        - The extremum of **opposite polarity** is searched in that interval
          using ``_NOTCH_PROMINENCE_FRACTION`` (very low, to detect shallow notches).
        - If multiple extrema are found, the one **most distant from zero**
          (largest absolute value) is selected to maximise the reported depth.
        - Depth = ``abs(peak_value[i] - notch_extremum) / abs(peak_value[i])``.
        - 0.0 if no extremum is detected between the two peaks.
        """
        peak_times  = np.concatenate(([anchor_time], extra_peaks["times"]))
        peak_values = np.concatenate(([anchor_value], extra_peaks["values"]))
        n_peaks     = len(peak_times)
        peak_intervals = np.diff(peak_times) if n_peaks > 1 else np.array([])

        # --- Notch depths ---
        if n_peaks <= 1:
            notch_depths = np.array([])
        else:
            notch_kind   = "min" if wave_kind == "max" else "max"
            notch_depths = np.zeros(n_peaks - 1)

            for i in range(n_peaks - 1):
                t_left  = peak_times[i]
                t_right = peak_times[i + 1]
                ref_value = peak_values[i]  # anchor for depth calculation 

                extrema = self._find_local_extrema(
                    # signal and times are accessed via self._ecg_12lead/self._times
                    # but _build_wave_dict does not carry them — they must be passed
                    # via the signal/times parameters added below
                    self._notch_signal,
                    self._notch_times,
                    t_start=t_left,
                    t_end=t_right,
                    kind=notch_kind,
                    margin_ms=0.0,
                    prominence_fraction=_NOTCH_PROMINENCE_FRACTION,
                    normalization_value=self._notch_norm,
                )

                if len(extrema["values"]) == 0:
                    notch_depths[i] = 0.0
                else:
                    # Select extremum most distant from zero (largest abs value)
                    ########## Note : problem with this definition, 
                    ########## the value of the extremum used for the notch depth should be the closest to zero
                    ########## When the implausibility score is computed, notches contributions are too high
                    ########## and do not represent accurately how bad teh notch really is
                    ########## Notches definitions and penalties should be reconsidered
                    most_distant_idx = np.argmax(np.abs(extrema["values"]))
                    notch_val = extrema["values"][most_distant_idx]
                    notch_depths[i] = abs(ref_value - notch_val) / abs(ref_value) 
                    # The reference value here is only considering the value of the first peak 
                    

        return WaveInfo(
            time=float(anchor_time),
            value=float(anchor_value),
            n_peaks=n_peaks,
            peak_times=peak_times,
            peak_values=peak_values,
            peak_intervals=peak_intervals,
            notch_depths=notch_depths,
            wave_duration=float(wave_duration),
        )


    def _identify_qrs_waves(self, lead: str, prominence_fraction: float = _DEFAULT_PROMINENCE_FRACTION) -> QrsWaves:
        """
        Identify the Q, R, and S waves of the QRS complex for a given lead,
        including peak counts, timestamps, intervals, durations, and late notches.

        Parameters
        ----------
        lead : str
            Lead name, e.g. ``"I"``, ``"V1"``. Must be a key of ``LEAD_INDEX``.
        prominence_fraction : float, optional
            Minimum prominence of a peak, expressed as a fraction of the signal
            range within the detection window. Passed to all internal calls to
            ``_find_local_extrema``. Lower values increase sensitivity to small
            deflections (useful for Q and S waves); higher values reduce
            sensitivity to noise. Default is ``_DEFAULT_PROMINENCE_FRACTION``
            (0.02, i.e. 2%).

        Returns
        -------
        QrsWaves
            Dataclass with fields ``Q``, ``R``, ``S``, ``extra_extrema``.

            Each of ``Q``, ``R``, ``S`` is a ``WaveInfo`` dataclass with fields:

            - ``time`` : float, timestamp of the primary peak (ms),
            ``np.nan`` if wave is absent
            - ``value`` : float, normalized amplitude,
            ``np.nan`` if wave is absent
            - ``n_peaks`` : int, total number of peaks for this wave,
            0 if wave is absent
            - ``peak_times`` : numpy.ndarray, timestamps of all peaks (ms),
            empty array if wave is absent
            - ``peak_intervals`` : numpy.ndarray, intervals between consecutive
            peaks (ms), shape (n_peaks - 1,), empty if n_peaks <= 1
            or wave is absent
            - ``wave_duration`` : float, duration of the wave (ms),
            ``np.nan`` if wave is absent or duration not calculable

            Wave durations are defined as:

            - **Q** : zero-crossing post-Q minus Q-wave onset, where onset is
            the first sample in the QRS window where the signal drops below
            ``_Q_ONSET_THRESHOLD_FRACTION`` (2 %) of the raw Q peak amplitude.
            Both boundaries are derived from the 1D ECG signal.
            - **R** : zero-crossing post-R minus zero-crossing post-Q
            (or min ventricular activation time if Q absent)
            - **S** : max ventricular activation time minus zero-crossing post-R

            ``extra_extrema`` is always present and contains:

            - ``"maxima"`` : dict with ``"times"`` and ``"values"``
            - ``"minima"`` : dict with ``"times"`` and ``"values"``

            These are extrema detected after the zero-crossing post-S within
            the QRS window, for late notch detection.

        Raises
        ------
        RuntimeError
            If ``ecg_12lead`` and ``times`` were not provided at initialization.
        ValueError
            If ``lead`` is not a valid lead name.

        Notes
        -----
        Wave definitions:

        - **Q** : first negative local minimum before any local maximum.
        Additional Q peaks are negative minima between Q and R.
        - **R** : first positive local maximum in the QRS window.
        Additional R peaks are positive maxima between R and S.
        - **S** : first negative local minimum strictly after R.
        Additional S peaks are negative minima between S and the first
        zero-crossing post-S (or ``t_qrs_end`` if no crossing found).
        - **extra_extrema** : all extrema after the zero-crossing post-S,
        within the QRS window.

        All amplitudes are normalized by ``qrs_peak_amplitudes[lead]``.

        The ``prominence_fraction`` parameter is the main tuning knob for this
        method. If expected waves are not detected, lowering this value
        (e.g. 0.05) increases sensitivity. If spurious peaks appear, raising it
        (e.g. 0.2) filters them out.
        """
        self._check_ecg_signals()

        if lead not in LEAD_INDEX:
            raise ValueError(f"Unknown lead '{lead}'. Valid leads: {list(LEAD_INDEX.keys())}")

        signal = self._ecg_12lead[LEAD_INDEX[lead]]
        norm = self.qrs_peak_amplitudes[lead]
        norm_value = norm if not np.isnan(norm) else None

        # Temporary attributes used by _build_wave_dict to compute notch depths.
        # Stored on self to avoid threading the signal through every call signature.
        # Cleared at the end of this method.
        self._notch_signal = signal
        self._notch_times  = self._times
        self._notch_norm   = norm_value

        # QRS window and ventricular activation bounds
        min_lv     = self._get_min_activation_time_for_part(self._model.left_ventricle)
        min_rv     = self._get_min_activation_time_for_part(self._model.right_ventricle)
        min_septum = self._get_min_activation_time_for_part(self._model.septum)
        t_min_ventricles = min(min_lv, min_rv, min_septum)
        t_max_ventricles = max(
            self._get_max_activation_time_for_part(self._model.left_ventricle),
            self._get_max_activation_time_for_part(self._model.right_ventricle),
            self._get_max_activation_time_for_part(self._model.septum),
        )
        t_start = t_min_ventricles
        t_end   = t_start + self.qrs_duration

        # Detect all extrema in QRS window
        maxima = self._find_local_extrema(
            signal, self._times, t_start, t_end,
            kind="max", normalization_value=norm_value,
            prominence_fraction=prominence_fraction,
        )
        minima = self._find_local_extrema(
            signal, self._times, t_start, t_end,
            kind="min", normalization_value=norm_value,
            prominence_fraction=prominence_fraction,
        )

        result = {
            "Q": _absent_wave(),
            "R": _absent_wave(),
            "S": _absent_wave(),
            "extra_extrema": {
                "maxima": {"times": np.array([]), "values": np.array([])},
                "minima": {"times": np.array([]), "values": np.array([])},
            },
        }

        # --- R peak : first positive local maximum ---
        positive_mask = maxima["values"] > 0
        t_r = None
        r_value = None
        if positive_mask.any():
            first_pos_idx = np.argmin(maxima["times"][positive_mask])
            t_r     = maxima["times"][positive_mask][first_pos_idx]
            r_value = maxima["values"][positive_mask][first_pos_idx]
            # LOGGER.info(f"[{lead}] R wave detected at t={t_r:.1f} ms (norm. value={r_value:.3f}).")
        # else:
            # LOGGER.info(f"[{lead}] No positive maximum found — R wave absent.")

        # --- S peak : first local minimum strictly after R ---
        t_s = None
        s_value = None
        if t_r is not None:
            negative_mask = minima["values"] < 0
            after_r_mask  = negative_mask & (minima["times"] > t_r)
            if after_r_mask.any():
                first_s_idx = np.argmin(minima["times"][after_r_mask])
                t_s     = minima["times"][after_r_mask][first_s_idx]
                s_value = minima["values"][after_r_mask][first_s_idx]
            #     LOGGER.info(f"[{lead}] S wave detected at t={t_s:.1f} ms (norm. value={s_value:.3f}).")
            # else:
            #     LOGGER.info(f"[{lead}] No negative minimum after R — S wave absent.")

        # --- Q peak : first local minimum before any local maximum ---
        t_q = None
        q_value = None
        negative_mask = minima["values"] < 0
        t_first_max   = maxima["times"][0] if len(maxima["times"]) > 0 else np.inf
        before_r_mask = negative_mask & (minima["times"] < t_first_max)
        if before_r_mask.any():
            first_q_idx = np.argmin(minima["times"][before_r_mask])
            t_q     = minima["times"][before_r_mask][first_q_idx]
            q_value = minima["values"][before_r_mask][first_q_idx]
        #     LOGGER.info(f"[{lead}] Q wave detected at t={t_q:.1f} ms (norm. value={q_value:.3f}).")
        # else:
        #     LOGGER.info(f"[{lead}] No negative minimum before first maximum — Q wave absent.")

        # --- Extra Peaks in the Q wave : negative minima between Q and R ---
        if t_q is not None:
            # Zero-crossing post-Q : first sample >= 0 after the Q peak
            t_zero_post_q = self._find_threshold_crossing(
                signal, self._times,
                t_start=t_q, t_end=self._times[-1],
                threshold=0.0, direction="forward", crossing="above",
            )
            # Q-wave onset : first sample in the QRS window dropping below
            # _Q_ONSET_THRESHOLD_FRACTION * raw Q peak amplitude.
            # q_value is normalised; multiply back by norm to get the raw threshold.
            q_peak_amplitude_raw = q_value * norm if norm_value is not None else q_value
            threshold_q_onset = _Q_ONSET_THRESHOLD_FRACTION * q_peak_amplitude_raw  # negative
            t_onset_q = self._find_threshold_crossing(
                signal, self._times,
                t_start=t_start, t_end=t_q,
                threshold=threshold_q_onset, direction="forward", crossing="below",
            )
            if t_onset_q is None:
                LOGGER.warning(
                    f"[{lead}] Q-wave onset not found "
                    f"(threshold={threshold_q_onset:.4e}); "
                    f"Q wave duration set to NaN."
                )
            duration_q = (
                t_zero_post_q - t_onset_q
                if (t_zero_post_q is not None and t_onset_q is not None)
                else np.nan
            )
            if t_r is not None:
                extra_q = self._find_local_extrema(
                    signal, self._times, t_q, t_r,
                    kind="min", normalization_value=norm_value,
                    margin_ms=0.0,
                )
                extra_q_mask = extra_q["times"] > t_q
                extra_q_filtered = {
                    "times":   extra_q["times"][extra_q_mask],
                    "values":  extra_q["values"][extra_q_mask],
                    "indices": extra_q["indices"][extra_q_mask],
                }
            else:
                extra_q_filtered = {"times": np.array([]), "values": np.array([])}

            result["Q"] = self._build_wave_dict(
                t_q, q_value, extra_q_filtered, wave_duration=duration_q, wave_kind="min"
            )

        # --- Extra R peaks : positive maxima between R and S ---
        if t_r is not None:
            t_zero_post_q = (
                self._find_threshold_crossing(
                    signal, self._times,
                    t_start=t_q, t_end=self._times[-1],
                    threshold=0.0, direction="forward", crossing="above",
                )
                if t_q is not None else None
            )
            t_r_left = t_zero_post_q if t_zero_post_q is not None else t_min_ventricles
            t_zero_post_r = self._find_threshold_crossing(
                signal, self._times,
                t_start=t_r, t_end=self._times[-1],
                threshold=0.0, direction="forward", crossing="above",
            )
            duration_r = (
                t_zero_post_r - t_r_left
                if t_zero_post_r is not None else np.nan
            )
            t_r_end = t_s if t_s is not None else t_end
            extra_r = self._find_local_extrema(
                signal, self._times, t_r, t_r_end,
                kind="max", normalization_value=norm_value,
                margin_ms=0.0,
            )
            extra_r_mask = extra_r["times"] > t_r
            extra_r_filtered = {
                "times":   extra_r["times"][extra_r_mask],
                "values":  extra_r["values"][extra_r_mask],
                "indices": extra_r["indices"][extra_r_mask],
            }
            result["R"] = self._build_wave_dict(
                t_r, r_value, extra_r_filtered, wave_duration=duration_r, wave_kind="max"
            )

        # --- Extra S peaks : negative minima between S and zero-crossing post-S---
        if t_s is not None:
            t_zero_post_r = self._find_threshold_crossing(
                signal, self._times,
                t_start=t_r, t_end=self._times[-1],
                threshold=0.0, direction="forward", crossing="above",
            )
            t_zero_post_s = self._find_threshold_crossing(
                signal, self._times,
                t_start=t_s, t_end=self._times[-1],
                threshold=0.0, direction="forward", crossing="above",
            )
            t_s_end = t_zero_post_s if t_zero_post_s is not None else t_end
            duration_s = (
                t_max_ventricles - t_zero_post_r
                if t_zero_post_r is not None else np.nan
            )
            extra_s = self._find_local_extrema(
                signal, self._times, t_s, t_s_end,
                kind="min", normalization_value=norm_value,
                margin_ms=0.0,
            )
            extra_s_mask = extra_s["times"] > t_s
            extra_s_filtered = {
                "times":   extra_s["times"][extra_s_mask],
                "values":  extra_s["values"][extra_s_mask],
                "indices": extra_s["indices"][extra_s_mask],
            }
            result["S"] = self._build_wave_dict(
                t_s, s_value, extra_s_filtered, wave_duration=duration_s, wave_kind="min"
            )

            # --- Extra extrema : after the zero-crossing post-S, in the QRS window---
            if t_zero_post_s is not None and t_zero_post_s < t_end:
                extra_max = self._find_local_extrema(
                    signal, self._times, t_zero_post_s, t_end,
                    kind="max", normalization_value=norm_value,
                    margin_ms=0.0,
                )
                extra_min = self._find_local_extrema(
                    signal, self._times, t_zero_post_s, t_end,
                    kind="min", normalization_value=norm_value,
                    margin_ms=0.0,
                )
                result["extra_extrema"] = {
                    "maxima": {"times": extra_max["times"], "values": extra_max["values"]},
                    "minima": {"times": extra_min["times"], "values": extra_min["values"]},
                }
                if len(extra_max["times"]) + len(extra_min["times"]) > 0:
                    LOGGER.info(
                        f"[{lead}] Late notches detected: "
                        f"{len(extra_max['times'])} maxima, "
                        f"{len(extra_min['times'])} minima "
                        f"after zero-crossing at t={t_zero_post_s:.1f} ms."
                    )

        # Clean up temporary attributes used by _build_wave_dict
        del self._notch_signal, self._notch_times, self._notch_norm

        return QrsWaves(
            Q=result["Q"],
            R=result["R"],
            S=result["S"],
            extra_extrema=result["extra_extrema"],
        )

    def _get_av_interface_node_ids(self) -> np.ndarray:
        """
        Retrieve node IDs at the atrioventricular interface.

        These are nodes shared between ventricular and atrial elements,
        responsible for the known P-wave duration overestimation.

        Returns
        -------
        numpy.ndarray
            Array of node IDs at the AV interface.

        Notes
        -----
        This method reproduces the interface node detection logic from
        ``_create_atrioventricular_isolation``, without requiring the
        atrioventricular isolation part to be accessible as a named part
        in the model.
        """
        mesh = self._model.mesh

        v_ele = np.concatenate([
            self._model.left_ventricle.get_element_ids(mesh),
            self._model.right_ventricle.get_element_ids(mesh),
            self._model.septum.get_element_ids(mesh),
        ])
        a_ele = np.concatenate([
            self._model.left_atrium.get_element_ids(mesh),
            self._model.right_atrium.get_element_ids(mesh),
        ])

        ventricles = mesh.extract_cells(v_ele)
        atrial     = mesh.extract_cells(a_ele)

        interface_nids = np.intersect1d(
            ventricles["_global-point-ids"],
            atrial["_global-point-ids"],
        )

        # LOGGER.info(f"AV interface: {len(interface_nids)} nodes identified.")
        return interface_nids


    # --- Private Core Computation Methods ---

    def _compute_repolarization(self, method="apd90", threshold=-85.0):
        """
        Calculates the repolarization times for all nodes based on clinical definitions.

        Parameters
        ----------
        activation_times : dpf.core.Field
            A Field object containing the activation time for each node.
        post : EPpostprocessor
            The postprocessor object used to retrieve potentials and time arrays.
        method : str, optional
            The definition of repolarization to use. Options are:
            - "apd90" : 90% of repolarization (standard).
            - "apd50" : 50% of repolarization (often used for drug testing).
            - "absolute" : Hard voltage threshold (uses the 'threshold' parameter).
        threshold : float, optional
            The fixed voltage threshold (in mV) used ONLY if method="absolute".

        Returns
        -------
        numpy.ndarray
            A 1D array containing the repolarization time for each node.

        Notes
        -----
        The temporal precision of the calculated repolarization times is strictly 
        limited by the output frequency of the simulation's d3plot files. For 
        example, if the d3plot states are written every 10 ms, the repolarization 
        time is discretized to that exact 10 ms grid. To achieve sub-interval 
        precision, either a higher output frequency is required during the simulation, 
        or linear interpolation must be applied post-simulation. 
        Precision could also be anhanced by using EP_spline_TraMP input files
        instead of calling the get_transmembrane_potential method that uses dyna outputs
        """
        LOGGER.info(f"Starting repolarization computation using method: {method.upper()}")
        
        all_internal_ids = np.arange(len(self.activation_times.data))
        transmembrane_pot, times = self._post.get_transmembrane_potential(all_internal_ids)
        
        dt = times[1] - times[0] if len(times) > 1 else 0
        # LOGGER.info(f"Detected temporal resolution (dt): {dt:.2f} ms.")

        # Time mask: True if time is strictly after the node's activation time
        act_data = self.activation_times.data
        after_activation_mask = times[:, np.newaxis] > act_data
        
        # Calculate target voltages dynamically based on the chosen method
        if method in ["apd50", "apd90"]:
            fraction = 0.90 if method == "apd90" else 0.50
            
            # Mask potentials BEFORE activation with a massive negative number 
            # so they are ignored when searching for the peak (V_max)
            pot_after_act = np.where(after_activation_mask, transmembrane_pot, -1000)
            
            # Extract V_max and V_rest for each node (axis=0 means across time)
            v_max = np.max(pot_after_act, axis=0)
            v_rest = np.min(transmembrane_pot, axis=0) 
            
            # Calculate target voltage vector (shape: N_nodes,)
            amplitude = v_max - v_rest
            target_voltages = v_max - (fraction * amplitude)
            
        elif method == "absolute":
            target_voltages = threshold
        else:
            raise ValueError("Invalid method. Use 'apd90', 'apd50', or 'absolute'.")

        # Voltage mask: True if potential is below the dynamic target
        repolarized_mask = transmembrane_pot <= target_voltages
        
        # Combined mask
        valid_repolarization = after_activation_mask & repolarized_mask
        
        # Extract times
        rep_indices = np.argmax(valid_repolarization, axis=0)
        found = valid_repolarization.any(axis=0)
        repolarization_times = np.where(found, times[rep_indices], np.nan)
        
        LOGGER.info("Repolarization computation successful.")
        return repolarization_times

    def _compute_qrs_duration(self):
        """Calculates QRS duration restricted to ventricular tissue."""
        lv_nodes = self._model.left_ventricle.get_node_ids(self._model.mesh)
        rv_nodes = self._model.right_ventricle.get_node_ids(self._model.mesh)
        septum_nodes = self._model.septum.get_node_ids(self._model.mesh)

        ventricle_nodes = np.unique(np.concatenate((lv_nodes, rv_nodes, septum_nodes)))
        ventricular_activation_times = self.activation_times.data[ventricle_nodes]
        
        # Max - Min of ventricular activation
        return np.nanmax(ventricular_activation_times) - np.nanmin(ventricular_activation_times)

    def _compute_qt_interval(self):
        """Calculates QT interval restricted to ventricular tissue."""
        # 1. Beginning of QRS (Min activation of ventricles)
        min_lv = self._get_min_activation_time_for_part(self._model.left_ventricle)
        min_rv = self._get_min_activation_time_for_part(self._model.right_ventricle)
        min_septum = self._get_min_activation_time_for_part(self._model.septum)
        
        min_ventricles = min(min_lv, min_rv, min_septum)

        # 2. End of T-wave (Max repolarization of ventricles)
        max_repo_lv = self._get_max_repolarization_time_for_part(self._model.left_ventricle)
        max_repo_rv = self._get_max_repolarization_time_for_part(self._model.right_ventricle)
        max_repo_septum = self._get_max_repolarization_time_for_part(self._model.septum)
        
        max_repo_ventricles = max(max_repo_lv, max_repo_septum, max_repo_rv)

        # 3. QT Interval
        return max_repo_ventricles - min_ventricles
    
    def _compute_p_wave_duration(self) -> float:
        """
        Calculate P-wave duration restricted to atrial tissue, excluding
        atrioventricular interface nodes.

        Returns
        -------
        float
            P-wave duration in milliseconds.

        Notes
        -----
        The AV interface nodes are excluded by computing the intersection of
        ventricular and atrial element nodes, following the same logic as
        ``_create_atrioventricular_isolation``.
        """
        av_interface_nids = self._get_av_interface_node_ids()

        durations = []
        for atrium in [self._model.left_atrium, self._model.right_atrium]:
            atrium_nids = atrium.get_node_ids(self._model.mesh)

            # Exclude AV interface nodes
            clean_nids = np.setdiff1d(atrium_nids, av_interface_nids)

            if len(clean_nids) == 0:
                LOGGER.warning(
                    f"No atrial nodes remaining after AV interface filtering "
                    f"for part '{atrium.name}'. P-wave duration may be unreliable."
                )
                continue

            act_times = self.activation_times.data[clean_nids]
            durations.append((np.nanmin(act_times), np.nanmax(act_times)))

        if len(durations) == 0:
            raise RuntimeError("P-wave duration could not be computed: no valid atrial nodes.")

        min_atria = min(t[0] for t in durations)
        max_atria = max(t[1] for t in durations)

        LOGGER.info(
            f"P-wave duration computed: {max_atria - min_atria:.2f} ms "
            f"({len(av_interface_nids)} AV interface nodes excluded)."
        )
        return max_atria - min_atria

    def _compute_pq_interval(self):
        """Calculates PQ interval."""
        # 1. Beginning of QRS
        min_lv = self._get_min_activation_time_for_part(self._model.left_ventricle)
        min_rv = self._get_min_activation_time_for_part(self._model.right_ventricle)
        min_septum = self._get_min_activation_time_for_part(self._model.septum)
        
        min_ventricles = min(min_lv, min_rv, min_septum)
        
        # 2. Beginning of P-wave
        min_atria = min(
            self._get_min_activation_time_for_part(self._model.left_atrium), 
            self._get_min_activation_time_for_part(self._model.right_atrium)
        )
        
        return min_ventricles - min_atria
    
    def compute_qrs_waves(self, prominence_fraction: float = _DEFAULT_PROMINENCE_FRACTION) -> None:
        """
        Compute and store QRS wave information for all 12 leads.

        Results are stored in ``self.qrs_waves`` as a dictionary mapping
        lead names to ``QrsWaves`` dataclass instances. Any previous result
        is overwritten.

        Parameters
        ----------
        prominence_fraction : float, optional
            Minimum prominence of a peak, expressed as a fraction of the signal
            range within the detection window. Passed to all internal calls to
            ``_find_local_extrema``. Lower values increase sensitivity to small
            deflections; higher values reduce sensitivity to noise.
            Default is ``_DEFAULT_PROMINENCE_FRACTION`` (0.1).

        Raises
        ------
        RuntimeError
            If ``ecg_12lead`` and ``times`` were not provided at initialization.

        Examples
        --------
        >>> EcgMetrics.compute_qrs_waves()
        >>> EcgMetrics.qrs_waves["I"].R.time
        >>> EcgMetrics.compute_qrs_waves(prominence_fraction=0.05)
        >>> EcgMetrics.qrs_waves["V1"].Q.peak_intervals
        """
        self._check_ecg_signals()

        self._qrs_waves_prominence_fraction = prominence_fraction
        self.qrs_waves = {}
        self._r_progression = None
        self._electrical_axis = None

        for lead in LEAD_INDEX:
            LOGGER.info(f"Computing QRS waves for lead '{lead}'.")
            self.qrs_waves[lead] = self._identify_qrs_waves(
                lead, prominence_fraction=prominence_fraction
            )

        LOGGER.info(
            f"QRS waves computed for all leads "
            f"(prominence_fraction={prominence_fraction})."
        )


    def _compute_electrical_axis(self) -> float:
        """
        Compute the mean QRS electrical axis in the frontal plane using the
        isoelectric lead method.

        Parameters
        ----------
        None — reads ``self.qrs_waves`` (must be computed beforehand).

        Returns
        -------
        float
            Axis angle in degrees, in [−180°, +180°].
            ``np.nan`` if ``compute_qrs_waves()`` has not been called or if
            the axis cannot be determined.

        Raises
        ------
        RuntimeError
            If ``compute_qrs_waves()`` has not been called yet.

        Notes
        -----
        Algorithm (isoelectric lead method):

        1. For each of the six frontal leads, compute the net QRS amplitude as
           ``max(R.peak_values) + min(Q.peak_values) + min(S.peak_values)``,
           using 0.0 for absent waves.  Values are normalised by
           ``qrs_peak_amplitudes[lead]`` (intra-lead normalisation).
        2. The isoelectric lead is the one with the smallest ``|net_amplitude|``.
        3. The QRS axis is perpendicular to the isoelectric lead.  Two
           candidates exist (±90° from the isoelectric lead angle).
        4. The correct candidate is the one whose associated frontal lead has a
           positive net amplitude — only the sign is used, not the magnitude,
           so cross-lead amplitude comparisons are avoided.
        5. If both perpendicular leads have the same sign (or both are zero),
           the axis is reported as the average of the two candidate angles.
        
        Using normalized amplitudes could lead to errors in the identification 
        of the isoelectric lead. The electrical axis shouldn't be used as a strong 
        validation criterion.

        References
        ----------
        https://www.ncbi.nlm.nih.gov/books/NBK470532/

        """
        if self.qrs_waves is None:
            raise RuntimeError(
                "QRS waves have not been computed yet. "
                "Call compute_qrs_waves() first."
            )

        frontal_leads = list(_HEXAXIAL.keys())  # I, II, III, aVR, aVL, aVF

        # --- Step 1 : net normalised amplitude per frontal lead ---
        net_amplitudes = {}
        for lead in frontal_leads:
            waves = self.qrs_waves[lead]
            r_net = float(np.max(waves.R.peak_values)) if waves.R.n_peaks > 0 else 0.0
            q_net = float(np.min(waves.Q.peak_values)) if waves.Q.n_peaks > 0 else 0.0
            s_net = float(np.min(waves.S.peak_values)) if waves.S.n_peaks > 0 else 0.0
            net_amplitudes[lead] = r_net + q_net + s_net

        # --- Step 2 : isoelectric lead ---
        iso_lead = min(frontal_leads, key=lambda l: abs(net_amplitudes[l]))
        LOGGER.info(
            f"Isoelectric lead: {iso_lead} "
            f"(net={net_amplitudes[iso_lead]:.4f})"
        )

        # --- Step 3 : two perpendicular candidate lead groups ---
        perp_pos_leads = _HEXAXIAL[iso_lead]["perp_pos"]
        perp_neg_leads = _HEXAXIAL[iso_lead]["perp_neg"]
        iso_angle = _HEXAXIAL[iso_lead]["angle"]
        candidate_angles = [iso_angle + 90, iso_angle - 90]
        # Normalize angles to [−180, +180]
        candidate_angles = [
            ((a + 180) % 360) - 180 for a in candidate_angles
        ]

        # --- Step 4 : resolve quadrant via summed net amplitude of perp lead group ---
        # For tied cases (multiple leads equidistant from the perpendicular),
        # the net amplitudes are summed. Only the sign of the sum is used —
        # no cross-lead amplitude comparison is implied.
        sum_pos = sum(net_amplitudes[l] for l in perp_pos_leads)
        sum_neg = sum(net_amplitudes[l] for l in perp_neg_leads)

        LOGGER.info(
            f"Perp+90° group {perp_pos_leads} → sum={sum_pos:.4f} | "
            f"Perp-90° group {perp_neg_leads} → sum={sum_neg:.4f}"
        )

        if sum_pos > 0 and sum_neg <= 0:
            axis = candidate_angles[0]
        elif sum_neg > 0 and sum_pos <= 0:
            axis = candidate_angles[1]
        else:
            # Both groups have the same sign or both are zero — truly ambiguous.
            axis = np.nan
            LOGGER.warning(
                f"Electrical axis ambiguous: perp+90° group {perp_pos_leads} "
                f"(sum={sum_pos:.4f}) and perp-90° group {perp_neg_leads} "
                f"(sum={sum_neg:.4f}) have the same sign or are both zero. "
                f"Axis set to NaN - flag this case explicitly in the cost function."
            )

        if not np.isnan(axis):
            LOGGER.info(f"Electrical axis = {axis:.1f}°.")
        return float(axis)

    def _compute_r_progression(self) -> dict:
        """
        Compute R-wave progression metrics for leads V1 to V4.

        For each lead, extracts the normalized R and S amplitudes from
        ``self.qrs_waves`` and computes their ratio R / |S|.  A single
        monotonicity penalty is then derived from the full V1→V4 sequence
        and stored in every ``RProgressionInfo`` instance.

        Amplitudes stored in ``WaveInfo.value`` are already normalized by
        ``qrs_peak_amplitudes[lead]``, so they are directly comparable within
        a lead but **not** across leads (different normalization denominators).
        The R/|S| ratio is intra-lead and is therefore cross-lead comparable.

        Parameters
        ----------
        None — reads ``self.qrs_waves`` (must be computed beforehand).

        Returns
        -------
        dict
            Mapping ``lead → RProgressionInfo`` for leads in
            ``_R_PROGRESSION_LEADS``.

        Raises
        ------
        RuntimeError
            If ``compute_qrs_waves()`` has not been called yet
            (``self.qrs_waves`` is ``None``).

        Notes
        -----
        - R amplitude  : ``WaveInfo.value`` of the primary R peak (normalized,
          expected > 0).
        - S amplitude  : ``WaveInfo.value`` of the primary S peak (normalized,
          expected ≤ 0).
        - R/|S| ratio conventions:
          - ``r_amplitude / abs(s_amplitude)`` when both waves are present.
          - ``0.0`` when R is absent (no positive deflection detected).
          - ``np.nan`` when S is absent or |S| < 1e-6 (indecidable; pair
            skipped in monotonicity penalty computation).
        - Monotonicity penalty : sum of R/S decreases between consecutive valid
          lead pairs. Pairs where either ratio is ``np.nan`` are skipped
          (conservative: an absent S does not trigger a penalty).
          ``np.nan`` when fewer than one valid pair is available.

        Possible changes
        -----
        - R amplitude -> max amplitude of the R peaks
        - S amplitude -> min amplitude of the S peaks 
        - S absent or S.value close to 0 currently sets R/S ratio to NaN and both pairs including NaN are excluded from the monotonicity calculation
        """
        if self.qrs_waves is None:
            raise RuntimeError(
                "QRS waves have not been computed yet. "
                "Call compute_qrs_waves() first."
            )

        result = {}

        # --- Pass 1 : per-lead R/S ratios ---
        for lead in _R_PROGRESSION_LEADS:
            waves = self.qrs_waves[lead]

            r_present = waves.R.n_peaks > 0
            s_present = waves.S.n_peaks > 0

            r_amplitude = waves.R.value if r_present else np.nan
            s_amplitude = waves.S.value if s_present else np.nan  # ≤ 0 expected

            # R / |S| ratio — intra-lead, dimensionless.
            # R absent → 0.0 (no positive deflection, worst-case for progression).
            # S absent or |S| near zero → np.nan (indecidable, excluded from penalty).
            if not r_present:
                r_over_s = 0.0
                LOGGER.info(f"[{lead}] R wave absent — R/S set to 0.0.")
            elif not s_present:
                r_over_s = np.nan
                LOGGER.info(f"[{lead}] S wave absent — R/S set to NaN (excluded from penalty).")
            else:
                abs_s = abs(s_amplitude)
                if abs_s < 1e-6:
                    LOGGER.warning(
                        f"[{lead}] S-wave amplitude is near zero (|S|={abs_s:.2e}); "
                        f"R/|S| ratio set to NaN to avoid division by near-zero."
                    )
                    r_over_s = np.nan
                else:
                    r_over_s = r_amplitude / abs_s

            # Store without penalty for now (filled in pass 2)
            result[lead] = RProgressionInfo(
                lead=lead,
                r_amplitude=r_amplitude,
                s_amplitude=s_amplitude,
                r_over_s=r_over_s,
                r_present=r_present,
                s_present=s_present,
                r_over_s_monotony_penalty=np.nan,
            )

            if np.isnan(r_over_s):
                LOGGER.info(f"[{lead}] R={r_amplitude}  S={s_amplitude}  R/|S|=NaN")
            else:
                LOGGER.info(
                    f"[{lead}] R={r_amplitude:.3f}  S={s_amplitude:.3f}  "
                    f"R/|S|={r_over_s:.3f}"
                )

        # --- Pass 2 : monotonicity penalty over the full V1→V4 sequence ---
        ratios = [result[lead].r_over_s for lead in _R_PROGRESSION_LEADS]
        valid_pairs = [
            (ratios[i], ratios[i + 1])
            for i in range(len(ratios) - 1)
            if not (np.isnan(ratios[i]) or np.isnan(ratios[i + 1]))
        ]

        if len(valid_pairs) < 1:
            monotony_penalty = np.nan
            LOGGER.warning(
                "R-wave progression monotonicity penalty set to NaN: "
                "fewer than two valid R/S ratios available across "
                f"{_R_PROGRESSION_LEADS}."
            )
        else:
            monotony_penalty = float(
                sum(max(0.0, r_curr - r_next) for r_curr, r_next in valid_pairs)
            )
            LOGGER.info(
                f"R-wave progression monotonicity penalty = {monotony_penalty:.4f} "
                f"(0 = perfectly non-decreasing)."
            )

        # Inject the global penalty into every per-lead instance
        for lead in _R_PROGRESSION_LEADS:
            result[lead].r_over_s_monotony_penalty = monotony_penalty

        LOGGER.info("R-wave progression computed for leads V1–V4.")
        return result
    
    
    def _evaluate_r_progression_criterion(self) -> Optional[bool]:
        """
        Evaluate whether R-wave progression from V1 to V4 is physiologically normal.

        Two sub-criteria must both be satisfied:

        1. **Transition zone present** : R/|S| ≥ 1.0 for at least one lead in
           V1–V4 (R becomes dominant before or at V4).
        2. **Monotonic non-decrease** : ``r_over_s_monotony_penalty == 0``
           (no R/S decrease between any consecutive valid lead pair).

        Returns ``None`` when insufficient data prevents a verdict
        (``r_over_s_monotony_penalty`` is ``np.nan``, i.e. fewer than two
        valid R/S ratios available).

        Returns
        -------
        bool or None
            ``True``  — both sub-criteria satisfied.
            ``False`` — at least one sub-criterion violated.
            ``None``  — undecidable (too few valid R/S ratios).

        Raises
        ------
        RuntimeError
            If ``compute_qrs_waves()`` has not been called yet.

        Possible changes 
        ------
        Method isn't used in the implausibility score. The monotony penalty 
        contribution is calculated using _compute_r_progression.
        """
        data = self.r_progression  # triggers lazy computation if needed

        # Retrieve the global monotonicity penalty (same value in all instances)
        penalty = data[_R_PROGRESSION_LEADS[0]].r_over_s_monotony_penalty

        if np.isnan(penalty):
            LOGGER.warning(
                "R-wave progression criterion undecidable: "
                "r_over_s_monotony_penalty is NaN (too few valid R/S ratios)."
            )
            return None

        # Sub-criterion 1 : transition zone
        valid_ratios = [
            data[lead].r_over_s
            for lead in _R_PROGRESSION_LEADS
            if not np.isnan(data[lead].r_over_s)
        ]
        transition_present = any(r >= 1.0 for r in valid_ratios)

        # Sub-criterion 2 : monotonic non-decrease
        is_monotone = penalty == 0.0

        result = transition_present and is_monotone
        LOGGER.info(
            f"R-wave progression criterion: "
            f"transition={'present' if transition_present else 'absent'}, "
            f"monotony_penalty={penalty:.4f} "
            f"→ {'NORMAL' if result else 'ABNORMAL'}."
        )
        return result

    # --- Implausibility score against a clinical reference -----------------

    def _wave_presence_penalty(
        self,
        sim_present: bool,
        ref_present: bool,
        weight_key: str,
        weights: dict,
        flags: dict,
        flag_key: str,
    ) -> Optional[float]:
        """
        Resolve the conventional penalty for a wave-presence mismatch.

        Returns
        -------
        float or None
            ``_NAN_PENALTY_SIGMA_MULTIPLE / weights[weight_key]`` if exactly
            one side (simulated or reference) has the wave absent while the
            other has it present (a genuine mismatch). 
            ``None`` if presence matches on both sides (caller should fall
            through to the normal penalty formula) or if the wave is absent
            on both sides (caller should report ``np.nan``, handled by the
            caller, not this helper).

        Possible changes
        ------
        It would probably be better to find ways to have continuous penalties 
        applied when there is a mismatch in wave presence. For example by using 
        wave duration or wave value (set to 0 when wave is absent). 
        Modification already made for Q wave durations but not for the
        q_over_r penalty
        """
        if sim_present == ref_present:
            return None  # no mismatch — caller computes/handles normally

        penalty = _NAN_PENALTY_SIGMA_MULTIPLE / weights[weight_key]
        side = "simulated" if ref_present else "reference"
        flags[flag_key] = (
            f"Wave absent in {side} ECG but present in the other — "
            f"conventional penalty {_NAN_PENALTY_SIGMA_MULTIPLE}/weight applied."
        )
        return penalty

    def _implausibility_qrs_duration(self, reference: "ReferenceMetrics") -> float:
        """Absolute difference in total QRS duration (ms)."""
        return float(abs(self.qrs_duration - reference.qrs_duration))

    def _implausibility_q_duration(
        self, reference: "ReferenceMetrics", weights: dict, flags: dict
    ) -> dict:
        """
        Per-lead Q-wave duration penalty (ms) over ``_Q_WAVE_LEADS``.

        Project decision: Q absence is treated as a neutral, physiologically
        meaningful value (duration = 0.0 ms) rather than triggering the
        conventional mismatch penalty — consistent with how
        ``q_amplitude_v1`` already treats Q absence in V1. A one-sided
        mismatch (Q present on one side, absent on the other) therefore
        still produces a penalty, but its magnitude is bounded by the
        present side's actual duration rather than by a fixed punitive
        multiple of the weight.

        Reference 
        ------
        https://cardvasc.org/ecg-qrs-complex-q-r-s-wave-duration-interval/
        &
        https://ecg.utah.edu/lesson/3
        """
        penalties = {}
        for lead in _Q_WAVE_LEADS:
            key = f"q_duration_{lead}"
            q_sim = self.qrs_waves[lead].Q
            sim_present = q_sim.n_peaks > 0
            ref_value = reference.q_duration.get(lead, np.nan)
            ref_present = not np.isnan(ref_value)

            duration_sim = float(q_sim.wave_duration) if sim_present else 0.0
            duration_ref = float(ref_value) if ref_present else 0.0

            if sim_present != ref_present:
                side = "simulated" if ref_present else "reference"
                flags[key] = (
                    f"Q wave absent in {side} ECG but present in the other — "
                    f"treated as duration=0.0 on the absent side (not a "
                    f"punitive mismatch penalty)."
                )

            penalties[key] = float(abs(duration_sim - duration_ref))
        return penalties

    def _implausibility_q_over_r(
        self, reference: "ReferenceMetrics", weights: dict, flags: dict
    ) -> dict:
        """
        Per-lead intra-lead |Q|/R amplitude ratio penalty over ``_Q_WAVE_LEADS``.

        Note: Q absence already resolves to ``q_over_r_sim = 0.0`` through
        the normal calculation path below (``q_min = 0.0`` when Q has no
        peaks), consistent with the ``q_duration`` convention — no special
        case needed for that side. The presence gate below is on **R**
        (the denominator) only: R absence on one side is a genuine
        structural mismatch with no neutral substitute value, so it still
        receives the conventional punitive penalty.

        Reference
        ------
        https://cardvasc.org/ecg-qrs-complex-q-r-s-wave-duration-interval/
        &
        https://ecg.utah.edu/lesson/3
        """
        penalties = {}
        for lead in _Q_WAVE_LEADS:
            key = f"q_over_r_{lead}"
            waves = self.qrs_waves[lead]
            r_present = waves.R.n_peaks > 0
            ref_value = reference.q_over_r.get(lead, np.nan)
            ref_present = not np.isnan(ref_value)

            # Penalty is gated on R presence (denominator), per project spec.
            mismatch_penalty = self._wave_presence_penalty(
                r_present, ref_present, "q_over_r", weights, flags, key
            )
            if mismatch_penalty is not None:
                penalties[key] = mismatch_penalty
            elif not r_present:  # absent on both sides
                penalties[key] = np.nan
            else:
                q_min = float(np.min(waves.Q.peak_values)) if waves.Q.n_peaks > 0 else 0.0
                r_max = float(np.max(waves.R.peak_values))
                q_over_r_sim = abs(q_min) / r_max
                penalties[key] = float(abs(q_over_r_sim - ref_value))
        return penalties

    def _implausibility_onset_to_peak(
        self, reference: "ReferenceMetrics", weights: dict, flags: dict
    ) -> dict:
        """
        Per-lead QRS-onset-to-R-peak penalty (ms) over ``_ONSET_TO_PEAK_LEADS``.

        R absence on either side (simulated or reference) keeps the
        conventional punitive mismatch penalty, symmetrically in both
        directions: there is no physiologically neutral "0 ms" substitute
        for a missing R wave in V5/V6 (unlike Q absence elsewhere), since R
        is expected to be dominant in these leads for a normal heart.

        Reference
        ------
        Aiken, A. V., Goldhaber, J. I., & Chugh, S. S. (2022). 
        Delayed intrinsicoid deflection: Electrocardiographic harbinger of heart disease. 
        Annals of Noninvasive Electrocardiology, 27, e12940. 
        https://doi.org/10.1111/anec.12940
        """
        penalties = {}
        min_lv = self._get_min_activation_time_for_part(self._model.left_ventricle)
        min_rv = self._get_min_activation_time_for_part(self._model.right_ventricle)
        min_septum = self._get_min_activation_time_for_part(self._model.septum)
        t_qrs_start = min(min_lv, min_rv, min_septum)

        for lead in _ONSET_TO_PEAK_LEADS:
            key = f"onset_to_peak_{lead}"
            r_wave = self.qrs_waves[lead].R
            r_present = r_wave.n_peaks > 0
            ref_value = reference.onset_to_peak.get(lead, np.nan)
            ref_present = not np.isnan(ref_value)

            mismatch_penalty = self._wave_presence_penalty(
                r_present, ref_present, "onset_to_peak", weights, flags, key
            )
            if mismatch_penalty is not None:
                penalties[key] = mismatch_penalty
            elif not r_present:
                penalties[key] = np.nan
            else:
                onset_to_peak_sim = r_wave.time - t_qrs_start
                penalties[key] = float(abs(onset_to_peak_sim - ref_value))
        return penalties

    def _implausibility_r_progression_monotony(
        self, reference: "ReferenceMetrics"
    ) -> float:
        """Global R/S monotonicity penalty mismatch across V1-V4 (dimensionless)."""
        monotony_sim = self.r_progression[_R_PROGRESSION_LEADS[0]].r_over_s_monotony_penalty
        if np.isnan(monotony_sim):
            return np.nan
        return float(abs(monotony_sim - reference.r_progression_monotony))

    def _implausibility_r_over_s(
        self, reference: "ReferenceMetrics", weights: dict, flags: dict
    ) -> dict:
        """
        Per-lead R/|S| amplitude ratio penalty over ``_R_OVER_S_LEADS``.

        Follows the project convention: 0.0 if R is absent (R dominance by
        construction is moot — ratio is trivially low and well-defined), a
        NaN-derived case only when S is absent (denominator undefined).

        Problem to solve 
        ------
        flaw in the logic here : penalty set to 0 if S absent in V5 is 
        normal but not in V1 where S should be dominant in a healthy heart
        both cases are treated equally in the loop on the _R_OVER_S_LEADS
        Cases should be treated differently !
        """
        penalties = {}
        for lead in _R_OVER_S_LEADS:
            key = f"r_over_s_{lead}"
            waves = self.qrs_waves[lead]
            r_present = waves.R.n_peaks > 0
            s_present = waves.S.n_peaks > 0
            ref_value = reference.r_over_s.get(lead, np.nan)

            if not r_present:
                r_over_s_sim = 0.0
            elif not s_present:
                r_over_s_sim = np.nan
            else:
                r_max = float(np.max(waves.R.peak_values))
                s_min = float(np.min(waves.S.peak_values))
                r_over_s_sim = r_max / abs(s_min) if abs(s_min) > 1e-9 else np.nan

            if np.isnan(r_over_s_sim):
                # S absent in simulated ECG: per spec, penalty is 0 if R is
                # dominant by construction (R present, large), otherwise the
                # conventional maximal penalty applies.

                # PROBLEM : V1 and V5 treated equally isn't correct !
                if r_present:
                    penalties[key] = 0.0
                    flags[key] = (
                        "S wave absent in simulated ECG with R present "
                        "(R-dominant by construction) — penalty set to 0."
                    )
                else:
                    penalties[key] = _NAN_PENALTY_SIGMA_MULTIPLE / weights[f"r_over_s_{lead}"]
                    flags[key] = (
                        "S wave absent and R absent in simulated ECG — "
                        "conventional maximal penalty applied."
                    )
            else:
                penalties[key] = float(abs(r_over_s_sim - ref_value))
        return penalties

    def _implausibility_q_amplitude_v1(self, reference: "ReferenceMetrics") -> float:
        """
        Q-wave amplitude penalty in V1 (no NaN case possible by construction).
        This amplitude is usually 0 in a healhy heart
        """
        q_wave_v1 = self.qrs_waves["V1"].Q
        q_amplitude_v1_sim = (
            float(abs(np.min(q_wave_v1.peak_values))) if q_wave_v1.n_peaks > 0 else 0.0
        )
        return float(abs(q_amplitude_v1_sim - reference.q_amplitude_v1))

    def _implausibility_notches(
        self, reference: "ReferenceMetrics"
    ) -> tuple:
        """
        Notch depth and interval penalties over ``_NOTCH_LEADS``.

        For each lead/wave, the **deepest** notch is identified
        (``argmax(notch_depths)``), and both its depth and its temporal
        width (``peak_intervals`` at that same index) are compared to the
        reference. Depth and interval are therefore always read from one
        single identified notch, not two independently-selected ones — a
        notch that is both deep and narrow is characterised differently
        from one that is deep but wide, which a depth-only criterion would
        not distinguish.

        Returns
        -------
        tuple(dict, dict)
            ``(depth_penalties, interval_penalties)``, each keyed by
            ``"{wave}_{lead}"``. No NaN case possible by construction: a
            wave with fewer than 2 peaks (no notch) or absent contributes
            ``0.0`` for both depth and interval, exactly like a reference
            with no notch.

        Notes
        ------
        Notches contribution is inaccurate (doesn't correlate to apparent 
        notch gravity),
        Notch depth and penalty definitions should be investigated
        """
        depth_penalties = {}
        interval_penalties = {}
        for wave_name, leads in _NOTCH_LEADS.items():
            for lead in leads:
                wave = getattr(self.qrs_waves[lead], wave_name)

                if wave.notch_depths.size > 0:
                    deepest_idx = int(np.argmax(wave.notch_depths))
                    depth_sim = float(wave.notch_depths[deepest_idx])
                    interval_sim = float(wave.peak_intervals[deepest_idx])
                else:
                    depth_sim = 0.0
                    interval_sim = 0.0

                ref_depth = reference.notch_depth.get(wave_name, {}).get(lead, 0.0)
                ref_interval = reference.notch_interval.get(wave_name, {}).get(lead, 0.0)

                key = f"{wave_name}_{lead}"
                depth_penalties[key] = float(abs(depth_sim - ref_depth))
                interval_penalties[key] = float(abs(interval_sim - ref_interval))
        return depth_penalties, interval_penalties

    def _implausibility_extra_extrema(
        self, reference: "ReferenceMetrics"
    ) -> dict:
        """
        Late-extrema penalty over all 12 leads in ``LEAD_INDEX``.

        For each lead, the SIGNED sum of every late extremum's normalized
        value (maxima positive, minima negative — see
        ``QrsWaves.extra_extrema``) is compared to the same signed sum on
        the reference side. Project decision: the sum is signed, not an
        absolute-value sum — a maximum and a minimum of similar magnitude
        on the same lead can cancel out.

        Returns
        -------
        dict
            Penalty keyed by lead name (``LEAD_INDEX`` keys). No NaN case
            possible by construction: no late extrema detected contributes
            a sum of ``0.0`` (``sum([]) == 0.0``), exactly like a reference
            with no late extrema — there is no "absent" state distinct from
            "zero" for this criterion, so no punitive mismatch penalty
            applies here.

        Notes 
        ------
        Using a signed sum could be reconsidered
        """
        penalties = {}
        for lead in LEAD_INDEX:
            extrema = self.qrs_waves[lead].extra_extrema
            sim_value = float(np.sum(extrema["maxima"]["values"])) + float(
                np.sum(extrema["minima"]["values"])
            )
            ref_value = reference.extra_extrema.get(lead, 0.0)
            penalties[lead] = float(abs(sim_value - ref_value))
        return penalties

    def _implausibility_electrical_axis(
        self, reference: "ReferenceMetrics", weights: dict, flags: dict
    ) -> tuple:
        """
        Circular-distance penalty on the electrical axis, plus an ambiguity flag.

        Returns
        -------
        tuple(float, float)
            ``(electrical_axis_penalty, axis_ambiguous_penalty)``.
            If either side's axis is NaN (ambiguous), the axis penalty is set
            to ``0.0`` and the ambiguity penalty is set to ``1.0`` (flagged
            explicitly), per project spec.
        
        Notes 
        ------
        The ambiguity penalty is completely arbitrary (like the wave presence mismatch penalty)
        It's value could be reconsidered.
        """
        axis_sim = self.electrical_axis
        axis_ref = reference.electrical_axis

        if np.isnan(axis_sim) or np.isnan(axis_ref):
            side = "simulated" if np.isnan(axis_sim) else "reference"
            flags["axis_ambiguous"] = f"Electrical axis ambiguous in {side} ECG."
            return 0.0, 1.0

        diff = (axis_sim - axis_ref + 180.0) % 360.0 - 180.0
        return float(abs(diff)), 0.0

    def compute_implausibility(
        self,
        reference: "ReferenceMetrics",
        weights: dict = None,
    ) -> "ImplausibilityResult":
        """
        Compute the implausibility score of the simulated QRS complex against
        a clinical reference.

        This is the entry point of the inverse-problem cost function: it
        aggregates every QRS biomarker penalty (durations, intra-lead
        amplitude ratios, R-wave progression, notches, electrical axis) into
        a single scalar ``total_score``, suitable for an SMC-ABC /
        optimisation loop (Camps et al., 2024, Med. Image Anal., section 2.5).

        Parameters
        ----------
        reference : ReferenceMetrics
            Pre-computed clinical biomarkers (e.g. from
            ``load_ptbxl_reference``). Independent of any PyAnsys model/post
            object — see :class:`ReferenceMetrics`.
        weights : dict, optional
            Per-criterion weight, keyed the same way as ``penalty_vector``
            (see Notes). Any key not provided falls back to
            ``DEFAULT_WEIGHTS``. Passing a partial dict only overrides the
            specified keys. Each default weight is a normalisation factor
            (``weight = 1 / sigma_estimate``, with ``sigma_estimate`` derived
            from a plausible range for that criterion over
            the full optimisation parameter space — see the module-level
            ``DEFAULT_WEIGHTS`` comment for the sourcing status and caveats
            of each value), **not** a measurement-uncertainty estimate.

        Returns
        -------
        ImplausibilityResult
            ``total_score`` (scalar cost), ``penalty_vector`` (raw per-leaf
            penalties), ``weights`` actually used, and ``flags`` (diagnostic
            notes for wave-presence mismatches / axis ambiguity).

        Raises
        ------
        RuntimeError
            If ``compute_qrs_waves()`` has not been called yet (required for
            every biomarker in this scope).

        Notes
        -----
        Aggregation formula:

            total_score = sum(
                weights[k] * penalty_vector[k]
                for k in penalty_vector
                if not np.isnan(penalty_vector[k])
            )

        This is a simplified history-matching-style score: it does not
        include a separate model-variance term (``Var[f(x)]``) as in the
        full formulation (Strocchi et al., 2026); only a single
        normalisation factor per criterion is used. Adding a model-variance
        term (e.g. from repeated simulations or an emulator) is a possible
        future improvement, not implemented here.

        ``penalty_vector`` keys and the weight key each maps to:

        - ``"qrs_duration"`` -> weight ``"qrs_duration"``
        - ``"q_duration_{lead}"`` for lead in ``_Q_WAVE_LEADS`` -> weight ``"q_duration"``
        - ``"q_over_r_{lead}"`` for lead in ``_Q_WAVE_LEADS`` -> weight ``"q_over_r"``
        - ``"onset_to_peak_{lead}"`` for lead in ``_ONSET_TO_PEAK_LEADS`` -> weight ``"onset_to_peak"``
        - ``"r_progression_monotony"`` -> weight ``"r_progression_monotony"``
        - ``"r_over_s_{lead}"`` for lead in ``_R_OVER_S_LEADS`` -> weight ``"r_over_s_{lead}"``
          (a separate weight per lead — V1 and V5 have distinct plausible
          ranges, see ``DEFAULT_WEIGHTS``)
        - ``"q_amplitude_v1"`` -> weight ``"q_amplitude_v1"``
        - ``"notch_depth_{wave}_{lead}"`` for (wave, lead) in ``_NOTCH_LEADS`` -> weight ``"notch_depth"``
        - ``"notch_interval_{wave}_{lead}"`` for (wave, lead) in ``_NOTCH_LEADS`` -> weight ``"notch_interval"``
        - ``"extra_extrema_{lead}"`` for lead in ``LEAD_INDEX`` -> weight ``"extra_extrema"``
        - ``"electrical_axis"`` -> weight ``"electrical_axis"``
        - ``"axis_ambiguous"`` -> weight ``"axis_ambiguous"``

        NaN handling: a NaN in ``penalty_vector`` only ever occurs when a
        wave is absent on **both** simulated and reference sides for a
        biomarker that requires it (genuinely undecidable, excluded from
        the sum). A one-sided absence is converted upstream into either a
        physiologically neutral substitute value (Q-wave criteria — see
        ``_implausibility_q_duration``) or, where no neutral substitute
        exists, the fixed conventional penalty
        ``_NAN_PENALTY_SIGMA_MULTIPLE / weight[k]`` (R-wave-gated criteria:
        ``q_over_r``, ``onset_to_peak``, ``r_over_s``). Either way it
        appears as a finite value here, not as NaN.

        Caching: the result is cached in ``self._implausibility`` and
        accessible afterwards via the ``implausibility`` property, until
        this method is called again.
        """
        if self.qrs_waves is None:
            raise RuntimeError(
                "QRS waves have not been computed yet. "
                "Call compute_qrs_waves() first."
            )

        weights_provided = dict(DEFAULT_WEIGHTS)
        if weights is not None:
            weights_provided.update(weights)

        flags: dict = {}
        penalty_vector: dict = {}

        # --- 1. QRS duration ---
        penalty_vector["qrs_duration"] = self._implausibility_qrs_duration(reference)

        # --- 2. Q-wave duration (per lead) ---
        penalty_vector.update(
            self._implausibility_q_duration(reference, weights_provided, flags)
        )

        # --- 3. Q/R amplitude ratio (per lead) ---
        penalty_vector.update(
            self._implausibility_q_over_r(reference, weights_provided, flags)
        )

        # --- 4. QRS onset-to-R-peak (per lead) ---
        penalty_vector.update(
            self._implausibility_onset_to_peak(reference, weights_provided, flags)
        )

        # --- 5. R-wave progression monotony (global) ---
        penalty_vector["r_progression_monotony"] = (
            self._implausibility_r_progression_monotony(reference)
        )

        # --- 6. R/S amplitude ratio (per lead) ---
        penalty_vector.update(
            self._implausibility_r_over_s(reference, weights_provided, flags)
        )

        # --- 7. Q-wave amplitude in V1 ---
        penalty_vector["q_amplitude_v1"] = self._implausibility_q_amplitude_v1(reference)

        # --- 8. Notches (depth + interval of the deepest notch, per wave/lead) ---
        depth_penalties, interval_penalties = self._implausibility_notches(reference)
        penalty_vector.update({f"notch_depth_{k}": v for k, v in depth_penalties.items()})
        penalty_vector.update({f"notch_interval_{k}": v for k, v in interval_penalties.items()})

        # --- 8bis. Late extrema (signed sum, per lead) ---
        penalty_vector.update(
            {
                f"extra_extrema_{lead}": v
                for lead, v in self._implausibility_extra_extrema(reference).items()
            }
        )

        # --- 9. Electrical axis ---
        axis_penalty, axis_ambiguous_penalty = self._implausibility_electrical_axis(
            reference, weights_provided, flags
        )
        penalty_vector["electrical_axis"] = axis_penalty
        penalty_vector["axis_ambiguous"] = axis_ambiguous_penalty

        # --- Weight lookup per penalty key ---
        # Most criteria share one weight across all their leads (category
        # weight, resolved by prefix below). `r_over_s_{lead}` is the
        # exception: V1 and V5 have distinct plausible ranges and therefore
        # distinct weights, already keyed by their full name in
        # DEFAULT_WEIGHTS — no prefix resolution needed for it.
        def _weight_for_key(key: str) -> float:
            if key in weights_provided:
                return weights_provided[key]
            if key.startswith("q_duration_"):
                return weights_provided["q_duration"]
            if key.startswith("q_over_r_"):
                return weights_provided["q_over_r"]
            if key.startswith("onset_to_peak_"):
                return weights_provided["onset_to_peak"]
            if key.startswith("notch_depth_"):
                return weights_provided["notch_depth"]
            if key.startswith("notch_interval_"):
                return weights_provided["notch_interval"]
            if key.startswith("extra_extrema_"):
                return weights_provided["extra_extrema"]
            raise KeyError(
                f"No weight value found for penalty key '{key}'. "
                f"Provide it explicitly via the `weights` argument."
            )

        weights_per_key = {k: _weight_for_key(k) for k in penalty_vector}

        total_score = sum(
            weights_per_key[k] * penalty_vector[k]
            for k in penalty_vector
            if not np.isnan(penalty_vector[k])
        )

        LOGGER.info(
            f"Implausibility score computed: total_score={total_score:.4f} "
            f"over {len(penalty_vector)} criteria "
            f"({sum(1 for v in penalty_vector.values() if np.isnan(v))} NaN-excluded, "
            f"{len(flags)} flagged)."
        )

        result = ImplausibilityResult(
            total_score=float(total_score),
            penalty_vector=penalty_vector,
            weights=weights_per_key,
            flags=flags,
        )
        self._implausibility = result
        self._implausibility_reference = reference
        return result
