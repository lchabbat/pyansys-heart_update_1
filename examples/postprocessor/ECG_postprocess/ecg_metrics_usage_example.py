# Copyright (C) 2023 - 2026 ANSYS, Inc. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
ecg_metrics_usage_example.py

Postprocess an ECG computed from a Reaction-Eikonal simulation.
-------------------------------------
This example shows how to postprocess an ECG, extract data from it measured 
using the EcgMetrics methods.
"""

###############################################################################
# .. warning::
#    When using a standalone version of the DPF Server, you must accept the `license terms
#    <https://dpf.docs.pyansys.com/version/stable/getting_started/licensing.html>`_. To
#    accept these terms, you can set this environment variable:
#
#    .. code-block:: python
#
#        import os
#        os.environ["ANSYS_DPF_ACCEPT_LA"] = "Y"

###############################################################################
# Perform the required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import the required modules and set relevant paths.

from pathlib import Path
from ansys.health.heart.examples import get_preprocessed_fullheart
from ansys.health.heart.post.dpf_utils import EPpostprocessor
import ansys.health.heart.models as models
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ansys.health.heart import LOG as LOGGER
from ecg_metrics import EcgMetrics, LEAD_INDEX, _R_PROGRESSION_LEADS

LOGGER.setLevel("WARNING")

os.environ["ANSYS_DPF_ACCEPT_LA"] = "Y"

###############################################################################
# Create a postprocessor object
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

###############################################################################
# .. note::
#    This example assumes that you have you ran a full heart electrophysiology simulation
#    and that the d3plot files are located in ``data_path``.

# Import the required modules and set relevant paths.
workdir = Path.home() / "pyansys-heart"/"downloads" / "Rodero2021" / "01" / "FullHeart"
path_to_model, path_to_partinfo, _ = get_preprocessed_fullheart(resolution="2.0mm")

# Specify the path to the d3plot that contains the simulation results.
simulation_folder_name = 'set_LHS_02'
data_path = workdir / simulation_folder_name / "main_ep_reaction_eikonal" / "d3plot"

path_to_ecg = workdir / simulation_folder_name / "main_ep_reaction_eikonal" / "em_EKG_001.dat"

# Check if the file exists.
if not data_path.is_file():
    raise FileNotFoundError(f"File not found: {data_path}")

# Initialize the postprocessor.
post = EPpostprocessor(data_path)

###############################################################################
# Load the full-heart model
model: models.FullHeart = models.HeartModel.load_model(
    path_to_model, path_to_partinfo, working_directory=workdir
)

# Save the model.
model.mesh.save(os.path.join(model.workdir, "simulation_model.vtu"))

###############################################################################
# Call methods to retrieve activation time
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get activation time of the full field at the last time step.
activation_times = post.get_activation_times()
 
metrics = EcgMetrics(model=model, post=post)

print("QRS is ",metrics.qrs_duration, " ms long")
# print("QT-Interval is ",metrics.qt_interval, " ms long")
print("PQ interval is ", metrics.pq_interval, " ms long")
print("P wave duration is ",metrics.p_wave_duration, " ms long")



# ---------------------------------------------------------------------------
# Attach ECG signals to the metrics object
# ---------------------------------------------------------------------------
ecgs, times_ecg = post.read_ECGs(path_to_ecg)
ecg_12lead = post.compute_12_lead_ECGs(ecgs, times_ecg, plot=False)

metrics = EcgMetrics(model=model, post=post, ecg_12lead=ecg_12lead, times=times_ecg)


def compute_12_lead_ECGs_interactive(
    ECGs: np.ndarray,
    times: np.ndarray,
    plot: bool = True,
    save_path: str | None = None,  
    with_vwct_ecg_plot: bool = True,
    superposed_with_and_without_vwct: bool = False,
) -> np.ndarray:
    """Compute 12-lead ECGs from 10 electrodes and plot interactively
    with both keyboard arrows and a slider.
    """

    # --- calcul des 12 dérivations ---
    right_arm = ECGs[:, 6]
    left_arm = ECGs[:, 7]
    left_leg = ECGs[:, 9]

    lead1 = left_arm - right_arm
    lead2 = left_leg - right_arm
    lead3 = left_leg - left_arm
    lead_avr = right_arm - (left_arm + left_leg) / 2
    lead_avl = left_arm - (left_leg + right_arm) / 2
    lead_avf = left_leg - (right_arm + left_arm) / 2
    Vwct = (left_arm + right_arm + left_leg) / 3
    lead_v1 = ECGs[:, 0] - Vwct
    lead_v2 = ECGs[:, 1] - Vwct
    lead_v3 = ECGs[:, 2] - Vwct
    lead_v4 = ECGs[:, 3] - Vwct
    lead_v5 = ECGs[:, 4] - Vwct
    lead_v6 = ECGs[:, 5] - Vwct

    electrode_v1 = ECGs[:, 0]
    electrode_v2 = ECGs[:, 1]
    electrode_v3 = ECGs[:, 2]
    electrode_v4 = ECGs[:, 3]
    electrode_v5 = ECGs[:, 4]
    electrode_v6 = ECGs[:, 5]       

    ecg_12lead = np.vstack(
        (
            lead1, lead2, lead3,
            lead_avr, lead_avl, lead_avf,
            lead_v1, lead_v2, lead_v3, lead_v4, lead_v5, lead_v6,
        )
    )

    if plot:
        # --- Création figure principale ---
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))
        plt.subplots_adjust(bottom=0.15)

        if with_vwct_ecg_plot:

            leads = [
                (lead1, "I",   axes[0, 0]),
                (lead2, "II",  axes[1, 0]),
                (lead3, "III", axes[2, 0]),
                (lead_avr, "aVR", axes[0, 1]),
                (lead_avl, "aVL", axes[1, 1]),
                (lead_avf, "aVF", axes[2, 1]),
                (lead_v1, "V1", axes[0, 2]),
                (lead_v2, "V2", axes[1, 2]),
                (lead_v3, "V3", axes[2, 2]),
                (lead_v4, "V4", axes[0, 3]),
                (lead_v5, "V5", axes[1, 3]),
                (lead_v6, "V6", axes[2, 3]),
            ]
        else :
            leads = [
                (lead1, "I",   axes[0, 0]),
                (lead2, "II",  axes[1, 0]),
                (lead3, "III", axes[2, 0]),
                (lead_avr, "aVR", axes[0, 1]),
                (lead_avl, "aVL", axes[1, 1]),
                (lead_avf, "aVF", axes[2, 1]),
                (electrode_v1, "V1", axes[0, 2]),
                (electrode_v2, "V2", axes[1, 2]),
                (electrode_v3, "V3", axes[2, 2]),
                (electrode_v4, "V4", axes[0, 3]),
                (electrode_v5, "V5", axes[1, 3]),
                (electrode_v6, "V6", axes[2, 3]),
            ]
            
        cursor_lines = []
        for lead, label, ax in leads:
            ax.plot(times, lead, color="black")
            ax.set_ylabel(label)
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            cursor_lines.append(ax.axvline(times[0], color="red", linewidth=1.5))

        idx = {"t": 0}

        def update_cursor():
            t = times[idx["t"]]
            for line in cursor_lines:
                line.set_xdata([t, t])
            fig.suptitle(f"Time = {t:.1f} ms (index {idx['t']})", fontsize=14)
            fig.canvas.draw_idle()

        def on_key(event):
            if event.key == "right":
                idx["t"] = min(idx["t"] + 1, len(times) - 1)
                slider.set_val(idx["t"])
                update_cursor()
            elif event.key == "left":
                idx["t"] = max(idx["t"] - 1, 0)
                slider.set_val(idx["t"])
                update_cursor()

        fig.canvas.mpl_connect("key_press_event", on_key)

        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        slider = Slider(
            ax=ax_slider,
            label="Time index",
            valmin=0,
            valmax=len(times) - 1,
            valinit=0,
            valstep=1,
        )

        def on_slider(val):
            idx["t"] = int(val)
            update_cursor()

        slider.on_changed(on_slider)

        update_cursor()
        plt.show(block=True)

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150)
            print(f"Figure sauvegardée dans {save_path}")

    return ecg_12lead




# ---------------------------------------------------------------------------
# QRS wave identification — all leads
# ---------------------------------------------------------------------------

metrics.compute_qrs_waves(prominence_fraction=0.02)
    
compute_12_lead_ECGs_interactive(ecgs, times_ecg, with_vwct_ecg_plot=False, superposed_with_and_without_vwct=False)

# ---------------------------------------------------------------------------
# Terminal examples 
# ---------------------------------------------------------------------------

print(f'Q wave duration in lead I is {metrics.qrs_waves["I"].Q.wave_duration}')
print(f'Q wave duration in lead V2 is {metrics.qrs_waves["V2"].Q.wave_duration}')

leads_to_print = ["I", "II", "V1", "V2"]

print("\n--- QRS wave identification ---")
for lead in leads_to_print:
    waves = metrics.qrs_waves[lead]
    print(f"\n  Lead {lead}")

    for wave_name in ["Q", "R", "S"]:
        wave = getattr(waves, wave_name)
        if wave is None:
            print(f"    {wave_name} : absent")
        else:
            print(f"    {wave_name} : t={wave.time:.1f} ms  "
                  f"val={wave.value:.3f}  "
                  f"n_peaks={wave.n_peaks}")
            if wave.n_peaks > 1:
                print(f"         peak_times     : {np.round(wave.peak_times, 1)} ms")
                print(f"         peak_intervals : {np.round(wave.peak_intervals, 1)} ms")

    extra = waves.extra_extrema
    n_extra = len(extra["maxima"]["times"]) + len(extra["minima"]["times"])
    if n_extra > 0:
        print(f"    Late notches : {len(extra['maxima']['times'])} maxima, "
              f"{len(extra['minima']['times'])} minima")
        if len(extra["maxima"]["times"]) > 0:
            print(f"         maxima times : {np.round(extra['maxima']['times'], 1)} ms")
        if len(extra["minima"]["times"]) > 0:
            print(f"         minima times : {np.round(extra['minima']['times'], 1)} ms")
    else:
        print(f"    Late notches : none")

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(len(leads_to_print), 1,
                         figsize=(10, 3 * len(leads_to_print)), layout="tight")

min_lv     = metrics._get_min_activation_time_for_part(metrics._model.left_ventricle)
min_rv     = metrics._get_min_activation_time_for_part(metrics._model.right_ventricle)
min_septum = metrics._get_min_activation_time_for_part(metrics._model.septum)
t_qrs_start = min(min_lv, min_rv, min_septum)
t_qrs_end   = t_qrs_start + metrics.qrs_duration

WAVE_STYLE = {
    "Q": {"color": "green",  "marker": "v"},
    "R": {"color": "red",    "marker": "^"},
    "S": {"color": "blue",   "marker": "v"},
}

for ax, lead in zip(axes, leads_to_print):
    signal = ecg_12lead[LEAD_INDEX[lead]]
    waves  = metrics.qrs_waves[lead]

    ax.plot(times_ecg, signal, color="steelblue", linewidth=1.0, label=lead)
    ax.axvspan(t_qrs_start, t_qrs_end, alpha=0.1, color="orange", label="QRS window")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    for wave_name, style in WAVE_STYLE.items():
        wave = getattr(waves, wave_name)
        if wave is not None:
            for i, t in enumerate(wave.peak_times):
                raw = signal[np.argmin(np.abs(times_ecg - t))]
                ax.scatter(
                    t, raw,
                    color=style["color"], marker=style["marker"],
                    zorder=5, s=80,
                    label=f"{wave_name} peaks" if i == 0 else None,
                )

    for t in waves.extra_extrema["maxima"]["times"]:
        raw = signal[np.argmin(np.abs(times_ecg - t))]
        ax.scatter(t, raw, color="purple", marker="^", zorder=5,
                   s=60, alpha=0.7, label="late notch max")

    for t in waves.extra_extrema["minima"]["times"]:
        raw = signal[np.argmin(np.abs(times_ecg - t))]
        ax.scatter(t, raw, color="purple", marker="v", zorder=5,
                   s=60, alpha=0.7, label="late notch min")

    ax.set_ylabel(lead)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (ms)")
fig.suptitle(
    f"QRS wave identification — prominence_fraction={metrics._qrs_waves_prominence_fraction}"
)
plt.savefig(os.path.join(model.workdir, "qrs_wave_identification.png"), format="png")
plt.show(block=True)

