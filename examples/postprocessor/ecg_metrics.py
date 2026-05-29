"""
ecg_metrics.py
Module dedicated to the extraction of ECG metrics from electrophysiology simulations.
"""

import numpy as np
from ansys.health.heart import LOG as LOGGER

class EcgMetrics:
    """
    Computes and stores in-silico ECG metrics from electrophysiology simulations.

    This class provides on-demand evaluation of standard ECG markers such as 
    QRS duration and QT intervals based on 3D tissue kinematics. It uses lazy loading 
    to ensure heavy computations (like repolarization) are only performed when explicitly requested.
    """
    
    def __init__(self, model, post, activation_times=None):
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
        
        self._repolarization_times = None 
        self._qrs_duration = None
        self._qt_interval = None
        self._p_wave_duration = None
        self._pq_interval = None
        self._activation_times = activation_times

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
    def repolarization_times(self):
        """
        numpy.ndarray: The repolarization times for all nodes (APD90 method).
        Computed only on the first call.
        """
        if self._repolarization_times is None:
            self._repolarization_times = self._compute_repolarization(method="apd90")
        return self._repolarization_times

    @property
    def qrs_duration(self):
        """
        float: The computed QRS duration in milliseconds (Ventricular activation time).
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
        part_repo_times = self.repolarization_times[part_nodes] # Uses the property
        return np.nanmin(part_repo_times)

    def _get_max_repolarization_time_for_part(self, part):
        part_nodes = part.get_node_ids(self._model.mesh)
        part_repo_times = self.repolarization_times[part_nodes] # Uses the property
        return np.nanmax(part_repo_times)

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
        """
        LOGGER.info(f"Starting repolarization computation using method: {method.upper()}")
        
        all_internal_ids = np.arange(len(self.activation_times.data))
        transmembrane_pot, times = self._post.get_transmembrane_potential(all_internal_ids)
        
        dt = times[1] - times[0] if len(times) > 1 else 0
        LOGGER.info(f"Detected temporal resolution (dt): {dt:.2f} ms.")

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
    
    def _compute_p_wave_duration(self):  
        """
        Calculates P-wave duration restricted to atrial tissue.

        .. warning::
            Known limitation: Due to atrioventricular isolation in the mesh, 
            some nodes at the top of the ventricles (basal nodes) are currently 
            considered as atrial nodes. This may produce an artificially long 
            P-wave duration.
        """
        # Avertissement dans la console pour prévenir l'utilisateur au runtime
        LOGGER.warning(
            "P-wave duration computation known issue: AV isolation may include "
            "ventricular basal nodes in the atrial parts, artificially prolonging the result."
        )

        # FIXME: Filter out ventricular nodes leaking into atrial parts due to AV isolation.
        
        min_atria = min(
            self._get_min_activation_time_for_part(self._model.left_atrium), 
            self._get_min_activation_time_for_part(self._model.right_atrium)
        )
        max_atria = max(
            self._get_max_activation_time_for_part(self._model.left_atrium), 
            self._get_max_activation_time_for_part(self._model.right_atrium)
        )
        return max_atria - min_atria
    # Problem: due to the atrioventricular isolation, 
    # Some nodes at the top of the ventricles are considered as atrial nodes 
    # Producing an artifially long P_wave duration

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