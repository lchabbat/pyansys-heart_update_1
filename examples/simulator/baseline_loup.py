# Copyright (C) 2023 - 2026 ANSYS, Inc. and/or its affiliates.
# SPDX-License-Identifier: MIT
"""Baseline configuration for cardiac electrophysiology and mechanics simulations.

This module provides centralized baseline settings and material definitions for
cardiac simulation workflows. It serves as a single source of truth for simulation
parameters that can be imported and reused across different example scripts.
"""

import numpy as np
from pint import Quantity
import pyvista as pv

import ansys.health.heart.models as models
import ansys.health.heart.models_utils as heart_model_utils
from ansys.health.heart.pre.conduction_path import ConductionPath, ConductionPathType
import ansys.health.heart.settings.material.cell_models as cell_models
import ansys.health.heart.settings.material.ep_material as mat
from ansys.health.heart.settings.material.material import (
    ACTIVE,
    ANISO,
    ISO,
    ActiveModel3,
    HGOFiber,
    Mat295,
)
from ansys.health.heart.settings.settings import Purkinje, SystemModel
from scipy.spatial.transform import Rotation as R
from ansys.health.heart.objects import Point


###############################################################################
# Electrophysiology (EP) Material Definitions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define EP materials for conduction system components and myocardial tissue.
# Conductivity values (sigma) are in mS/mm.



# Ventricular myocardium EP properties (anisotropic conductivity)
ep_vent = mat.Active()
ep_vent.sigma_fiber = 0.7  # Fiber direction conductivity
ep_vent.sigma_sheet = 0.35  # Sheet direction conductivity
ep_vent.sigma_sheet_normal = 0.18  # Sheet-normal direction conductivity

# Atrial myocardium EP properties
ep_atrium = mat.Active()
ep_atrium.cell_model = cell_models.TentusscherAtria()  # Atrial cell model
ep_atrium.sigma_fiber = 1.2
ep_atrium.sigma_sheet = 0.6
ep_atrium.sigma_sheet_normal = 0.3


###############################################################################
# Purkinje Network Generation Settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parameters controlling the fractal-like Purkinje network generation algorithm.
purkinje_settings = Purkinje(
    pmjtype=Quantity(1, "dimensionless"),  # PMJ coupling type
    edgelen=Quantity(0.5, "dimensionless"),  # Target edge length for branches
    nsplit=Quantity(6, "dimensionless"),  # Number of splits at each bifurcation
    ngen=Quantity(500, "dimensionless"),  # Number of generations to grow
    nbrinit=Quantity(6, "dimensionless"),  # Initial number of branches from origin
    pmjrestype=Quantity(1, "dimensionless"),  # PMJ resistance type
    pmjradius=Quantity(1.5, "dimensionless"),  # PMJ coupling radius
    pmjres=Quantity(0.001, "1/mS"),  # PMJ resistance value
)

###############################################################################
# Fascicle Endpoint default UHC Coordinates
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
default_laf_endpoint_uhc = [0.0, 0.639, 2.469] 
default_user_fascicle_endpoint_uhc = [0.0, 0.866, -2.589]  
default_lpf_endpoint_uhc = [0.0, 0.81, 2.5951]

###############################################################################
# Space-Varying Cell Model Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Utilities for assigning heterogeneous cell models based on spatial coordinates.


def define_endo_to_epi_node_groups(selected_nodes, coord, bins=[0, 1]):
    """Define node groups based on one coordinate and its respective bins.

    Parameters
    ----------
    selected_nodes : array-like
        Indices of the selected nodes.
    coord : array-like
        Coordinate values for the nodes.
    bins : list, default: [0, 1]
        Bin edges for the coordinate.

    Returns
    -------
    list of np.ndarray
        Node groups organized by the bins of the coordinate.
    """
    masked_coord = coord.copy()
    # set nan for the rest so that they will not be included in the group definition
    masked_coord[~np.isin(np.arange(len(masked_coord)), selected_nodes)] = np.nan

    node_groups = [None] * (len(bins) - 1)

    for i in range(len(bins) - 1):
        mask = (masked_coord >= bins[i]) & (masked_coord <= bins[i + 1])
        node_groups[i] = np.where(mask)[0]

    return node_groups

def define_transmurally_variying_cell_model(model, endo_epi):
    endo_epi_bins = [0, 0.17, 0.41, 1]  # 0-0.17: endo, 0.17-0.41: mid, 0.41-1: epi
    gto_min = cell_models.TentusscherEndo().gto  # extreme value at endo apex
    gto_max = cell_models.TentusscherEpi().gto  # extreme value at epi base
    gks_max = cell_models.TentusscherEpi().gks  # extreme value at epi base
    gks_min = 0.16 # extreme value at endo apex (changed value compared to Tentusscher cell models to ensure an epi to endo repolarization sequence)
        
    ###############################################################################
    # Assign cell models to the ventricles
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Group left- and right-ventricle nodes by endo-epi bins, then
    # interpolate ``gks`` and ``gto`` linearly across the thickness.

    lv_nodes = model.left_ventricle.get_node_ids(model.mesh)
    rv_nodes = model.right_ventricle.get_node_ids(model.mesh)
    ventricle_nodes = np.unique(np.concatenate((lv_nodes, rv_nodes)))

    ventricle_node_groups = define_endo_to_epi_node_groups(
        ventricle_nodes, endo_epi, endo_epi_bins
    )

    n_groups = len(endo_epi_bins) - 1
    factors_1d = np.linspace(0, 1, n_groups)

    gks_array = gks_min + factors_1d * (gks_max - gks_min)
    gto_array = gto_min + factors_1d * (gto_max - gto_min)


    ventricle_cell_models = [
        cell_models.Tentusscher(gks=gks_array[i], gto=gto_array[i]) 
        for i in range(n_groups)
    ]

    ventricle_groups_flat = ventricle_node_groups
    ventricle_models_flat = ventricle_cell_models

    ###############################################################################
    # Assign a uniform endocardial cell model to the septum
    septum_nodes = [model.septum.get_node_ids(model.mesh)]  
    septal_model = [cell_models.TentusscherEndo(gks=0.16)]

    ###############################################################################
    # Tune atrial cell model to make the atria repolarize during the QRS complex
    left_atrium_nodes = [model.left_atrium.get_node_ids(model.mesh)]
    right_atrium_nodes = [model.right_atrium.get_node_ids(model.mesh)]
    atrial_nodes = left_atrium_nodes + right_atrium_nodes

    atrial_models = [cell_models.TentusscherAtria(gks = 0.7, gcal = 0.00002), cell_models.TentusscherAtria(gks = 0.7, gcal = 0.00002)]

    ###############################################################################
    # Merge and assign all cell models
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Combine the ventricle and septum groups and models into flat lists ready
    # for assignment to the heart model.

    all_groups = ventricle_groups_flat + septum_nodes + atrial_nodes
    all_models = ventricle_models_flat + septal_model + atrial_models
    # The cell model will be defined based on the nodeset
    # This overwrites any existing cell model assignment based on part.
    model._nodeset_cellmodel = (all_groups, all_models)

    return all_groups, all_models


def define_conduction_system(
    model: models.HeartModel,
    purkinje_path: str,
    bachman_bundle_keypoints: list,
    mid_sa_av_keypoints: list,
    post_sa_av_keypoints: list,
    left_anterior_fascicle_endpoint: list,
    left_posterior_fascicle_endpoint: list,
    left_user_defined_fascicle_endpoint: list,
):
    """Define the conduction system pathways and assign materials."""

    # Get default conduction system
    beam_list, _landmarks = heart_model_utils.define_full_conduction_system(
        model, purkinje_path
    )
    model._landmarks = _landmarks

    # Unpack the 8 conduction paths from the generated network
    (
        left_purkinje,  # Left ventricular Purkinje network
        right_purkinje,  # Right ventricular Purkinje network
        sa_av,  # SA to AV node pathway (anterior internodal tract)
        his_top,  # His bundle proximal segment (AV delay region)
        his_left,  # Left branch of His bundle
        his_right,  # Right branch of His bundle
        left_bundle,  # Left bundle branch
        right_bundle,  # Right bundle branch
    ) = beam_list

    # Create the Bachmann bundle
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    bachman_bundle = ConductionPath.create_from_keypoints(
        name=ConductionPathType.BACHMANN_BUNDLE,
        keypoints=[_landmarks.sa_node.xyz] + bachman_bundle_keypoints,
        id=9,
        base_mesh=pv.merge([model.left_atrium.epicardium, model.right_atrium.epicardium]),
        line_length=None,
        center=True,
    )
    bachman_bundle.add_pmj_path(list(range(1, bachman_bundle.mesh.n_points - 1, 4)))
    bachman_bundle.up_path = sa_av

    # The mid and post SA-AV node conduction paths are created by providing a list of keypoints.
    mid_sa_av = ConductionPath.create_from_keypoints(
        name=ConductionPathType.MID_SAN_AVN,
        keypoints=[_landmarks.sa_node.xyz]
        + mid_sa_av_keypoints
        + [_landmarks.av_node.xyz],
        id=10,
        base_mesh=model.right_atrium.endocardium,
        line_length=None,
        center=True,
    )
    mid_sa_av.add_pmj_path(list(range(5, mid_sa_av.mesh.n_points - 5, 4)))
    mid_sa_av.up_path = sa_av
    mid_sa_av.down_path = sa_av

    post_sa_av = ConductionPath.create_from_keypoints(
        name=ConductionPathType.POST_SAN_AVN,
        keypoints=[_landmarks.sa_node.xyz]
        + post_sa_av_keypoints
        + [_landmarks.av_node.xyz],
        id=11,
        base_mesh=model.right_atrium.endocardium,
        line_length=None,
        center=True,
    )
    post_sa_av.add_pmj_path(list(range(5, post_sa_av.mesh.n_points - 5, 4)))
    post_sa_av.up_path = sa_av
    post_sa_av.down_path = sa_av

    ###############################################################################
    # Create the Fascicle
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    left_anterio_fascile = ConductionPath.create_from_keypoints(
        name=ConductionPathType.LEFT_ANTERIOR_FASCILE,
        keypoints=[_landmarks.his_left_end_node.xyz, left_anterior_fascicle_endpoint],
        id=12,
        base_mesh=model.left_ventricle.endocardium,
        connection=None,
        line_length=None,
    )
    left_anterio_fascile.up_path = his_left
    left_anterio_fascile.down_path = left_purkinje

    left_posterior_fascile = ConductionPath.create_from_keypoints(
        name=ConductionPathType.LEFT_POSTERIOR_FASCICLE,
        keypoints=[_landmarks.his_left_end_node.xyz, left_posterior_fascicle_endpoint],
        id=13,
        base_mesh=model.left_ventricle.endocardium,
        connection=None,
        line_length=None,
    )
    left_posterior_fascile.up_path = his_left
    left_posterior_fascile.down_path = left_purkinje

    left_user_defined_fascicle = ConductionPath.create_from_keypoints(
        name=ConductionPathType.USER_PAHT_1,
        keypoints=[_landmarks.his_left_end_node.xyz, left_user_defined_fascicle_endpoint],   
        id=14,
        base_mesh=model.left_ventricle.endocardium,
        connection=None,
        line_length=None,
    )
    left_user_defined_fascicle.up_path = his_left
    left_user_defined_fascicle.down_path = left_purkinje

    # Addition of new conduction fibers to the model
    model.assign_conduction_paths(
        [
            left_purkinje,
            right_purkinje,
            sa_av,
            his_top,
            his_left,
            his_right,
            left_bundle,
            right_bundle,
            bachman_bundle,
            mid_sa_av,
            post_sa_av,
            left_anterio_fascile,
            left_posterior_fascile,
            left_user_defined_fascicle,
        ]
    )

def define_conduction_velocities_in_conduction_system(
        model, 
        atrial_beams_cv = 2.25, 
        his_top_cv = 0.1, 
        his_cv = 2.0, 
        purkinje_cv = 3.0,
        bundle_branches_cv = 1.786,
        lv_fascicles_traveltime = 28, #ms
        ):
    # Atrial conduction pathways (SA-AV node connections, Bachmann bundle)
    atrial_conduction_material = mat.ActiveBeam()
    atrial_conduction_material.sigma_fiber = (
        atrial_beams_cv  # High conductivity for fast atrial conduction
    )
    atrial_conduction_material.cell_model = cell_models.TentusscherEpi(gks=0.0392)

    model.conduction_paths[2].ep_material = atrial_conduction_material # sa_av
    model.conduction_paths[8].ep_material = atrial_conduction_material # bachman bundle
    model.conduction_paths[9].ep_material = atrial_conduction_material # mid sa - av 
    model.conduction_paths[10].ep_material = atrial_conduction_material # post sa - av

    # His bundle top segment (slow conduction for AV delay)
    his_top_material = mat.ActiveBeam()
    his_top_material.sigma_fiber = his_top_cv  # Low conductivity to create physiological AV delay
    his_top_material.cell_model = cell_models.TentusscherEpi(gks=0.0392)
    model.conduction_paths[3].ep_material = his_top_material

    # Purkinje fiber network (fast ventricular conduction)
    mat_purkinje = mat.ActiveBeam()
    mat_purkinje.sigma_fiber = purkinje_cv  # Fast conduction for rapid ventricular activation
    mat_purkinje.cell_model = cell_models.TentusscherEpi(gks=0.0392)

    model.conduction_paths[0].ep_material = mat_purkinje # Left purkinje
    model.conduction_paths[1].ep_material = mat_purkinje # Right purkinje

    # His_left & His_right
    mat_his = mat.ActiveBeam()
    mat_his.sigma_fiber = his_cv

    model.conduction_paths[4].ep_material = mat_his # His left
    model.conduction_paths[5].ep_material = mat_his # His right

    # Left bundle branch & right bundle branch
    mat_bundle_branches = mat.ActiveBeam()
    mat_bundle_branches.sigma_fiber = bundle_branches_cv

    model.conduction_paths[6].ep_material = mat_bundle_branches # Left bundle branch
    model.conduction_paths[7].ep_material = mat_bundle_branches # Right bundle branch

    # Left ventricle fascicles (LAF, LPF and user defined fascicle)
    traveltime = lv_fascicles_traveltime
    mat_left_anterio_fascile = mat.ActiveBeam()
    mat_left_anterio_fascile.sigma_fiber = model.conduction_paths[11].mesh.length/traveltime
    model.conduction_paths[11].ep_material = mat_left_anterio_fascile # LAF

    mat_left_post_fascile = mat.ActiveBeam()
    mat_left_post_fascile.sigma_fiber = model.conduction_paths[12].mesh.length/traveltime
    model.conduction_paths[12].ep_material = mat_left_post_fascile # LPF

    mat_left_user_defined_fascicle = mat.ActiveBeam()
    mat_left_user_defined_fascicle.sigma_fiber = model.conduction_paths[13].mesh.length/traveltime
    model.conduction_paths[13].ep_material = mat_left_user_defined_fascicle # User defined fascicle

    return

def rotate_heart_in_torso(heart_model, electrodes, angle_XY_plane = 0, angle_ZX_plane = 0, angle_YZ_plane = 0, angle_mitral_tricuspide=0) -> None:
    """Rotates eletrodes to simulate a different heart orientation in the torso."""
    model_centroid = np.array(heart_model.mesh.center_of_mass())
    model_tricuspid = np.array(
            next(cap.centroid for cap in heart_model.all_caps if cap.name == "tricuspid-valve"),
        )
    model_mitral = np.array(
        next(cap.centroid for cap in heart_model.all_caps if cap.name == "mitral-valve"),
    )

    # Rotation 1 in the ZX plane, rotation around Y axis
    Y_axis = np.array([0,1,0])  # Y axis
    rot_ZX_plane = R.from_rotvec( np.radians(angle_ZX_plane) * Y_axis)
    electrodes_rot1 = rot_ZX_plane.apply(electrodes - model_centroid) + model_centroid

    # Rotation 2 in the XY plane, rotation around Z axis
    Z_axis = np.array([0,0,1])  # Z axis
    rot_XY_plane = R.from_rotvec( np.radians(angle_XY_plane) * Z_axis)
    electrodes_rot2 = rot_XY_plane.apply(electrodes_rot1 - model_centroid) + model_centroid

    # Rotation 3 in the YZ plane, rotation around X axis
    X_axis = np.array([1,0,0])  # X axis
    rot_YZ_plane = R.from_rotvec( np.radians(angle_YZ_plane) * X_axis)
    electrodes_rot3 = rot_YZ_plane.apply(electrodes_rot2 - model_centroid) + model_centroid

    # Rotation around the mitral-tricuspide axis
    mitral_tricu_axis = (model_mitral - model_tricuspid)/np.linalg.norm(model_mitral - model_tricuspid)
    rot_mitral_tricu = R.from_rotvec( np.radians(angle_mitral_tricuspide) * mitral_tricu_axis)
    rotated_electrodes = rot_mitral_tricu.apply(electrodes_rot3 - model_mitral) + model_mitral

    # Instantiating Point objects for all electrodes and assign it
    names = ["V1", "V2", "V3", "V4", "V5", "V6", "RA", "LA", "RL", "LL"]
    heart_model.electrodes = [Point(name=n, xyz=xyz) for n, xyz in zip(names, rotated_electrodes)]

    return rotated_electrodes



def get_xyz_from_uhc(mesh, target_uhc_coords):
    """
    Finds the XYZ Cartesian coordinates of the node closest to a target UHC coordinate.
    
    Parameters
    ----------
    mesh : pyvista.UnstructuredGrid
        The mesh containing the computed UHC scalar fields.
    target_uhc_coords : list or tuple of float
        A 3-element sequence containing the target UHC coordinates in the following order:
        [transmural, apico-basal, rotational].
        - transmural: typically [0, 1], where 0 is endocardium and 1 is epicardium.
        - apico-basal: typically [0, 1], where 0 is apex and 1 is base.
        - rotational: radians.
        
    Returns
    -------
    list
        The [X, Y, Z] Cartesian coordinates of the closest valid node.
    """
    target_trans = target_uhc_coords[0]
    target_ab = target_uhc_coords[1]
    target_rot = target_uhc_coords[2]
    
    # Extract UHC scalar fields
    rho = mesh["transmural"]
    z = mesh["apico-basal"]
    theta = mesh["rotational"]
    
    # Ignore nodes without UHC (NaN values) to prevent calculation errors
    valid_mask = ~np.isnan(rho) & ~np.isnan(z) & ~np.isnan(theta)
    
    # Create the point cloud in the UHC parametric space.
    uhc_space = np.vstack((
        rho[valid_mask], 
        z[valid_mask], 
        theta[valid_mask]
    )).T
    
    target_uhc = np.array([target_trans, target_ab, target_rot])
    
    # Calculate Euclidean distances in the parametric space
    distances = np.linalg.norm(uhc_space - target_uhc, axis=1)
    
    # Retrieve the index of the closest valid node
    closest_valid_idx = np.argmin(distances)
    real_node_id = np.where(valid_mask)[0][closest_valid_idx]
    
    # Return the Cartesian coordinates [X, Y, Z]
    return mesh.points[real_node_id].tolist()


###############################################################################
# Mechanics: Windkessel System Model Settings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Values tuned for improved ejection fraction (Ca*2, Rp*0.5 from defaults).
system_model_settings = SystemModel(
    name="ConstantPreloadWindkesselAfterload",
    left_ventricle={
        "constants": {
            "Rv": Quantity(0.05, "mmHg*s/mL"),
            "Ra": Quantity(0.13, "mmHg*s/mL"),
            "Rp": Quantity(2.88, "mmHg*s/mL"),  # 5.76 * 0.5
            "Ca": Quantity(1.7, "mL/mmHg"),  # 0.85 * 2.0
            "Pven": Quantity(15, "mmHg"),
        },
        "initial_value": {"part": Quantity(70.0, "mmHg")},
    },
    right_ventricle={
        "constants": {
            "Rv": Quantity(0.025, "mmHg*s/mL"),  # 0.05 * 0.5
            "Ra": Quantity(0.0455, "mmHg*s/mL"),  # 0.13 * 0.35
            "Rp": Quantity(0.36, "mmHg*s/mL"),  # 5.76 * 0.125 * 0.5
            "Ca": Quantity(7.65, "mL/mmHg"),  # 0.85 * 4.5 * 2.0
            "Pven": Quantity(8, "mmHg"),
        },
        "initial_value": {"part": Quantity(15.0, "mmHg")},
    },
)


###############################################################################
# Mechanics: Myocardium Material Definition
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TODO ED pressure
# ventricle
myocardium = Mat295(
    rho=0.001,
    iso=ISO(itype=-3, beta=2, kappa=1.0, k1=0.00236, k2=1.75),
    aniso=ANISO(
        atype=-1,
        fibers=[HGOFiber(k1=0.00049, k2=9.01)],
    ),
    active=ACTIVE(
        model=ActiveModel3(ca2ion50=0.001, n=2, sigmax=0.125, l=1.9, eta=1.45),
        sf=1.5,
        sn=0.3,
        acthr=0.0002,
        ca2_curve=None,
    ),
)


###############################################################################
# Stiff Region Thresholds
# ~~~~~~~~~~~~~~~~~~~~~~~
# UVC apico-basal thresholds for creating passive stiff regions near valve planes.
threshold_left_ventricle = 0.95  # LV base stiffening threshold
threshold_right_ventricle = (
    0.98  # RV base stiffening threshold (higher due to geometry)
)

# Atrial stiff ring radius (mm) - for stabilizing atrial cap regions
atrial_stiff_ring_radius = 5

# Passive stiff material for valve plane regions and atrial rings
stiff_iso = Mat295(rho=0.001, iso=ISO(itype=-1, beta=2, kappa=10, mu1=0.1, alpha1=2))
