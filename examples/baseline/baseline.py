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

###############################################################################
# Electrophysiology (EP) Material Definitions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define EP materials for conduction system components and myocardial tissue.
# Conductivity values (sigma) are in mS/mm.

# Atrial conduction pathways (SA-AV node connections, Bachmann bundle)
atrial_conduction_material = mat.ActiveBeam()
atrial_conduction_material.sigma_fiber = (
    2.25  # High conductivity for fast atrial conduction
)
atrial_conduction_material.cell_model = cell_models.TentusscherEpi(gks=0.0392)

# His bundle top segment (slow conduction for AV delay)
his_top_material = mat.ActiveBeam()
his_top_material.sigma_fiber = 0.1  # Low conductivity to create physiological AV delay
his_top_material.cell_model = cell_models.TentusscherEpi(gks=0.0392)

# Purkinje fiber network (fast ventricular conduction)
mat_purkinje = mat.ActiveBeam()
mat_purkinje.sigma_fiber = 2.0  # Fast conduction for rapid ventricular activation
mat_purkinje.cell_model = cell_models.TentusscherEpi(gks=0.0392)

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
# Space-Varying Cell Model Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Utilities for assigning heterogeneous cell models based on spatial coordinates.


def define_node_group(selected_nodes, coord1, coord2, bins1=[0, 1], bins2=[0, 1]):
    """Define node groups based on two coordinates and their respective bins.

    Parameters
    ----------
    selected_nodes : array-like
        Indices of the selected nodes.
    coord1 : array-like
        First coordinate values for the nodes.
    coord2 : array-like
        Second coordinate values for the nodes.
    bins1 : list, default: [0, 1]
        Bin edges for the first coordinate.
    bins2 : list, default: [0, 1]
        Bin edges for the second coordinate.

    Returns
    -------
    list of list of np.ndarray
        Node groups organized by the bins of coord1 and coord2.
    """
    masked_coord1 = coord1.copy()
    masked_coord2 = coord2.copy()
    # set nan for the rest so that they will not be included in the group definition
    masked_coord1[~np.isin(np.arange(len(masked_coord1)), selected_nodes)] = np.nan
    masked_coord2[~np.isin(np.arange(len(masked_coord2)), selected_nodes)] = np.nan

    node_groups = [[None] * (len(bins2) - 1) for _ in range(len(bins1) - 1)]

    for i in range(len(bins1) - 1):
        for j in range(len(bins2) - 1):
            mask = (
                (masked_coord1 >= bins1[i])
                & (masked_coord1 <= bins1[i + 1])
                & (masked_coord2 >= bins2[j])
                & (masked_coord2 <= bins2[j + 1])
            )
            node_groups[i][j] = np.where(mask)[0]

    return node_groups


# Space-varying cell model functions
# TODO extract parameters
def define_space_varying_cell_model_maeyls(model, endo_epi, apex_base):
    """Define space-varying cell models for ventricles and septum.

    Parameters
    ----------
    model : FullHeart
        The heart model.
    endo_epi : array-like
        Transmural coordinate values (0=endo, 1=epi).
    apex_base : array-like
        Apico-basal coordinate values (0=apex, 1=base).

    Returns
    -------
    tuple
        (node_groups_flat, cell_models_flat) lists for assignment.
    """
    # Define the binning and parameter ranges
    endo_epi_bins = np.linspace(0, 1, 5)
    apex_base_bins = [0, 0.5, 0.7, 0.8, 0.9, 1]  # apex to base

    gks_min = 0.062  # extreme value at endo apex
    gks_max = 0.245  # extreme value at epi base
    gto_min = 0.073  # extreme value at endo apex
    gto_max = 0.294  # extreme value at epi base

    # Assign cell models to the ventricles
    lv_nodes = model.left_ventricle.get_node_ids(model.mesh)
    rv_nodes = model.right_ventricle.get_node_ids(model.mesh)
    ventricle_nodes = np.unique(np.concatenate((lv_nodes, rv_nodes)))

    ventricle_node_groups = define_node_group(
        ventricle_nodes, endo_epi, apex_base, endo_epi_bins, apex_base_bins
    )

    n_row, n_col = len(endo_epi_bins) - 1, len(apex_base_bins) - 1
    factors = np.arange(n_row * n_col).reshape(n_row, n_col) / (n_row * n_col - 1)

    gks_matrix = gks_min + factors * (gks_max - gks_min)
    gto_matrix = gto_min + factors * (gto_max - gto_min)

    ventricle_cell_models = [
        [
            cell_models.Tentusscher(gks=gks_matrix[i, j], gto=gto_matrix[i, j])
            for j in range(n_col)
        ]
        for i in range(n_row)
    ]

    ventricle_groups_flat = [x for sublist in ventricle_node_groups for x in sublist]
    ventricle_models_flat = [x for sublist in ventricle_cell_models for x in sublist]

    # Assign cell models to the septum
    apex_base_bins = [0, 1]
    septum_nodes = model.septum.get_node_ids(model.mesh)
    septum_node_groups = define_node_group(septum_nodes, endo_epi, apex_base)

    n_row, n_col = 1, len(apex_base_bins) - 1
    factors = np.arange(n_row * n_col).reshape(n_row, n_col) / (n_row * n_col - 1)
    factors = (
        np.array([[0]]) if n_row * n_col == 1 else factors
    )  # handle the case with only one group

    gks_matrix = gks_min + factors * (gks_max - gks_min)
    gto_matrix = gto_min + factors * (gto_max - gto_min)

    septum_cell_models = [
        [
            cell_models.Tentusscher(gks=gks_matrix[i, j], gto=gto_matrix[i, j])
            for j in range(n_col)
        ]
        for i in range(n_row)
    ]
    septum_groups_flat = [x for sublist in septum_node_groups for x in sublist]
    septum_models_flat = [x for sublist in septum_cell_models for x in sublist]

    # Merge and assign all cell models
    all_groups_flat = ventricle_groups_flat + septum_groups_flat
    all_models_flat = ventricle_models_flat + septum_models_flat

    return all_groups_flat, all_models_flat

def define_space_varying_cell_model(model, endo_epi, apex_base):

    ###############################################################################
    # Define the binning and parameter ranges
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set the bin edges for the endo-epi and apex-base directions, and the
    # extremal values of ``gks`` and ``gto`` used for linear interpolation.

    endo_epi_bins = [0, 0.3, 0.7, 1]
    apex_base_bins = [
        0,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1,
    ]  # apex to base
    vec = np.linspace(0.75, 1.25, len(apex_base_bins) - 1)
    ###############################################################################
    # Assign cell models to the ventricles
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Group left-ventricle nodes by endo-epi and apex-base bins, then
    # interpolate ``gks`` and ``gto`` linearly across the 2-D grid.

    lv_nodes = model.left_ventricle.get_node_ids(model.mesh)

    ventricle_node_groups = define_node_group(
        lv_nodes, endo_epi, apex_base, endo_epi_bins, apex_base_bins
    )
    # ventricle_node_groups 
    gks = cell_models.Tentusscher().gks
    gks_matrix = gks * np.outer(vec, np.array([0.5, 0.75, 1])) # endo/mid/epi
    ventricle_cell_models = [
        [
            cell_models.Tentusscher(gks=gks_matrix[i, j])
            for i in range(gks_matrix.shape[0])
        ]
        for j in range(gks_matrix.shape[1])
    ]

    ventricle_groups_flat = [x for sublist in ventricle_node_groups for x in sublist]
    ventricle_models_flat = [x for sublist in ventricle_cell_models for x in sublist]

    ###############################################################################
    # Assign cell models to the septum
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # For the septum, only the apex-base direction is used for variation.
    # Parameters will be interpolated from endo values.
    def sep_or_rv(nodes, gks_matrix_septum):

        septum_node_groups = define_node_group(
            nodes, endo_epi, apex_base, bins2=apex_base_bins
        )

        septum_cell_models = [
            [
                cell_models.Tentusscher(gks=gks_matrix_septum[j])
                for j in range(gks_matrix_septum.shape[0])
            ]
            for i in range(len(septum_node_groups))
        ]
        septum_groups_flat = [x for sublist in septum_node_groups for x in sublist]
        septum_models_flat = [x for sublist in septum_cell_models for x in sublist]

        return septum_groups_flat, septum_models_flat

    gks_matrix_septum = gks * vec * 0.5  # same as endo
    septum_nodes = model.septum.get_node_ids(model.mesh)
    septum_groups_flat, septum_models_flat = sep_or_rv(septum_nodes, gks_matrix_septum)

    gks_matrix_rv = gks * vec  # same as epi
    rv_nodes = model.right_ventricle.get_node_ids(model.mesh)
    rv_groups_flat, rv_models_flat = sep_or_rv(rv_nodes, gks_matrix_rv)

    ###############################################################################
    # Merge and assign all cell models
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Combine the ventricle and septum groups and models into flat lists ready
    # for assignment to the heart model.

    all_groups_flat = ventricle_groups_flat + septum_groups_flat + rv_groups_flat
    all_models_flat = ventricle_models_flat + septum_models_flat + rv_models_flat

    return all_groups_flat, all_models_flat


def define_conduction_system(
    model: models.HeartModel,
    purkinje_path: str,
    bachman_bundle_keypoints: list,
    mid_sa_av_keypoints: list,
    post_sa_av_keypoints: list,
    left_anterior_fascicle_endpoint: list,
    left_posterior_fascicle_endpoint: list,
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

    # assign materials from top to bottom of the conduction system
    sa_av.ep_material = atrial_conduction_material
    mid_sa_av.ep_material = atrial_conduction_material
    post_sa_av.ep_material = atrial_conduction_material
    bachman_bundle.ep_material = atrial_conduction_material

    his_top.ep_material = his_top_material

    left_purkinje.ep_material = mat_purkinje
    right_purkinje.ep_material = mat_purkinje
    his_left.ep_material = mat_purkinje
    his_right.ep_material = mat_purkinje
    left_bundle.ep_material = mat_purkinje
    right_bundle.ep_material = mat_purkinje
    left_anterio_fascile.ep_material = mat_purkinje
    left_posterior_fascile.ep_material = mat_purkinje

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
        ]
    )


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
