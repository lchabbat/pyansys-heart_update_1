"""
.. ep-simulator-realistic_ecg:

Generate Realistic ECGs with a Reaction-Eikonal Model
=====================================================
This example demonstrates a complete end-to-end workflow to parameterize, 
simulate, and post-process an Electrophysiology (EP) Reaction-Eikonal model 
to generate realistic Electrocardiograms (ECGs).

To achieve physiological accuracy, this script customizes:
- The fast conduction system (branches and Purkinje).
- Atrial and ventricular cellular models.
- Myocardial conduction velocities.
- Electrode positioning relative to the heart.

.. warning::
   **Prerequisite:** You must have the meshed Rodero2021 model 01 before 
   running this script. If you haven't done so, please run the 
   ``preprocess-fullheart.py`` example first to download and prepare 
   the geometry.

"""

###############################################################################
# Perform the required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import the required modules and set relevant paths, including that of the working
# directory, heart model, and LS-DYNA executable file.

import os
from pathlib import Path
import ansys.health.heart.models as models
from ansys.health.heart.simulator import DynaSettings, EPSimulator
import numpy as np
from ansys.health.heart.pre.conduction_path import ConductionPath, ConductionPathType
import ansys.health.heart.models_utils as model_utils
import pyvista as pv
import ansys.health.heart.models_utils as heart_model_utils
from ansys.health.heart.objects import Point
from scipy.spatial.transform import Rotation as R
from ansys.health.heart.landmarks import LandMarks
from ansys.health.heart.examples import get_preprocessed_fullheart
from ansys.health.heart.settings.settings import FibersDRBM
from ansys.health.heart.settings.material.ep_material import ActiveBeam, ActiveNew
import ansys.health.heart.settings.material.cell_models as cell_models
from ansys.health.heart.simulator import  run_lsdyna
from ansys.health.heart.post.dpf_utils import EPpostprocessor


# Set the working directory and path to the model you downloaded and meshed
workdir = Path.home() / "pyansys-heart" / "downloads" / "Rodero2021" / "01" / "FullHeart"
path_to_model, path_to_partinfo, _ = get_preprocessed_fullheart(resolution="2.0mm")

simulation_folder_name = 'your_simulation_folder'

plot = False

# Specify the LS-DYNA path. 
lsdyna_path = r"ls-dyna_msmpi.exe"

os.environ["ANSYS_DPF_ACCEPT_LA"] = "Y"

###############################################################################
# Load the full-heart model
model: models.FullHeart = models.HeartModel.load_model(
    path_to_model, path_to_partinfo, working_directory=workdir
)

# Save the model.
model.mesh.save(os.path.join(model.workdir, "simulation_model.vtu"))

###############################################################################
# Instantiate the simulator and define settings.
# ~~~~~~~~~~~~~~~~~~~~~~~~~

# Instantiate LS-DYNA settings.
dyna_settings = DynaSettings(
    lsdyna_path=lsdyna_path, dynatype="msmpi", num_cpus=8, platform="windows"
)

# Instantiate the simulator, modifying options as necessary.
simulator = EPSimulator(
    model=model,
    dyna_settings=dyna_settings,
    simulation_directory=os.path.join(workdir, simulation_folder_name),
)

###############################################################################
# Load simulation settings
# ~~~~~~~~~~~~~~~~~~~~~~~~
simulator.settings.load_defaults()

###############################################################################
# Define and position electrodes
electrodes = model.define_12lead_electrodes()

# Rotate the electrodes to simulate a different heart orientation in the torso. 
# The angles of rotation can be tuned to simulate different orientations.
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

rotated_electrodes = rotate_heart_in_torso(model, electrodes, angle_XY_plane=15, angle_mitral_tricuspide=10)

###############################################################################
# Plot the rotated electrodes and the heart to check the new orientation

if plot == True:
    heart_model = pv.read(str(workdir / f"heart_model.vtu"))
    plotter = pv.Plotter()
    point_cloud = pv.PolyData(rotated_electrodes)
    plotter.add_mesh(point_cloud, color='blue', point_size=10, render_points_as_spheres=True)
    plotter.add_mesh(heart_model, color='red', show_edges=False, opacity = 0.5)
    plotter.show()


###############################################################################
# Compute fiber orientation and plot the fibers on the entire model.

# Import the appendage landmarks for the reference Rodero model.
from ansys.health.heart.pre.database_utils import right_atrium_appendage_landmarks

# Get the right atrium appendage landmark of the first case of Rodero2021.
right_atrium_appendage_coordinates = right_atrium_appendage_landmarks.get("Rodero2021").get(1)

# Compute ventricular fibers.
simulator.compute_fibers(fiber_settings = FibersDRBM())

# Compute atrial fibers.
simulator.model.right_atrium.active = True
simulator.model.left_atrium.active = True
simulator.model.right_atrium.fiber = True
simulator.model.left_atrium.fiber = True
simulator.compute_left_atrial_fiber()
simulator.compute_right_atrial_fiber(appendage=right_atrium_appendage_coordinates)
if plot == True:
    simulator.model.plot_fibers(n_seed_points=1000)    

simulator.compute_uhc()

####################################################################################
#Compute Purkinje network 
directory = os.path.join(simulator.root_directory, "purkinjegeneration")
orig_num_cpus = simulator.dyna_settings.num_cpus
simulator.dyna_settings.num_cpus = 1

ureg = simulator.settings.purkinje.pmjtype._REGISTRY
simulator.settings.purkinje.pmjtype = 1 * ureg.dimensionless

#shorten the length of the purkinje branches
simulator.settings.purkinje.edgelen = 0.5 * ureg.dimensionless

#increase the density of the purkinje network
simulator.settings.purkinje.nsplit = 6 * ureg.dimensionless
simulator.settings.purkinje.nbrinit = 6 * ureg.dimensionless

# make sure the purkinje goes to the top of the ventricles 
simulator.settings.purkinje.ngen = 300 * ureg.dimensionless


simulator._write_purkinje_files(directory)
input_file = os.path.join(directory, "main.k")
simulator._run_dyna(input_file)
simulator.dyna_settings.num_cpus = orig_num_cpus

directory = os.path.join(simulator.root_directory, "purkinjegeneration")
print(os.path.join(simulator.root_directory, "purkinjegeneration"))


###############################################################################
# Compute the conduction system 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
beam_list, simulator.model._landmarks = heart_model_utils.define_full_conduction_system(
                simulator.model, os.path.join(simulator.root_directory, "purkinjegeneration")
            )

[   left_purkinje,
    right_purkinje,
    sa_av,
    his_top,
    his_left,
    his_right,
    left_bundle,
    right_bundle,
    ]   = beam_list

# Initialize landmarks to store anatomical points
landmarks = LandMarks()

sa = model_utils.define_sino_atrial_node(model, landmarks=landmarks, target_coord=[6, 66, 88])
av = model_utils.define_atrio_ventricular_node(model, landmarks=landmarks)

# Create the Bachmann bundle
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
surface_ids = [
    model.left_atrium.epicardium.id,
    model.right_atrium.epicardium.id,
]
surface = model.mesh.get_surface(surface_ids)
bachman_bundle = ConductionPath.create_from_keypoints(
    name=ConductionPathType.BACHMANN_BUNDLE,
    keypoints=[sa.xyz, [46, 102, 97]],
    id=9,
    base_mesh=surface,
    line_length=None,
    center=True,
)
bachman_bundle.add_pmj_path(list(range(1, bachman_bundle.mesh.n_points - 1, 4)))
bachman_bundle.up_path = sa_av

# The mid and post SA-AV node conduction paths are created by providing a list of keypoints.
mid_sa_av = ConductionPath.create_from_keypoints(
    name=ConductionPathType.MID_SAN_AVN,
    keypoints=[
        sa.xyz,
        [10, 79, 64],
        [18, 95, 41],
        [32, 93, 31],
        [43, 88, 26],
        av.xyz,
    ],
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
    keypoints=[sa.xyz, [2, 65, 53], [6, 73, 34], [25, 75, 26], av.xyz],
    id=11,
    base_mesh=model.right_atrium.endocardium,
    line_length=None,
    center=True,
)
post_sa_av.add_pmj_path(list(range(5, post_sa_av.mesh.n_points - 5, 4)))
post_sa_av.up_path = sa_av
post_sa_av.down_path = sa_av

###############################################################################
# Create the left anterior fascicle
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

left_anterio_fascile = ConductionPath.create_from_keypoints(
    name=ConductionPathType.LEFT_ANTERIOR_FASCILE,
    keypoints=[simulator.model._landmarks.his_left_end_node.xyz, [115, 91, 53]],  

    id=12,
    base_mesh=model.left_ventricle.endocardium,
    connection=None,
    line_length=None,
)
left_anterio_fascile.up_path = his_left
left_anterio_fascile.down_path = left_purkinje

###############################################################################
# Create the left posterior fascicle
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

left_posterior_fascile = ConductionPath.create_from_keypoints(
    name=ConductionPathType.LEFT_POSTERIOR_FASCICLE,
    keypoints=[simulator.model._landmarks.his_left_end_node.xyz, [124, 60, 32]],  
    id=13,
    base_mesh=model.left_ventricle.endocardium,
    connection=None,
    line_length=None,
)
left_posterior_fascile.up_path = his_left
left_posterior_fascile.down_path = left_purkinje



###############################################################################
# Create a bonus fascicle in the LV to adjust the activation sequence. Could be removed by tuning more precisely the endpoints of the posterior and anterior fascicles
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

left_user_defined_fascicle = ConductionPath.create_from_keypoints(
    name=ConductionPathType.USER_PAHT_1,
    keypoints=[simulator.model._landmarks.his_left_end_node.xyz, [82, 86, 42]],   
    id=14,
    base_mesh=model.left_ventricle.endocardium,
    connection=None,
    line_length=None,
)
left_user_defined_fascicle.up_path = his_left
left_user_defined_fascicle.down_path = left_purkinje

model.assign_conduction_paths([
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
        ])

if plot == True : 
    # Visualize the entire conduction system
    simulator.model.plot_purkinje()


###############################################################################
# Material parameters for the new parts of the conduction system

# Atria conduction path
mat_atrial_beams = ActiveBeam()
mat_atrial_beams.sigma_fiber = 2.25
simulator.model.conduction_paths[2].ep_material = mat_atrial_beams # sa_av
simulator.model.conduction_paths[8].ep_material = mat_atrial_beams # bachman bundle
simulator.model.conduction_paths[9].ep_material = mat_atrial_beams # mid sa - av 
simulator.model.conduction_paths[10].ep_material = mat_atrial_beams # post sa - av

# His_top
mat_his_top = ActiveBeam()
mat_his_top.sigma_fiber = 0.1 # Slow conduction velocity in the top part of the His bundle to simulate the pause in the AV node
simulator.model.conduction_paths[3].ep_material = mat_his_top

# His_left & His_right
mat_his = ActiveBeam()
mat_his.sigma_fiber = 2.0
simulator.model.conduction_paths[4].ep_material = mat_his # His left
simulator.model.conduction_paths[5].ep_material = mat_his # His right

# Left bundle branch & right bundle branch
mat_bundle_branches = ActiveBeam()
mat_bundle_branches.sigma_fiber = 1.786
simulator.model.conduction_paths[6].ep_material = mat_bundle_branches # Left bundle branch
simulator.model.conduction_paths[7].ep_material = mat_bundle_branches # Right bundle branch

# Fast conducting Purkinje fibers
mat_purkinje = ActiveBeam()
mat_purkinje.sigma_fiber = 3.0
simulator.model.conduction_paths[0].ep_material = mat_purkinje # Left purkinje
simulator.model.conduction_paths[1].ep_material = mat_purkinje # Right purkinje

# Left ventricle fascicles (LAF, LPF and user defined fascicle)
traveltime = 28
mat_left_anterio_fascile = ActiveBeam()
mat_left_anterio_fascile.sigma_fiber = simulator.model.conduction_paths[11].mesh.length/traveltime
simulator.model.conduction_paths[11].ep_material = mat_left_anterio_fascile # LAF

mat_left_post_fascile = ActiveBeam()
mat_left_post_fascile.sigma_fiber = simulator.model.conduction_paths[12].mesh.length/traveltime
simulator.model.conduction_paths[12].ep_material = mat_left_post_fascile # LPF

mat_left_user_defined_fascicle = ActiveBeam()
mat_left_user_defined_fascicle.sigma_fiber = simulator.model.conduction_paths[13].mesh.length/traveltime
simulator.model.conduction_paths[13].ep_material = mat_left_user_defined_fascicle # User defined fascicle


# Define ventricular and septal myocardium conduction velocities
mat_ventricles = ActiveNew()
mat_ventricles.sigma_fiber = 0.7
mat_ventricles.sigma_sheet = 0.35
mat_ventricles.sigma_sheet_normal = 0.18
mat_ventricles.cond_sigma_fiber = 0.17
mat_ventricles.cond_sigma_sheet = 0.08
mat_ventricles.cond_sigma_sheet_normal = 0.08

model.left_ventricle.ep_material = mat_ventricles
model.right_ventricle.ep_material = mat_ventricles
model.septum.ep_material = mat_ventricles

# Define atrial myocardium conduction velocities
mat_atria = ActiveNew()
mat_atria.sigma_fiber = 1.2
mat_atria.sigma_sheet = 0.6
mat_atria.sigma_sheet_normal = 0.3
mat_atria.cond_sigma_fiber = 0.17
mat_atria.cond_sigma_sheet = 0.08
mat_atria.cond_sigma_sheet_normal = 0.08

model.left_atrium.ep_material = mat_atria
model.right_atrium.ep_material = mat_atria


###############################################################################
# Define space varying cell model

###############################################################################
# Normalize the coordinate fields
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Extract and normalize the transmural (endo-epi) field to [0, 1]. 
# Needs to compute the Universal Ventricular Coordinates beforehand 

endo_epi = model.mesh["transmural"].copy()
endo_epi = (endo_epi - np.nanmin(endo_epi)) / (np.nanmax(endo_epi) - np.nanmin(endo_epi))


###############################################################################
# Define helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~
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

###############################################################################
# Define the binning and parameter ranges
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set the bin edges for the endo-epi and apex-base directions, and the
# extremal values of ``gks`` and ``gto`` used for linear interpolation.

endo_epi_bins = [0, 0.17, 0.41, 1]  # 0-0.17: endo, 0.17-0.41: mid, 0.41-1: epi

# Set bounds for gks and gto based on the range of values in Tentusscher cell models,
# which will be used for 2D linear interpolation.

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

###############################################################################
# Plot node groups
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Each group is rendered with a distinct color and labeled with its ``gks`` value.
if plot == True:
    colors = list(pv.colors.hexcolors.keys())[::10]
    p = pv.Plotter()
    model.mesh.set_active_scalars(None)
    p.add_mesh(model.mesh, opacity=0.2)
    for i in range(len(all_groups)):
        point_cloud = pv.PolyData(model.mesh.points[all_groups[i]])
        color = colors[i % len(colors)]
        p.add_mesh(
            point_cloud,
            color=color,
            label=f"gks={all_models[i].gks:.3f}",
            point_size=5,
            render_points_as_spheres=True,
        )
    p.add_legend()
    p.show()

# The cell model will be defined based on the nodeset
# This overwrites any existing cell model assignment based on part.
model._nodeset_cellmodel = (all_groups, all_models)

# Isolate the atria from the ventricles 
model._create_atrioventricular_isolation()

# Assign default materials for undefined parts
simulator._assign_default_materials()

# Write the simulation files
simulator.settings.electrophysiology.analysis.solvertype = "ReactionEikonal"
simulator._write_main_simulation_files(folder_name="main_ep_reaction_eikonal")


###############################################################################
# Run simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
path_to_input = str(workdir / simulation_folder_name / "main_ep_reaction_eikonal" / "main.k")

run_lsdyna(
    path_to_input=path_to_input,
    settings=dyna_settings,
    simulation_directory=workdir,
)


###############################################################################
# Post-process the results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Specify the path to the d3plot that contains the simulation results.

data_path = workdir / simulation_folder_name / "main_ep_reaction_eikonal" / "d3plot"

# Check if the file exists.
if not data_path.is_file():
    raise FileNotFoundError(f"File not found: {data_path}")

# Initialize the postprocessor.
post = EPpostprocessor(data_path)

###############################################################################
# Call methods to retrieve activation time
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Get activation time of the full field at the last time step.
activation_times = post.get_activation_times()
print(activation_times.data)

activation_times.plot(show_edges=False, show_scalar_bar=True)

###############################################################################
# Create a clip view.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Create a clip view of the activation time using ``pyvista``.
# Retrieve the unstructured grid.
grid: pv.UnstructuredGrid = post.reader.model.metadata.meshed_region.grid
grid.point_data["activation_time"] = activation_times.data
grid.set_active_scalars("activation_time")

# Clip the model and plot.
grid.clip(
    normal=[0.7785200198880087, -0.027403237199259987, 0.6270212446357586],
    origin=[88.24004990770091, 54.41149629465821, 49.1801566480857],
).plot(show_scalar_bar=True)

###############################################################################
# Read the ECGs and plot them.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

path_data_ECG = workdir / simulation_folder_name / "main_ep_reaction_eikonal" / "em_EKG_001.dat"
ecgs, times= post.read_ECGs(path_data_ECG)
post.compute_12_lead_ECGs(ecgs, times)
