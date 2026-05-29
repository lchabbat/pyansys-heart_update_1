"""
.. ep-simulator-realistic_ecg_using_baseline:

Generate Realistic ECGs with a Reaction-Eikonal Model 
=====================================================
This example generates realistic Electrocardiograms (ECGs) using the same method as ep-simulator-realistic_ecg.py 
The difference between these two scripts lies in the usage of a baseline to simplify steps. 
This script needs the file baseline_loup.py to work.

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
import pyvista as pv

from ansys.health.heart.examples import get_preprocessed_fullheart
from ansys.health.heart.settings.settings import FibersDRBM
from ansys.health.heart.simulator import  run_lsdyna
from ansys.health.heart.post.dpf_utils import EPpostprocessor

from baseline_loup import (
    define_conduction_system,
    define_transmurally_variying_cell_model,
    rotate_heart_in_torso,
    define_conduction_velocities_in_conduction_system,
    get_xyz_from_uhc,
    ep_atrium,
    ep_vent,
    purkinje_settings,
    default_laf_endpoint_uhc,
    default_lpf_endpoint_uhc,
    default_user_fascicle_endpoint_uhc,
)

# Set the working directory and path to the model you downloaded and meshed
workdir = Path.home() / "pyansys-heart" / "downloads" / "Rodero2021" / "01" / "FullHeart"
path_to_model, path_to_partinfo, _ = get_preprocessed_fullheart(resolution="2.0mm")

simulation_folder_name = 'your_simulation_folder'

plot = False

# Specify the LS-DYNA path. 
lsdyna_path = r"C:/Users/Raphael/lchabbat/ls-dyna_mpp_d_DEV_122687-g00479ad686_winx64_ifort190_sse2_msmpi/ls-dyna_mpp_d_DEV_122687-g00479ad686_winx64_ifort190_sse2_msmpi.exe"

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

simulator.settings.purkinje = purkinje_settings
simulator.compute_purkinje()

####################################################################################
# Define the full conduction system
define_conduction_system(
    simulator.model,
    os.path.join(simulator.root_directory, "purkinjegeneration"),
    bachman_bundle_keypoints=[[46, 102, 97]],
    mid_sa_av_keypoints=[[10, 79, 64], [18, 95, 41], [32, 93, 31], [43, 88, 26]],
    post_sa_av_keypoints=[[2, 65, 53], [6, 73, 34], [25, 75, 26]],
    left_anterior_fascicle_endpoint=get_xyz_from_uhc(simulator.model.mesh, default_laf_endpoint_uhc),
    left_posterior_fascicle_endpoint=get_xyz_from_uhc(simulator.model.mesh, default_lpf_endpoint_uhc),
    left_user_defined_fascicle_endpoint=get_xyz_from_uhc(simulator.model.mesh, default_user_fascicle_endpoint_uhc),
)

if plot == True : 
    # Visualize the entire conduction system
    simulator.model.plot_purkinje()

###############################################################################
# Assign conduction velocities to the elements of the conduction system
define_conduction_velocities_in_conduction_system(simulator.model)

###############################################################################
# Define space varying cell model

###############################################################################
# Normalize the coordinate fields
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Extract and normalize the transmural (endo-epi) field to [0, 1]. 
# Needs to compute the Universal Ventricular Coordinates beforehand 

endo_epi = model.mesh["transmural"].copy()
endo_epi = (endo_epi - np.nanmin(endo_epi)) / (np.nanmax(endo_epi) - np.nanmin(endo_epi))

# The cell model will be defined based on the nodeset
# This overwrites any existing cell model assignment based on part.
all_groups, all_models = define_transmurally_variying_cell_model(simulator.model, endo_epi)


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

############################################################
# Assign material properties to the ventricles and atria
simulator.model.left_ventricle.ep_material = ep_vent
simulator.model.right_ventricle.ep_material = ep_vent
simulator.model.septum.ep_material = ep_vent

simulator.model.right_atrium.ep_material = ep_atrium
simulator.model.left_atrium.ep_material = ep_atrium


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
