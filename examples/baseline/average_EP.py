import os
from pathlib import Path

from baseline import (
    define_conduction_system,
    define_space_varying_cell_model,
    ep_atrium,
    ep_vent,
    purkinje_settings,
    stiff_iso,
    threshold_left_ventricle,
    threshold_right_ventricle,
)

import ansys.health.heart.models as models
from ansys.health.heart.settings.settings import FibersDRBM
from ansys.health.heart.simulator import DynaSettings, EPSimulator

###############################################################################
workdir = r"D:\average\1.5mm"

path_to_model = os.path.join(workdir, "heart_model.vtu")
path_to_partinfo = os.path.join(workdir, "heart_model.partinfo.json")

# Set the working directory.
workdir = str(Path(path_to_model).parent / "ep_baseline")

###############################################################################
# Load the full-heart model
# ~~~~~~~~~~~~~~~~~~~~~~~~~

# Load the full-heart model.
model = models.HeartModel.load_model(
    path_to_model, path_to_partinfo, working_directory=workdir
)
model.define_12lead_electrodes()

lsdyna_path = r"D:\ansys_heart\LSDYNA\R16.1\ls-dyna_mpp_d_R16.1.1_20-g0c90cad538_winx64_ifort190_avx2_impi2019.exe"
dyna_settings = DynaSettings(
    lsdyna_path=lsdyna_path,
    dynatype="intelmpi",
    platform="windows",
    num_cpus=6,
    mpi_options="-localonly",
)

# Instantiate the simulator.
simulator = EPSimulator(
    model=model,
    dyna_settings=dyna_settings,
    simulation_directory=os.path.join(workdir, "EP"),
)

# Load default simulation settings.
simulator.settings.load_defaults()

# Use the ReactionEikonal solver for the electrophysiology simulation.
simulator.settings.electrophysiology.analysis.solvertype = "ReactionEikonal"

###############################################################################
# Compute fiber orientation in the ventricles and atria.
simulator.compute_fibers(FibersDRBM())
simulator.compute_left_atrial_fiber()
right_atrium_appendage_coordinates = [-30.1868, -33.2404, 40.7314]
simulator.compute_right_atrial_fiber(
    appendage=right_atrium_appendage_coordinates,
    top=[
        [-50.2706, 9.98426, 42.1868],
        [-56.0671, 25.0901, -1.80736],
        [-59.8343, 17.5185, 22.3063],
    ],
)

# Switch the atria to active.
simulator.model.left_atrium.fiber = True
simulator.model.left_atrium.active = True

simulator.model.right_atrium.fiber = True
simulator.model.right_atrium.active = True

###############################################################################
# Compute UHCs (Universal Heart Coordinates).
simulator.compute_uhc()

# Define cell model for each layer and assign to the corresponding nodesets.
node_set_list, cell_model_list = define_space_varying_cell_model(
    model,
    simulator.model.mesh["transmural"],
    simulator.model.mesh["apico-basal"],
)

model._nodeset_cellmodel = (node_set_list, cell_model_list)

# Extract elements around atrial caps and assign as a passive material.
ring = simulator.model.create_atrial_stiff_ring(radius=5)
ring.ep_material = ep_atrium

# Extract elements close to the valves and assign these a passive material.
base = simulator.model.create_stiff_ventricle_base(
    threshold_left_ventricle=threshold_left_ventricle,
    threshold_right_ventricle=threshold_right_ventricle,
    stiff_material=stiff_iso,
)
base.ep_material = ep_vent

simulator.settings.purkinje = purkinje_settings
simulator.compute_purkinje()

# # Use landmarks to compute the rest of the conduction system.
# simulator.compute_conduction_system()
# # TODO add bachman, fascle ...
# simulator.model.conduction_paths[0].ep_material = mat_purkinje
# simulator.model.conduction_paths[1].ep_material = mat_purkinje
# simulator.model.conduction_paths[2].ep_material = atrial_conduction_material
# simulator.model.conduction_paths[3].ep_material = his_top_material
# simulator.model.conduction_paths[4].ep_material = mat_purkinje
# simulator.model.conduction_paths[5].ep_material = mat_purkinje
# simulator.model.conduction_paths[6].ep_material = mat_purkinje
# simulator.model.conduction_paths[7].ep_material = mat_purkinje
define_conduction_system(
    simulator.model,
    os.path.join(simulator.root_directory, "purkinjegeneration"),
    bachman_bundle_keypoints=[[-46, 27, 30], [-47, 32, 40], [-17, 50, 40]],
    mid_sa_av_keypoints=[[-58, 4, 40], [-62, -15, 34], [-53, -30, 19]],
    post_sa_av_keypoints=[[-45, 29, 30], [-64, 14, 3], [-58, -13, -11]],
    left_anterior_fascicle_endpoint=[38, 34, 10],
    left_posterior_fascicle_endpoint=[58, 3, -15],
)

simulator.model.parts[0].ep_material = ep_vent
simulator.model.parts[1].ep_material = ep_vent
simulator.model.parts[2].ep_material = ep_vent

simulator.model.parts[3].ep_material = ep_atrium
simulator.model.parts[4].ep_material = ep_atrium
###############################################################################
# Start the main simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~
from pint import Quantity

simulator.settings.electrophysiology.analysis.dt_d3plot = Quantity(40, "ms")
simulator.simulate(folder_name="fullconduction")
