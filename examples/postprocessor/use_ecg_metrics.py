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

Postprocess a Reaction-Eikonal model.
-------------------------------------
This example shows how to postprocess a full heart reaction eikonal model.
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
from ecg_metrics import EcgMetrics


os.environ["ANSYS_DPF_ACCEPT_LA"] = "Y"

###############################################################################
# Create a postprocessor object
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

###############################################################################
# .. note::
#    This example assumes that you have you ran a full heart electrophysiology simulation
#    and that the d3plot files are located in ``data_path``.

# Import the required modules and set relevant paths.
# workdir = Path.home() / "pyansys-heart" / "downloads" / "Rodero2021" / "01" / "FullHeart"
workdir = Path.home() / "test_pri_mesh_2mm" / "Rodero2021" / "01" / "FullHeart"


# Specify the path to the d3plot that contains the simulation results.
# data_path = workdir / "simulation-EP" / "main-ep-ReactionEikonal" / "d3plot"
simulation_folder_name = 'test_example3'
data_path = workdir / simulation_folder_name / "main_ep_reaction_eikonal" / "d3plot"

# Check if the file exists.
if not data_path.is_file():
    raise FileNotFoundError(f"File not found: {data_path}")

# Initialize the postprocessor.
post = EPpostprocessor(data_path)

###############################################################################
# Load the full-heart model
path_to_model, path_to_partinfo, _ = get_preprocessed_fullheart(resolution="2.0mm")

model: models.FullHeart = models.HeartModel.load_model(
    path_to_model, path_to_partinfo, working_directory=workdir
)

# Save the model.
model.mesh.save(os.path.join(model.workdir, "simulation_model.vtu"))

# ###############################################################################
# # Call methods to retrieve activation time
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # Get activation time of the full field at the last time step.
# activation_times = post.get_activation_times()
 
metrics = EcgMetrics(model=model, post=post)

print("QT-Interval is ",metrics.qt_interval, " ms long")
print("QRS is ",metrics.qrs_duration, " ms long")
print("PQ interval is ", metrics.pq_interval, " ms long")
print("P wave duration is ",metrics.p_wave_duration, " ms long")
