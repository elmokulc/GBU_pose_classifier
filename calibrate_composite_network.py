import json
import time 
import numpy as np
from cv2 import aruco

import gbu

################################################################################
# SETUP
print("#SETTING UP DATASET")
image_directory = "./dataset14_subsample/"

metadata = json.load(open(image_directory + "metadata.json"))
camera_matrix = np.array(metadata["camera_matrix"])
distortion_coefficients = np.array(metadata["distortion_coefficients"])
md = float(metadata["aruco_marker_size"].split()[0])
marker_dimension = {i: md for i in range(1, 250)}
marker_size = 12.4079e-3
marker_dimension = {i: marker_size for i in range(1, 250)}

# Aruco settings
parameters = aruco.DetectorParameters_create()
parameters.cornerRefinementMethod = 3
parameters.cornerRefinementWinSize = 5
parameters.cornerRefinementMaxIterations = 100

# STEP 1: PREPROCESSING
print("# BATCH INTITIALIZATION")
batch = gbu.calibration.ImageBatchCalibration(aruco_dict=aruco.DICT_6X6_250,
                                              parameters=parameters,
                                              marker_dimension=marker_dimension,
                                              output_directory="./_outputs_calibration/",
                                              camera_matrix=camera_matrix,
                                              distortion_coefficients=distortion_coefficients)

batch.load_image_batch(directory=image_directory)
batch.detect_markers(plot_markers=False, enforce=True)
batch.estimate_pose()

print("Number of poses : {0}".format(len(batch.data_pose)))
# STEP 2: GRAPH THEORY & PRE OPTIMIZATION EDUCATED GUESS
print("#CREATING GRAPH")
batch.get_graph_data(plot_markers=False, criterion=0.10, alpha_criterion=1.5)
batch.graph_calibration()

# STEP 3: LEAST SQUARE OPTIMIZATION
print("#OPTIMIZING")
t0 = time.time()
compo, poseBatch, sol = batch.optimize_calibration()
t1 = time.time()
print("=> Ran optimization in {0}s for {1} poses".format(
    (t1 - t0), len(batch.data_graph)))

# STEP 4 : PLOT RESULTS
batch.plot_reprojection_errors(
    global_plot=True,
    individual_plots=False,
    plot_type="classic")

# STEP 5 : EXPORTING AS JSON FILE
compo = batch.composites[-1]
compo.save(path="./vpcm.json")

