# Filename: <utils.py>
# Copyright (C) <2021> Authors: <Pierre Vacher, Ludovic Charleux, Emile Roux, Christian Elmo Kulanesan>
#
# This program is free software: you can redistribute it and / or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 2 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
################################################################################
# GBU/UTILS
################################################################################

################################################################################
# IMPORTS
import itertools
import json
import os
import pickle
import random
import warnings

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import more_itertools
import networkx as nx
import numpy as np
import pandas as pd
from cv2 import aruco
from skimage import io
from tqdm import tqdm

import gbu

warnings.filterwarnings('ignore')


################################################################################
# CAMERA
def read_camera_calibration(path):
    """
    Reads a JSON file containing a camera calibration file.

    Inputs:
    * path: path to a JSON file.

    Outputs:
    * camera_matrix: a 3x3 float numpy.array instance.
    * distortion_coefficients: a float numpy.array instance.
    """
    data = json.load(open(path))
    return (np.array(data["camera_matrix"]), np.array(data["distortion_coefficients"]))


def save_camera_calibration(path, camera_matrix, distortion_coefficients):
    """
    Saves a JSON file containing a camera calibration file.

    Inputs:
    * path: path to a JSON file.
    * camera_matrix: a 3x3 float numpy.array instance.
    * distortion_coefficients: a float numpy.array instance.

    Ouputs: None
    """
    C = np.array(camera_matrix).tolist()
    D = np.array(distortion_coefficients).tolist()
    data = {"camera_matrix": C, "distortion_coefficients": D}
    with open(path, "w") as f:
        json.dump(data, f)

################################################################################

################################################################################
# PROJECTIVE GEOMETRY


def invert_RT(R, T):
    """
    Inverts a RT transformation
    """
    Ti = -cv2.Rodrigues(-R)[0].dot(T)
    return -R, Ti


def compose_RT(R1, T1, R2, T2):
    """
    Composed 2 RT transformations
    """
    R, T = cv2.composeRT(R1, T1, R2, T2)[:2]
    return R.flatten(), T.flatten()


def change_reference(R10, T10, R20, T20):
    """
    Change the reference frame in which RT 2 transformation is expressed. The
    new reference frame is 1.
    Return R21,T21
    """
    R01, T01 = invert_RT(R10, T10)
    return compose_RT(R20, T20, R01, T01)


def apply_RT(P, R, T):
    """
    Applies RT transformation to 3D points P.
    """
    P = cv2.Rodrigues(R)[0].dot(P.T).T
    P += T.T
    return P


################################################################################


################################################################################
# MARKER REPRESENTATION
_XYZ = np.array(
    [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    ],
    dtype=np.float32,
)
_RGB = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
_VERTS = np.float32(
    [
        [-0.5, -0.5, 0],
        [-0.5, 0.5, 0],
        [0.5, 0.5, 0],
        [0.5, -0.5, 0],
        [-0.5, -0.5, -1],
        [-0.5, 0.5, -1],
        [0.5, 0.5, -1],
        [0.5, -0.5, -1],
    ]
)


def _draw_axis(
    im,
    axis,
    rvec,
    tvec,
    camera_matrix=np.eye(3),
    distortion_coefficients=np.zeros(5),
    color=(255, 0, 0),
    thickness=2,
):
    axproj, jac = cv2.projectPoints(
        axis, rvec, tvec, camera_matrix, distortion_coefficients
    )
    axproj = np.int32(axproj).reshape(-1, 2)
    im = cv2.arrowedLine(
        im,
        pt1=tuple(axproj[0]),
        pt2=tuple(axproj[1]),
        color=color,
        thickness=int(thickness),
    )
    return im


def draw_markers(ids2D, corners2D, imout, rejected=None):
    """
    Draw detected markers on an image.

    Inputs:
    * ids2D: (len K array_like) marker ids as produced by detect_markers.
    * corners2D: corners as detected by detect_markers.
    * imout: (NxMx3 array-like) image used to write markers.
    * rejected: if None, does nothing. If boolean array-like with len K, False will plot markers normally
    and True will plot markers with a cross as deactivated.
    """
    Nm = len(ids2D)
    i2D = ids2D.reshape(Nm, 1)
    c2D = corners2D.reshape(Nm, 1, 4, 2)
    aruco.drawDetectedMarkers(imout, c2D, i2D)
    if rejected is not None:
        cross = np.array([[0, 2], [1, 3], [0, 1], [1, 2], [2, 3], [3, 0]])
        pts = corners2D[rejected][:, cross].round()
        pts = pts.astype(np.int32).reshape(-1, 1, 2, 2)
        imout = cv2.polylines(imout, pts, False, (255, 0, 0), 1)

    return imout


def draw_xyz(
    im,
    rvec,
    tvec,
    dimension=1.0,
    camera_matrix=np.eye(3),
    distortion_coefficients=np.zeros(5),
    thickness=2,
):
    """
    Draws the XYZ frame on an image.
    """
    for i in range(3):
        im = _draw_axis(
            im,
            axis=_XYZ[i] * dimension / 2.0,
            rvec=rvec,
            tvec=tvec,
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            color=_RGB[i],
            thickness=thickness,
        )
    return im


def checkdir(directory="./outputs/"):
    """
    Creates dir not exists.

    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def create_image_dataFrame():
    # image dataframe
    cols = pd.MultiIndex.from_tuples([
        ("image", "path", ""),
        ("image", "fname", ""), ])

    data_img = pd.DataFrame(columns=cols)
    data_img.index.name = "images"
    return data_img


def create_detect_dataFrame():
    # detect dataframe
    cols = pd.MultiIndex.from_tuples([
        ("images", "", ""),
        ("markers", "label", ""),
        ("markers", "dimension", ""),
        ("markers", "area", ""),
        ("method", "", ""),
        ("p2d", "c0", "x"),
        ("p2d", "c0", "y"),
        ("p2d", "c1", "x"),
        ("p2d", "c1", "y"),
        ("p2d", "c2", "x"),
        ("p2d", "c2", "y"),
        ("p2d", "c3", "x"),
        ("p2d", "c3", "y"), ])

    data_detect = pd.DataFrame(columns=cols)
    data_detect.index.name = "detects"
    return data_detect


def create_pose_dataFrame():
    # pose dataframe
    cols = pd.MultiIndex.from_tuples([
        ("detects", "", ""),
        ("rvec", "x", ""),
        ("rvec", "y", ""),
        ("rvec", "z", ""),
        ("tvec", "x", ""),
        ("tvec", "y", ""),
        ("tvec", "z", ""),
        ("flag", "graph", "processed"),
        ("flag", "graph", "validated"),
        ("flag", "graph", "rejected"),
        ("flag", "pose", ""),
        ("occurency", "good", ""),
        ("occurency", "bad", "")])

    data_pose = pd.DataFrame(columns=cols)
    data_pose.index.name = "poses"
    return data_pose


def dump_dataframe_HDF(img_dir="./",
                       meta_filename='metadata.h5',
                       key=None,
                       dataframe=None):

    dataframe.to_hdf(img_dir + meta_filename, key=key, mode='w')


def cherry_dataframe_HDF(img_dir="./",
                         meta_filename='metadata.h5',
                         key=None):

    return pd.read_hdf(img_dir + meta_filename, key)


def load_image_batch(data_img=None, directory="./", format=".tif", slice=None):
    """
    Get path of every images in a given directory
    store and/or return as dataframe

    Inputs:
    * directory:(string) path of the directory to check
    * image:(string) format to check
    * slice: (int) downsampling number of image by slice value

    Return :
    * data_img:(pandasDataframe)
    """
    if data_img is None:
        data_img = create_image_dataFrame()
    data_img_temp = create_image_dataFrame()

    im_fnames = sorted(
        [f for f in os.listdir(directory) if f.endswith(format)])

    if slice is not None:
        im_fnames = im_fnames[::slice]

    Ni = len(im_fnames)
    data_img_temp[("image", "path", "")] = [directory for i in range(Ni)]
    data_img_temp[("image", "fname", "")] = im_fnames

    data_img = pd.concat([data_img, data_img_temp]).reset_index(
        drop=True).drop_duplicates()
    data_img.index.name = "images"
    return data_img


def detect_markers(parameters=None,
                   marker_dimension=None,
                   aruco_dict=aruco.DICT_6X6_250,
                   output_directory="/_outputs/",
                   data_img=None,
                   data_detect=None,
                   plot_markers=False,
                   plot_image_type="jpg"):
    """
    Detects markers from data_img. Output is stored in
    Inputs:
    * plot_markers:(bool)
    Return :
    * data_detect:(pandasDataframe)
    """
    if data_detect is None:
        data_detect = create_detect_dataFrame()
    data_detect_temp = create_detect_dataFrame()

    if parameters is None:
        parameters = aruco.DetectorParameters_create()
    if marker_dimension is None:
        marker_dimension = {m: 1.0 for m in range(1, 1001)}

    pbar = tqdm(range(len(data_img)))
    cpt = 0
    for ii in pbar:
        path = data_img.image.path.loc[ii]
        fname = data_img.image.fname.loc[ii]
        pbar.set_description("Detecting markers on: {0}".format(fname))

        rgb = io.imread(path + fname)
        frame = rgb if len(rgb.shape) == 2 else cv2.cvtColor(
            rgb, cv2.COLOR_RGB2GRAY)
        success, ids2D, points2D = gbu.core.detect_markers(
            frame,
            dictionary=aruco_dict,
            parameters=parameters,
            drop_duplicates=True,
        )
        if success:
            Nm = len(ids2D)
            if plot_markers:
                imout = np.stack([frame, frame, frame], axis=2)
                draw_markers(
                    ids2D, points2D, imout, rejected=None)
                output_directory = checkdir(
                    directory=output_directory)

                io.imsave(
                    output_directory +
                    fname.split(".")[-2].split("/")[-1] +
                    "_markers." + plot_image_type,
                    imout,
                )
            for m in range(Nm):
                marker_id = ids2D[m]
                if marker_id in marker_dimension.keys():
                    data_detect_temp.at[ii + cpt,
                                        ("images", "", "")] = data_img.index[ii]
                    data_detect_temp.at[ii + cpt, ("method", "", "")
                                        ] = parameters.cornerRefinementMethod
                    data_detect_temp.at[ii + cpt,
                                        ("markers", "label", "")] = marker_id
                    data_detect_temp.at[ii + cpt,
                                        ("markers", "dimension", "")] = marker_dimension[marker_id]
                    data_detect_temp.at[ii + cpt,
                                        ("markers", "area", "")] = gbu.core.marker_area(points2D)[m]

                    for i, j in itertools.product(range(4), range(2)):
                        data_detect_temp.at[ii + cpt, ("p2d", "c{0}".format(
                            i), "xy"[j])] = points2D[m][i][j].astype(np.float64)

                    cpt += 1
        else:
            print("No markers found on {0}".format(fname))

    data_detect = pd.concat([data_detect, data_detect_temp]).reset_index(
        drop=True).drop_duplicates()
    data_detect.reset_index(drop=True, inplace=True)
    data_detect.index.name = "detects"
    return data_detect


def estimate_pose(data_detect=None,
                  data_pose=None,
                  camera_matrix=np.zeros([3, 3]),
                  distortion_coefficients=np.zeros([1, 5])):
    """
    Estimate pose of single marker from 2D points
    """
    def _get_status(pose, status):
        if status == 0:
            return True

        if status == 2:
            return False

        if status == 1:
            return pose == 0

    if data_pose is None:
        data_pose = create_pose_dataFrame()

    data_pose_temp = create_pose_dataFrame()

    pbar = tqdm(range(len(data_detect)))
    row = 0
    for ii in pbar:
        pbar.set_description(
            "Estimating pose on detect id_detect: {0}".format(data_detect.index[ii]))

        points3D = np.array(gbu.core.get_corners(
            dimension=data_detect.markers.dimension.loc[ii]))
        points2D = np.array(data_detect.p2d.loc[ii]).reshape([4, 2])
        out = gbu.core.estimate_pose_planar(points2D=points2D,
                                            points3D=points3D,
                                            camera_matrix=camera_matrix,
                                            distortion_coefficients=distortion_coefficients,
                                            )
        status, rvec0, tvec0, rvec1, tvec1, sol0, sol1 = out
        rvecs = [rvec0, rvec1]
        tvecs = [tvec0, tvec1]

        for pose in range(2):
            for i in range(3):
                data_pose_temp.at[row,
                                  ("rvec", "xyz"[i], "")] = rvecs[pose][i]
                data_pose_temp.at[row,
                                  ("tvec", "xyz"[i], "")] = tvecs[pose][i]
                data_pose_temp.at[row,
                                  ("detects", "", "")] = data_detect.index[ii]

                data_pose_temp.at[row,
                                  ("flag", "pose", "")] = _get_status(pose, status)

                if pose == 0:
                    data_pose_temp.at[row,
                                      ("type", "pose", "")] = "regular"
                    if sol0 is not None:
                        data_pose_temp.at[row,
                                          ("type", "error", "")] = np.sqrt(sol0.fun.sum()**2) / 8
                    else:
                        data_pose_temp.at[row,
                                          ("type", "error", "")] = np.nan
                else:
                    data_pose_temp.at[row,
                                      ("type", "pose", "")] = "planar"
                    if sol1 is not None:
                        data_pose_temp.at[row,
                                          ("type", "error", "")] = np.sqrt(sol1.fun.sum()**2) / 8
                    else:
                        data_pose_temp.at[row,
                                          ("type", "error", "")] = np.nan

            row += 1
    data_pose_temp[("flag", "graph", "processed")] = False
    data_pose_temp[("flag", "graph", "validated")] = False
    data_pose_temp[("flag", "graph", "rejected")] = False
    data_pose_temp[("occurency", "bad", "")] = 0
    data_pose_temp[("occurency", "good", "")] = 0
    data_pose = pd.concat([data_pose, data_pose_temp]).reset_index(
        drop=True).drop_duplicates()
    data_pose.reset_index(drop=True, inplace=True)
    data_pose.index.name = "poses"
    return data_pose


def calculate_inner_angles(data=None):
    """
    Calculates the 8 inner angles between the edges and the diagonals.
    """
    p2d = data.p2d
    angles = p2d.copy() * 0.
    angles.columns = pd.MultiIndex.from_product(
        [["angles"], ["c0", "c1", "c2", "c3", ], ["a0", "a1"]])
    vertices = [
        [3, 0, 2],
        [2, 0, 1],
        [0, 1, 3],
        [3, 1, 2],
        [1, 2, 0],
        [0, 2, 3],
        [2, 3, 1],
        [1, 3, 0], ]
    for i in range(len(vertices)):
        v = vertices[i]
        B = p2d.iloc[:, 2 * v[0]:2 * v[0] + 2].values.astype(np.float64)
        A = p2d.iloc[:, 2 * v[1]:2 * v[1] + 2].values.astype(np.float64)
        C = p2d.iloc[:, 2 * v[2]:2 * v[2] + 2].values.astype(np.float64)
        AB = B - A
        AC = C - A
        AB /= np.linalg.norm(AB, axis=1)[:, None]
        AC /= np.linalg.norm(AC, axis=1)[:, None]
        angle = -np.arcsin(np.cross(AB, AC))
        angles[("angles", "c{0}".format(i // 2), "a{0}".format(i % 2))
               ] = np.degrees(angle)
    deviation = angles.copy() - 45.
    deviation.columns = pd.MultiIndex.from_product(
        [["angular_deviation"], ["c0", "c1", "c2", "c3", ], ["a0", "a1"]])
    abs_max = np.abs(deviation.values).max(axis=1)
    deviation[("angular_deviation", "abs_max", "")] = abs_max
    data_out = pd.concat([angles, deviation], axis=1)
    data_out.index.name = data.index.name
    return data_out


def merge_dataFrames(dataFrame1, dataFrame2):
    """
    Merge to pandas Dataframe together according a common key column
    """
    dataFrame1 = dataFrame1.reset_index()
    dataFrame2 = dataFrame2.reset_index()
    diff_cols = dataFrame2.columns.difference(dataFrame1.columns)
    inter_cols = dataFrame2.columns.intersection(dataFrame1.columns)
    if len(inter_cols) != 0:
        key_col = inter_cols[0][0]
        return pd.merge(dataFrame1, dataFrame2[diff_cols],
                        left_on=key_col, right_index=True)
    else:
        print("No corespondance key column was found !")
        return None


def get_multigraph(data):
    """
    Request to compute central marker
    """

    im_dicin, im_dicout = string_map(data.images.unique(),
                                     prefix="i")
    marker_dicin, marker_dicout = string_map(data.markers.label.unique(),
                                             prefix="m")
    MG = nx.MultiGraph()
    for imlabel, group in data.groupby("images"):
        for combination in itertools.combinations(group.markers.label, 2):
            n0 = combination[0]
            n1 = combination[1]
            MG.add_edge(n0, n1, label=im_dicin[imlabel])
    return MG


def get_central_marker(data):
    """
    Returns the central marker
    """
    return sorted(nx.center(get_multigraph(data)))[0]


def get_graph(data):
    """
    Get graph from data.
    """
    im_dicin, im_dicout = string_map(data.images.unique(),
                                     prefix="i")
    marker_dicin, marker_dicout = string_map(data.markers.label.unique(),
                                             prefix="m")
    G = nx.Graph()
    dp = data
    successful_detects = dp[dp.flag.pose].detects.unique()
    dd = data.loc[data['detects'].isin(successful_detects)]
    for i, row in dd.iterrows():
        im = im_dicin[row.images.values[0]]
        ma = marker_dicin[row.markers.label.values[0]]
        G.add_edge(im, ma, dist=1.)
    return G


def string_map(ids, prefix="", zfill=True):
    """
    Returns a string / int reversible map.
    """
    lel = 0
    if zfill:
        for i in ids:
            lel = max(lel, len(str(i)))
    dic_in = {i: prefix + "{0}".format(i).zfill(lel) for i in ids}
    dic_out = {v: k for k, v in dic_in.items()}
    return dic_in, dic_out


def plot_graph(graph, colors = None, title = "Graph", figsize=None, fname=None, show_legend=True, *args, **kwargs):
    """
    Plots the Graph.
    """
    
    if colors is None:
        colors = ["blue", "red"]
    color_map = []

    for node in graph.nodes:
        if node.startswith('i'):
            color_map.append('red')
        else:
            color_map.append('blue')
    nx.draw(graph, node_color=color_map, with_labels=True, font_weight='bold')
    plt.draw()
    plt.title(title)

    f = lambda m,c: plt.plot([],[],marker='o', color=c, ls="none")[0]
    handles = [f("s", colors[i]) for i in range(2)]
    
    if show_legend:
        labels = ["Markers", "Images"]
        plt.legend(handles, labels,bbox_to_anchor=(0.72, .05),
                ncol=3, fancybox=True, shadow=True, *args, **kwargs)

    if figsize is not None:
        plt.figure(figsize = figsize)

    if fname is not None:
        plt.savefig(fname,*args,**kwargs)

    plt.show()

def graph_path_RTs(data, cycle):
    """
    Gets the R/T transformation along a path.
    """
    marker_dicin, marker_dicout = string_map(data.markers.label.unique(),
                                             prefix="m")
    im_dicin, im_dicout = string_map(data.images.unique(),
                                     prefix="i")
    cycle = list(cycle)
    cycle.append(cycle[0])
    Ne = len(cycle) - 1
    poses = {}
    edges_ids = []
    for e in range(Ne):
        edges_ids.append([])
        ni, no = cycle[e], cycle[e + 1]
        inverted = ni.startswith("i")  # inverted path = image to marker
        imageLabel, marker = (ni, no) if inverted else (no, ni)
        edge_data = data[(data.images == im_dicout[imageLabel])
                         & (data.markers.label == marker_dicout[marker])]

        for ind, row in edge_data.iterrows():
            Re = row.rvec.values.flatten().astype(np.float64)
            Te = row.tvec.values.flatten().astype(np.float64)
            if inverted:
                Re, Te = invert_RT(Re, Te)

            poses[row.poses[0]] = Re, Te
            edges_ids[-1].append(row.poses[0])
    return poses, edges_ids


def graph_path_RT(graph, data, nodeIn, nodeOut, weight="dist"):
    """
    Gets the R/T transformation using the shortest path between any 2 nodes.
    """
    marker_dicin, marker_dicout = string_map(data.markers.label.unique(),
                                             prefix="m")
    im_dicin, im_dicout = string_map(data.images.unique(),
                                     prefix="i")
    R, T = np.zeros((2, 3), dtype=np.float64)
    path = nx.shortest_path(graph, nodeIn, nodeOut, weight="dist")
    Ne = len(path) - 1
    for e in range(Ne):
        ni, no = path[e], path[e + 1]
        inverted = ni.startswith("i")  # inverted path = image to marker
        imageLabel, marker = (ni, no) if inverted else (no, ni)
        d = data[(data.images == im_dicout[imageLabel])
                 & (data.markers.label == marker_dicout[marker])]
        Re = d.rvec.values.astype(np.float64).flatten()
        Te = d.tvec.values.astype(np.float64).flatten()
        if inverted:
            Re, Te = invert_RT(Re, Te)
        R, T = compose_RT(R, T, Re, Te)
    return R, T


def cycle_permutation_residuals(edges_ids, poses, size_of_cycle=1):
    """
    Tests all the possible permutations.
    """

    permutations = list(itertools.product(*edges_ids))
    residual_RT = []
    prbar = tqdm(range(len(permutations)), leave=False)
    for jj in prbar:
        R = np.zeros(3, dtype=np.float64)
        T = np.zeros(3, dtype=np.float64)
        for edge_id in permutations[jj]:
            prbar.set_description(
                "Working on permuation: {0}".format(permutations[jj]))
            Re, Te = poses[edge_id]
            R, T = compose_RT(R, T, Re, Te)
        residual_RT.append([R, T])

    T_residuals = np.linalg.norm(
        np.array(residual_RT)[:, 1], axis=1) / size_of_cycle
    R_residuals = np.linalg.norm(
        np.array(residual_RT)[:, 0], axis=1) / size_of_cycle

    return permutations, R_residuals, T_residuals


def get_cycles(data, graph, root=None, atomic_cycle=False, kind="basis", seed=None):
    marker_dicin, marker_dicout = string_map(data.markers.label.unique(),
                                             prefix="m")
    im_dicin, im_dicout = string_map(data.images.unique(),
                                     prefix="i")
    if root is not None:
        root = marker_dicin[root]

    if kind == "basis":
        raw_cycles = cycle_basis(graph, root=root, seed=seed)

    cycles = []
    for rc in raw_cycles:
        perms = more_itertools.circular_shifts(rc)
        for perm in perms:
            if perm[0].startswith("m"):
                break
        cycles.append(perm)

    if atomic_cycle:
        return [cycle for cycle in cycles if len(cycle) == 4]
    else:
        return cycles


def get_occurencies(data, cycles, criterion=0.25):
    """
    criterion : mean angular criterion allowed for each transformation of the cycle
    """
    occurrency = {pose: {"bad": 0,
                         "good": 0,
                         "processed": False} for pose in data.index.values}
    data_out = data.copy()
    data_out.loc[:, ("occurency", "bad", "")] = 0
    data_out.loc[:, ("occurency", "good", "")] = 0
    pbar = tqdm(range(len(cycles)))
    for ii in pbar:
        pbar.set_description(
            "Working on cycle: {0}".format(cycles[ii]))
        poses, edges_ids = graph_path_RTs(data_out, cycles[ii])
        permutations, R_residuals, T_residuals = cycle_permutation_residuals(
            edges_ids=edges_ids,
            poses=poses,
            size_of_cycle=len(cycles[ii]))

        best_locs = (np.argwhere(np.degrees(
            R_residuals) < criterion)).flatten()

        if len(best_locs):
            best_poses = np.array(permutations)[best_locs]
            bad_poses = np.delete(
                np.array(permutations), best_locs, axis=0)

            for bad in bad_poses.flatten():
                occurrency[bad]["bad"] += 1

            for best in best_poses.flatten():
                occurrency[best]["good"] += 1
        else:
            for perm in np.array(permutations).flatten():
                occurrency[perm]["bad"] += 1

        for perm in np.array(permutations).flatten():
            occurrency[perm]["processed"] = True

    for key, values in occurrency.items():
        data_out.at[key, ("occurency", "bad", "")] = values['bad']
        data_out.at[key, ("occurency", "good", "")] = values['good']
        data_out.at[key, ("flag", "graph", "processed")] = values['processed']
    return data_out


def occurency_analysis(data, alpha_criterion=1.5):
    no_pose_class = data.loc[(data.flag.pose == False)]
    out_of_cycle_class = data.loc[(data.flag.pose == True) & (
        data.flag.graph.processed == False)]

    list_of_interest = [
        ('markers', 'label', ''),
        ('detects', '', ''),
        ('occurency', 'good', ''),
        ('occurency', 'bad', ''),
        ('image', 'fname', ''), ]

    full_class = data.loc[(data.flag.pose == True) & (
        data.flag.graph.processed == True)][list_of_interest]

    # handle truely bad poses
    ind_bad = full_class.loc[(full_class.occurency.good == 0)].index
    data.loc[data.index.isin(ind_bad),
             ('flag', 'graph', 'rejected')] = True

    bad_class = full_class.loc[full_class.occurency.good == 0]
    not_bad_class = full_class.loc[full_class.occurency.good != 0]

    detects_unknow = not_bad_class[
        not_bad_class.duplicated().values]["detects"].values

    known_class = not_bad_class.loc[(
        ~not_bad_class['detects'].isin(detects_unknow))]

    detects_toChoose = known_class[known_class['detects'].duplicated(
    ).values]['detects']

    # spot prim valid poses
    prim_valid_poses = known_class.loc[(
        ~known_class['detects'].isin(detects_toChoose))].index.values
    data.loc[prim_valid_poses, ('flag', 'graph', 'validated')] = True

    # spot 2nd valid poses
    toChoose_class = known_class.loc[known_class['detects'].isin(
        detects_toChoose)]
    for detect in detects_toChoose.values:
        sub_df = toChoose_class.loc[toChoose_class['detects'] == detect]
        loc_valid = np.argmax(sub_df.occurency.good.values)
        loc_reject = np.argmin(sub_df.occurency.good.values)
        good_pose = sub_df.iloc[loc_valid].name
        bad_pose = sub_df.iloc[loc_reject].name

        count_max = sub_df.iloc[loc_valid].occurency.good.values.astype(np.float64)[
            0]
        count_min = sub_df.iloc[loc_reject].occurency.good.values.astype(np.float64)[
            0]
        alpha = count_max / count_min

        if alpha >= alpha_criterion:
            data.loc[good_pose, ('flag', 'graph', 'validated')] = True
            data.loc[bad_pose, ('flag', 'graph', 'rejected')] = True
        else:
            data.loc[good_pose, ('flag', 'graph', 'validated')] = False
            data.loc[bad_pose, ('flag', 'graph', 'rejected')] = False

    ambigus_class = data.loc[(data.flag.pose == True) &
                             (data.flag.graph.processed == True) &
                             (data.flag.graph.validated == False) &
                             (data.flag.graph.rejected == False)]

    bad_class = data.loc[(data.flag.pose == True) &
                         (data.flag.graph.processed == True) &
                         (data.flag.graph.validated == False) &
                         (data.flag.graph.rejected == True)]

    good_class = data.loc[(data.flag.pose == True) &
                          (data.flag.graph.processed == True) &
                          (data.flag.graph.validated == True) &
                          (data.flag.graph.rejected == False)]

    print("full data \t= {0} poses".format(data.shape[0]))
    print("no valid pose \t= {0} poses".format(no_pose_class.shape[0]))
    print("out of cycle \t= {0} poses".format(out_of_cycle_class.shape[0]))
    print("good data \t= {0} poses".format(good_class.shape[0]))
    print("bad data \t= {0} poses".format(bad_class.shape[0]))
    print("ambigus data \t= {0} poses".format(ambigus_class.shape[0]))

    return no_pose_class, out_of_cycle_class, ambigus_class, bad_class, good_class


def check_pose_proximity(data):
    """
    Todo
    """
    return None


def check_graph_connection(data):
    G = get_graph(data)
    return nx.is_connected(G)


def get_good_core(data, criterion=1, deph_max=10, alpha_criterion=1.5, atomic_cycle=False):

    no_pose_core, out_of_cycle_core, ambigus_core, bad_core, good_core = occurency_analysis(
        data=data, alpha_criterion=alpha_criterion)

    if len(ambigus_core) > 0:
        deph = 0
        while deph < deph_max:
            new_data = pd.concat([good_core, ambigus_core])
            new_graph = get_graph(new_data)
            if nx.is_connected(new_graph):
                new_central_mk = get_central_marker(new_data)
                new_cycles = get_cycles(
                    data=new_data, graph=new_graph, root=new_central_mk, atomic_cycle=atomic_cycle)
                new_data_out = get_occurencies(
                    data=new_data, cycles=new_cycles, criterion=criterion)
                ret, ret, ambigus_core, bad_core, good_core = occurency_analysis(
                    data=new_data_out, alpha_criterion=alpha_criterion)
                if (len(ambigus_core) == 0) or len(pd.concat([good_core, ambigus_core])) == len(new_data):
                    break
                deph += 1
            else:
                print("Graph not connected, end recursion here")
                break

    return good_core.sort_index()


def cycle_basis(G, root=None, seed=None):
    """
    A rewritting of networkx cycles basis computation in order to handle 
    random behaviours.

    Returns a list of cycles which form a basis for cycles of G.

    A basis for cycles of a network is a minimal collection of
    cycles such that any cycle in the network can be written
    as a sum of cycles in the basis.  Here summation of cycles
    is defined as "exclusive or" of the edges. Cycle bases are
    useful, e.g. when deriving equations for electric circuits
    using Kirchhoff's Laws.

    Source : https://networkx.org/documentation/stable/_modules/networkx/algorithms/cycles.html#cycle_basis

    Parameters
    ----------
    G : NetworkX Graph
    root : node, optional
       Specify starting node for basis.

    Returns
    -------
    A list of cycle lists.  Each cycle list is a list of nodes
    which forms a cycle (loop) in G.

    Examples
    --------
    >>> G = nx.Graph()
    >>> nx.add_cycle(G, [0, 1, 2, 3])
    >>> nx.add_cycle(G, [0, 3, 4, 5])
    >>> print(nx.cycle_basis(G, 0))
    [[3, 4, 5, 0], [1, 2, 3, 0]]

    Notes
    -----
    This is adapted from algorithm CACM 491 [1]_.

    References
    ----------
    .. [1] Paton, K. An algorithm for finding a fundamental set of
       cycles of a graph. Comm. ACM 12, 9 (Sept 1969), 514-518.

    See Also
    --------
    simple_cycles
    """
    random.seed(seed)
    np.random.seed(seed)
    gnodes = set(G.nodes())
    cycles = []
    while gnodes:  # loop over connected components
        if root is None:
            if seed is not None:
                root = sorted(list(gnodes))[
                    np.random.randint(len(list(gnodes)))]
                gnodes.remove(root)
            else:
                root = gnodes.pop()

        stack = [root]
        pred = {root: root}
        used = {root: set()}
        while stack:  # walk the spanning tree finding cycles
            z = stack.pop()  # use last-in so cycles easier to find
            zused = used[z]
            for nbr in G[z]:
                if nbr not in used:  # new node
                    pred[nbr] = z
                    stack.append(nbr)
                    used[nbr] = {z}
                elif nbr == z:  # self loops
                    cycles.append([z])
                elif nbr not in zused:  # found a cycle
                    pn = used[nbr]
                    cycle = [nbr, z]
                    p = pred[z]
                    while p not in pn:
                        cycle.append(p)
                        p = pred[p]
                    cycle.append(p)
                    cycles.append(cycle)
                    used[nbr].add(z)
        gnodes -= set(pred)
        root = None
    return cycles


class ImageBatch(gbu.core.Container):
    """
    A class to process images containing markers as batches.

    """

    def __init__(self,
                 aruco_dict=aruco.DICT_6X6_250,
                 parameters=None,
                 marker_dimension={m: 1.0 for m in range(1, 1001)},
                 output_directory=None,
                 camera_matrix=np.eye(3),
                 distortion_coefficients=np.zeros(5), **kwargs):

        self.aruco_dict = aruco_dict
        self.parameters = parameters
        self.marker_dimension = marker_dimension
        self.output_directory = output_directory
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients

        # load dataframes
        self.data_img = create_image_dataFrame()
        self.data_detect = create_detect_dataFrame()
        self.data_pose = create_pose_dataFrame()

        if 'directory' in kwargs:
            self.input_directory = kwargs['directory']
            if output_directory is None:
                self.output_directory = self.input_directory + "/_outputs/"

            self.load_image_batch(directory=self.input_directory)
            self.detect_markers(plot_markers=True, enforce=False)

        if output_directory is None:
            self.output_directory = "./_outputs/"

    def load_image_batch(self, directory="./", format=".tif", slice=None):
        """
        Wrapper
        """
        self.input_directory = directory
        self.data_img = load_image_batch(
            data_img=self.data_img, directory=directory, format=format, slice=slice)

        return self.data_img

    def detect_markers(self,
                       plot_markers=False,
                       plot_image_type="jpg",
                       meta_filename="metadata.h5",
                       enforce=False):
        """
        Wrapper
        """
        if enforce:
            self.data_detect = detect_markers(parameters=self.parameters,
                                              marker_dimension=self.marker_dimension,
                                              aruco_dict=self.aruco_dict,
                                              output_directory=self.output_directory,
                                              data_img=self.data_img,
                                              data_detect=self.data_detect,
                                              plot_markers=plot_markers,
                                              plot_image_type=plot_image_type)
            dump_dataframe_HDF(img_dir=self.input_directory,
                               key='detects',
                               dataframe=self.data_detect)

        elif os.path.exists(self.input_directory + meta_filename):
            self.data_detect = cherry_dataframe_HDF(img_dir=self.input_directory,
                                                    key="detects")
        else:
            self.data_detect = detect_markers(parameters=self.parameters,
                                              marker_dimension=self.marker_dimension,
                                              aruco_dict=self.aruco_dict,
                                              output_directory=self.output_directory,
                                              data_img=self.data_img,
                                              data_detect=self.data_detect,
                                              plot_markers=plot_markers,
                                              plot_image_type=plot_image_type)
            dump_dataframe_HDF(img_dir=self.input_directory,
                               key='detects',
                               dataframe=self.data_detect)

        return self.data_detect

    def estimate_pose(self, **kwargs):
        """
        Estimate pose of single markers
        Wrapper
        """
        if len(self.data_detect) == 0:
            print("Never found makers on image")
            self.data_pose = None
        else:
            self.data_pose = estimate_pose(data_detect=self.data_detect,
                                           data_pose=self.data_pose,
                                           camera_matrix=self.camera_matrix,
                                           distortion_coefficients=self.distortion_coefficients)
        return self.data_pose

    def calculate_inner_angles(self, data=None):
        """
        Wrapper
        """
        if data is None:
            data = self.data_detect
        self.data_angle = calculate_inner_angles(data=self.data_detect)

        return self.data_angle

    def merge_all(self):
        data_temp = merge_dataFrames(self.data_pose, self.data_detect)
        return merge_dataFrames(data_temp, self.data_img)

    def get_graph_data(self, plot_markers=False, criterion=1, deph_max=10, alpha_criterion=1.5, atomic_cycle=False, enabled_central_mk=True):
        data = self.merge_all()
        graph = get_graph(data)
        central_mk = get_central_marker(data) if enabled_central_mk else None
        cycles = get_cycles(data=data, graph=graph,
                            atomic_cycle=atomic_cycle, root=central_mk)
        self.data_raw = get_occurencies(
            data=data, cycles=cycles, criterion=criterion)
        self.data_graph = get_good_core(
            data=self.data_raw,
            criterion=criterion,
            deph_max=deph_max,
            alpha_criterion=alpha_criterion,
            atomic_cycle=atomic_cycle)

        self.markers_lost_checking()

        if plot_markers:
            self.draw_xyz()

        return self.data_graph

    def markers_lost_checking(self):
        group_calib_img = self.data_graph.images.unique()
        group_calib_detect = self.data_graph.detects.unique()
        group_calib_pose = self.data_graph.poses.unique()
        self.data_graph_img = self.data_img.loc[group_calib_img]
        self.data_graph_detect = self.data_detect.loc[group_calib_detect]
        self.data_graph_pose = self.data_pose.loc[group_calib_pose]

        # CHECK IF SOME MARKERS HAVE BEEN LOST
        mks_in = self.data_detect.markers.label.unique()
        mks_out = self.data_graph.markers.label.unique()
        mks_missing = set(mks_in) - set(mks_out)
        if len(mks_missing) != 0:
            print("/!\\ we lose marker(s) : {0}".format(mks_missing))

    def draw_xyz(self, data=None, plot_image_type="jpg", *args, **kwargs):
        if data is None:
            data = self.data_graph

        Np = len(data)
        pbar = tqdm(range(Np))
        path = self.output_directory
        for ii in pbar:
            fname = data.image.fname.iloc[ii]
            out_fname = fname.split(
                ".")[-2].split("/")[-1] + "_markers." + plot_image_type
            pbar.set_description(
                "Drawing axis markers on: {0}".format(fname))

            imout = io.imread(path + out_fname)
            imout = draw_xyz(
                imout,
                data.rvec.iloc[ii].values.astype(np.float64),
                data.tvec.iloc[ii].values.astype(np.float64),
                camera_matrix=self.camera_matrix,
                distortion_coefficients=self.distortion_coefficients,                
                *args,
                **kwargs
            )

            self.output_directory = checkdir(
                directory=self.output_directory)

            io.imsave(self.output_directory +
                      fname.split(".")[-2].split("/")[-1] +
                      "_markers." + plot_image_type,
                      imout,
                      )

    def draw_markers(self, plot_image_type='jpg'):
        def draw_panda_func(sub_df):
            ind = sub_df.images.values[0]
            path = self.data_img.image.path.loc[ind]
            fname = self.data_img.image.fname.loc[ind]

            frame = io.imread(path + fname)
            if len(frame.shape) == 2:
                imout = np.stack([frame, frame, frame], axis=2)
            else:
                imout = frame

            ids2D = sub_df.markers.label.values.astype(np.int64)
            points2D = sub_df.p2d.values.astype(np.float32)
            draw_markers(
                ids2D, points2D, imout, rejected=None)
            output_directory = checkdir(
                directory=self.output_directory)

            io.imsave(
                self.output_directory +
                fname.split(".")[-2].split("/")[-1] +
                "_markers." + plot_image_type,
                imout,
            )
            return sub_df
        tqdm.pandas()
        print("Drawing markers...")
        self.data_detect.groupby('images').progress_apply(draw_panda_func)
        print("Drawing markers done")

    def get_central_marker(self):
        self.central_mk = get_central_marker(self.merge_all())
        return self.central_mk

    def get_multigraph(self):
        self.multigraph = get_multigraph(self.merge_all())
        return self.multigraph

    def export_batch(self, spath="./", fname="batch.p"):

        cornerRefinementWinSize = self.parameters.cornerRefinementWinSize
        cornerRefinementMaxIterations = self.parameters.cornerRefinementMaxIterations
        cornerRefinementMethod = self.parameters.cornerRefinementMethod
        del self.parameters
        data_batch = {"batch": self.__dict__,
                      "parameters": {
                          "cornerRefinementWinSize": cornerRefinementWinSize,
                          "cornerRefinementMaxIterations": cornerRefinementMaxIterations,
                          "cornerRefinementMethod": cornerRefinementMethod,
                      }}

        pickle.dump(data_batch,
                    open(spath + fname, "wb"))

        parameters = aruco.DetectorParameters_create()
        parameters.cornerRefinementWinSize = cornerRefinementWinSize
        parameters.cornerRefinementMaxIterations = cornerRefinementMaxIterations
        parameters.cornerRefinementMethod = cornerRefinementMethod
        self.parameters = parameters

    def load_batch(self, lpath="./", fname="batch.p"):

        data_batch = pickle.load(open(lpath + fname, "rb"))

        self.__dict__ = data_batch["batch"]

        self.parameters = aruco.DetectorParameters_create()
        self.parameters.cornerRefinementWinSize = data_batch["parameters"]["cornerRefinementWinSize"]
        self.parameters.cornerRefinementMaxIterations = data_batch[
            "parameters"]["cornerRefinementMaxIterations"]
        self.parameters.cornerRefinementMethod = data_batch["parameters"]["cornerRefinementMethod"]
