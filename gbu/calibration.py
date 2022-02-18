# Filename: <calibration.py>
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
# GBU/CALIBRATION
################################################################################

import copy
import os
import threading
import time
################################################################################
# IMPORTS
from datetime import datetime

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from cv2 import aruco
from scipy import optimize
from skimage import io
from tqdm import tqdm

import gbu

if [e for e in list(os.environ) if e == "MPLBACKEND"]:
    from tqdm import tqdm_notebook as tqdm



def set_composite_metric(compo, marker_size=None, marker_coeff=None, inplace=False):
    """
    #########################################################################
    Careful this is insane and works if all markers have the same length
    #########################################################################
    mode : "size" or "coeff"
    if mode = coeff use marker_coeff
    if mode = size use marker_size in meter
    """

    compo_metric = compo.copy()
    if marker_size is not None:
        dims = compo.dimensions
        marker_coeff = marker_size / dims[0]
        print("marker_coeff: {0}".format(marker_coeff))

    compo_metric.dimensions = compo_metric.dimensions * marker_coeff
    compo_metric.tvecs = compo_metric.tvecs * marker_coeff

    if inplace:
        compo = compo_metric
    else:
        return compo_metric


def graph_calibration(data, reference=None, weight="dist"):
    """
    Guesses the marker and camera pose using graph theory.

    Inputs:
    * reference: the reference marker id. If None, the central marker of the
                 graph will be used. If not None, has to be the id of an
                 existing marker (experimental).
    """
    marker_dicin, marker_dicout = gbu.utils.string_map(data.markers.label.unique(),
                                                       prefix="m")
    im_dicin, im_dicout = gbu.utils.string_map(data.images.unique(),
                                               prefix="i")

    MG = gbu.utils.get_multigraph(data=data)

    G = gbu.utils.get_graph(data=data)
    if reference is None:
        centralMarker = gbu.utils.get_central_marker(data)
    else:
        centralMarker = reference
    markers = np.array(MG.nodes)
    nonCentralMarkers = set(markers) - set([centralMarker])

    # MARKER POSE ESTIMATION
    cols = [
        ("rvec", "x"),
        ("rvec", "y"),
        ("rvec", "z"),
        ("tvec", "x"),
        ("tvec", "y"),
        ("tvec", "z"),
    ]
    cols = pd.MultiIndex.from_tuples(cols)
    mpose = pd.DataFrame(
        columns=cols, index=markers, data=np.zeros((len(markers), len(cols)))
    )
    mpose.index.name = "marker"
    for marker in nonCentralMarkers:
        R, T = gbu.utils.graph_path_RT(graph=G,
                                       data=data,
                                       nodeIn=marker_dicin[marker],
                                       nodeOut=marker_dicin[centralMarker])
        mpose.loc[marker] = np.concatenate([R, T])
    mpose["central"] = False
    mpose.loc[centralMarker, "central"] = True
    # CAMERA POSE ESTIMATION
    imageLabels = data.images.unique()
    cpose = pd.DataFrame(
        columns=cols, index=imageLabels, data=np.zeros(
            (len(imageLabels), len(cols)))
    )
    cpose.index.name = "imageLabel"
    for imageLabel in imageLabels:
        R, T = gbu.utils.graph_path_RT(
            graph=G,
            data=data,
            nodeIn=marker_dicin[centralMarker],
            nodeOut=im_dicin[imageLabel]
        )
        cpose.loc[imageLabel] = np.concatenate([R, T])
    return cpose, mpose, G, MG


def composite_geometrical_feature(compo_target, targetPoseBatch, refPoseBatch=None,
                                  kind="point", fixed="z", more_features=False):

    rvec, tvec, residuals = gbu.calibration.find_geometrical_feature(
        targetPoseBatch=targetPoseBatch,
        refPoseBatch=refPoseBatch,
        kind=kind,
        fixed=fixed)

    if kind == "point":
        rvec_compo2newcompo, tvec_compo2newcompo = gbu.utils.invert_RT(
            R=rvec, T=tvec)

    if kind == "axis":
        rvec_compo2newcompo = rvec
        tvec_compo2newcompo = tvec

    if kind == "bit":
        rvec_compo2newcompo = rvec
        tvec_compo2newcompo = tvec

    new_compo = compo_target.apply_RT(rvec_compo2newcompo, tvec_compo2newcompo)

    if not more_features:
        return new_compo
    else:
        return new_compo, rvec_compo2newcompo, tvec_compo2newcompo


def find_geometrical_feature(targetPoseBatch, refPoseBatch=None,
                             kind="point", fixed="z"):
    """
    Finds the transformation to a geometrical feature (fixed point, axis, ...).

    Inputs:
    * targetPoseBatch: a poseBatch instance of the target.
    * reference: if not None, implies that the camera is not the movement
    reference. In this case, has to be a poseBatch of the reference used for
    the movement.
    * kind: can be in ["point"].
        * "point": a fixed point is searched.
        * todo.
    * fixed: todo.
    """

    Nim = len(targetPoseBatch.ids)
    print(f"Choosen fixed= {fixed}")

    if kind == "point":
        def cost(P, Pt, rvecs, tvecs):
            for i in range(Nim):
                Pt[i] = gbu.utils.apply_RT(P, rvecs[i], tvecs[i])
                print("optimizing\t", datetime.now().strftime(
                    "%H:%M:%S"), end='\r')
            return (Pt - Pt.mean(axis=0)).flatten()

        if refPoseBatch is None:
            data = targetPoseBatch
            Pt = np.zeros((Nim, 3))
            P0 = np.array([0., 0., 0.])  # Initial condition
            R_target2cam = data.rvecs
            T_target2cam = data.tvecs
            sol = optimize.least_squares(cost, 
                                         P0, 
                                         method="lm",
                                         ftol=1.0e-12,
                                         xtol=1.0e-12,
                                         gtol=1.e-10,
                                         args=(Pt, R_target2cam, T_target2cam))
            Psol = sol.x
            # rvec, tvec, residuals
            return np.zeros(3), Psol, cost(Psol, Pt, R_target2cam, T_target2cam).reshape(Pt.shape)

        else:
            data_target = targetPoseBatch
            data_ref = refPoseBatch
            Pt_target = np.zeros((Nim, 3))
            P0 = np.array([0., 0., 0.])  # Initial condition

            R_cam2ref = np.zeros([Nim, 3])
            T_cam2ref = np.zeros([Nim, 3])

            R_target2ref = np.zeros([Nim, 3])
            T_target2ref = np.zeros([Nim, 3])

            for i in range(Nim):
                R_cam2ref[i], T_cam2ref[i] = gbu.utils.invert_RT(
                    R=data_ref.rvecs[i], T=data_ref.tvecs[i])

                R_target2ref[i], T_target2ref[i] = gbu.utils.compose_RT(
                    data_target.rvecs[i], data_target.tvecs[i],
                    R_cam2ref[i], T_cam2ref[i])

            sol = optimize.least_squares(cost,
                                         P0,
                                         method="lm",
                                         ftol=1.0e-12,
                                         xtol=1.0e-12,
                                         gtol=1.e-10,
                                         args=(Pt_target, R_target2ref, T_target2ref))
            Psol = sol.x
            # rvec, tvec, residuals
            return np.zeros(3), Psol, cost(
                Psol, Pt_target, R_target2ref, T_target2ref).reshape(Pt_target.shape)

    if kind == "axis":
        axis_dict = {"x":0,
                     "y":1,
                     "z":2,}

        if refPoseBatch is None:
            def unpack(X):
                R_target2a = X[:3]
                T_target2a = np.zeros(3)
                T_target2a[1] = X[3]
                R_b2cam = X[4:7]
                T_b2cam = X[7:10]
                theta = X[10:]
                return R_target2a, T_target2a, R_b2cam, T_b2cam, theta

            def cost(X, R_cam2target, T_cam2target, axis):
                R_target2a, T_target2a, R_b2cam, T_b2cam, theta = unpack(X)
                out = np.zeros((Nim, 6))
                for i in range(Nim):
                    # Rotation !
                    R_a2b = np.zeros(3)
                    R_a2b[axis] = theta[i]
                    T_a2b = np.zeros(3)
                    # COMPOSITION
                    R_target2b, T_target2b = gbu.utils.compose_RT(
                        R_target2a, T_target2a, R_a2b, T_a2b)
                    R_target2cam, T_target2cam = gbu.utils.compose_RT(
                        R_target2b, T_target2b, R_b2cam, T_b2cam)
                    # RESIDUALS
                    R_target2target, T_target2target = gbu.utils.compose_RT(
                        R_target2cam, T_target2cam,
                        R_cam2target[i], T_cam2target[i])
                    out[i, :3] = R_target2target
                    out[i, 3:] = T_target2target

                    print("optimizing\t", datetime.now().strftime(
                        "%H:%M:%S"),f"\tmean resisdu = {out.mean()}", end='\r')
                return out.flatten()

            data = targetPoseBatch
            R_cam2target = np.zeros((Nim, 3))
            T_cam2target = np.zeros((Nim, 3))
            for i in range(Nim):
                R_cam2target[i], T_cam2target[i] = gbu.utils.invert_RT(
                    data.rvecs[i],
                    data.tvecs[i])
            Nun = 6 + Nim + 4  # Number of unknown parameters in optimization
            X0[:3] = np.ones(3)
            X0[10:] = 1  # Init theta with non zeros angles
            sol = optimize.least_squares(cost,
                                         X0,
                                         method="lm",
                                         ftol=1.0e-12,
                                         xtol=1.0e-12,
                                         gtol=1.e-10,
                                         args=(R_cam2target, T_cam2target, axis_dict[fixed]))
            
            R_target2a, T_target2a, R_b2cam, T_b2cam, theta = unpack(sol.x)

            # rvec, tvec, residuals (Nim, Rvec-Tvec, x-y-z)
            return R_target2a, T_target2a, cost(sol.x, R_cam2target, T_cam2target, axis_dict[fixed]).reshape(Nim, -1, 3)

        else:
            def unpack(X):
                """
                Don't touch my tralala !!
                """
                R_target2a = X[0:3]
                T_target2a = np.zeros(3)
                T_target2a[1] = X[3]
                R_b2ref = X[4:7]
                T_b2ref = X[7:10]
                theta = X[10:]
                return R_target2a, T_target2a, R_b2ref, T_b2ref, theta

            def cost(X, R_ref2target, T_ref2target, axis):
                """
                Cost with residuals based on R and T (ninja style)
                """
                R_target2a, T_target2a, R_b2ref, T_b2ref, theta = unpack(X)
                out = np.zeros((Nim, 6))
                for i in range(Nim):
                    # Rotation !
                    R_a2b = np.zeros(3)
                    R_a2b[axis] = theta[i]
                    T_a2b = np.zeros(3)
                    # COMPOSITION
                    R_target2b, T_target2b = gbu.utils.compose_RT(
                        R_target2a, T_target2a, R_a2b, T_a2b)
                    R_target2ref, T_target2ref = gbu.utils.compose_RT(
                        R_target2b, T_target2b, R_b2ref, T_b2ref)
                    # RESIDUALS
                    R_target2target, T_target2target = gbu.utils.compose_RT(
                        R_target2ref, T_target2ref,
                        R_ref2target[i], T_ref2target[i])
                    out[i, :3] = R_target2target
                    out[i, 3:] = T_target2target

                    print("optimizing\t", datetime.now().strftime(
                        "%H:%M:%S"),f"\tmean resisdu = {out.mean()}", end='\r')
                return out.flatten()

            data_target = targetPoseBatch  # driller
            data_ref = refPoseBatch  # stylus

            R_ref2target = np.zeros((Nim, 3))
            T_ref2target = np.zeros((Nim, 3))
            for i in range(Nim):
                R_cam2target, T_cam2target = gbu.utils.invert_RT(
                    data_target.rvecs[i],
                    data_target.tvecs[i])
                R_ref2target[i], T_ref2target[i] = gbu.utils.compose_RT(
                    data_ref.rvecs[i],
                    data_ref.tvecs[i],
                    R_cam2target, T_cam2target)

            Nun = 6 + Nim + 4  # Number of unknown parameters in optimization
            X0 = np.zeros(Nun)
            X0[:3] = np.ones(3)
            X0[10:] = 1  # Init theta with non zeros angles
            sol = optimize.least_squares(cost, 
                                         X0,
                                         method="lm", 
                                         ftol=1.0e-12,
                                         xtol=1.0e-12,
                                         gtol=1.e-10,
                                         args=(R_ref2target, T_ref2target, axis_dict[fixed]))
            R_target2a, T_target2a, R_b2ref, T_b2ref, theta = unpack(sol.x)

            # Check if reference is reversed along z
            if gbu.utils.apply_RT(np.array([0., 0., 1.]),
                                  R_target2a, np.zeros(3))[2] > 0:
                R_target2a, T_target2a = gbu.utils.compose_RT(
                    R_target2a, T_target2a,
                    np.array([np.pi, 0., 0.]), np.zeros(3))

            # Check orientation of y Axis
            if T_target2a[1] < 0.:
                R_target2a, T_target2a = gbu.utils.compose_RT(
                    R_target2a, T_target2a,
                    np.array([0., 0., np.pi]), np.zeros(3))

            # rvec, tvec, residuals (Nim, Rvec-Tvec, x-y-z)
            return R_target2a, T_target2a, cost(sol.x, R_ref2target, T_ref2target).reshape(Nim, -1, 3)

    if kind == "bit":
        """
        Todo !
        """
        return None, None, None


def markers_on_images(index=None,
                      data_detect=None,
                      poseBatches=None,
                      composites=None):
    """
    Returns the location of individual markers on the composite in each
    image.
    """

    Ni = len(poseBatches[index].ids)
    only = [
        [m for m in group.markers.label]
        for im, group in data_detect.groupby("images")
    ]
    locs = []
    for im in range(Ni):
        trash, trash, loc = np.intersect1d(
            only[im], composites[index].ids, return_indices=True
        )
        locs.append(loc)
    return locs


class PoseBatch(gbu.core.Container):
    """
    A class to store camera pose batches.
    """

    def __init__(self, ids, rvecs, tvecs, label=None):
        self.ids = np.array(ids)
        self.rvecs = np.array(rvecs)
        self.tvecs = np.array(tvecs)
        self.label = label

    def __repr__(self):
        return "<Gbu Pose batch w. {0} entries>".format(len(self.ids))

    def __str__(self):
        return "Gbu Pose batch\n" + self.as_dataframe().__str__()

    def as_dataframe(self):
        """
        Returns the content of the batch as a pandas.DataFrame.
        """
        drvecs = pd.DataFrame(
            index=self.ids,
            columns=pd.MultiIndex.from_product([["rvec"], list("xyz")]),
            data=self.rvecs,
        )
        dtvecs = pd.DataFrame(
            index=self.ids,
            columns=pd.MultiIndex.from_product([["tvec"], list("xyz")]),
            data=self.tvecs,
        )
        data = pd.concat([drvecs, dtvecs, ], axis=1)
        data.index.name = "pose"
        return data


class ImageBatchCalibration(gbu.utils.ImageBatch):
    """
    A subclass of ImageBatch class for composite marker calibration.
    """

    def __init__(self,
                 composites=[],
                 poseBatches=[],
                 history=[],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.composites = composites
        self.poseBatches = poseBatches
        self.history = history

        if 'directory' in kwargs:
            self.estimate_pose()
            self.get_graph_data(criterion=0.5)
            self.graph_calibration()
            self.optimize_calibration()

    def graph_calibration(self, reference=None):
        cpose, mpose, graph, multigraph = graph_calibration(
            data=self.data_graph, reference=reference
        )

        self.centralMarker = mpose.index[mpose.central][0]
        self.graph = graph
        self.multigraph = multigraph
        # COMPOSITE MARKER CONSTRUCTION
        cids = np.array([m for m in mpose.index])
        crvecs = mpose.rvec.values.astype(np.float64)
        ctvecs = mpose.tvec.values.astype(np.float64)
        dimensions = np.array([self.marker_dimension[m] for m in cids])
        composite = gbu.core.CompositeMarker(
            ids=cids,
            dimensions=dimensions,
            rvecs=crvecs,
            tvecs=ctvecs)
        # POSE CALIBRATION CONSTRUCTION
        pids = np.array([m for m in cpose.index])
        prvecs = cpose.rvec.values.astype(np.float64)
        ptvecs = cpose.tvec.values.astype(np.float64)
        poseBatch = PoseBatch(
            ids=pids,
            rvecs=prvecs,
            tvecs=ptvecs)
        self.composites.append(composite)
        self.poseBatches.append(poseBatch)
        self.history.append("graph")
        return composite, poseBatch

    def plot_graph(self, path=None, *arg, **kwargs):
        """
        Plots the graph.
        """
        gbu.utils.plot_graph(graph=self.graph, path=path, *arg, **kwargs)

    def observed_points_2D(self):
        """
        Returns the observed 2D points in data.
        """
        data = self.data_graph
        return data.p2d.values.reshape(len(data), 4, 2)

    def markers_on_images(self, index=-1):
        """
        Returns the location of individual markers on the composite in each
        image.
        """
        return gbu.calibration.markers_on_images(index=index,
                                                 data_detect=self.data_graph,
                                                 poseBatches=self.poseBatches,
                                                 composites=self.composites)

    def project_composite(self, index=-1, locs=None):
        """
        Projects the child composite for each image. The result can be compared
        directly with data.
        """
        composite = self.composites[index]
        pose = self.poseBatches[index]
        rvecs = pose.rvecs
        tvecs = pose.tvecs
        p2d = composite.project(
            rvecs, tvecs, self.camera_matrix, self.distortion_coefficients
        )
        if locs is None:
            locs = self.markers_on_images(index=index)
        p2d = [p2d[i][locs[i]] for i in range(len(rvecs))]
        return p2d

    def optimize_calibration(self, from_index=-1):
        """
        Optimizes the calibation using LM on reprojection error.
        """
        # INTERNAL USE FUNCTIONS
        def expand_inputs(X, nonRefMarkers, compo, poseBatch):
            """
            Inverse of *flatten_inputs*
            """
            crvecs = poseBatch.rvecs
            ctvecs = poseBatch.tvecs
            loc = 0
            Nm = len(nonRefMarkers) - 1
            Ni = len(crvecs)
            compo.rvecs[nonRefMarkers] = X[loc: loc + Nm * 3].reshape(Nm, 3)
            loc += Nm * 3
            compo.tvecs[nonRefMarkers] = X[loc: loc + Nm * 3].reshape(Nm, 3)
            loc += Nm * 3
            crvecs[:] = X[loc: loc + Ni * 3].reshape(Ni, 3)
            loc += Ni * 3
            ctvecs[:] = X[loc: loc + Ni * 3].reshape(Ni, 3)

        def flatten_inputs(nonRefMarkers, compo, poseBatch):
            """
            Flattens the inputs for optimization purposes.
            """
            crvecs = poseBatch.rvecs
            ctvecs = poseBatch.tvecs
            return np.concatenate(
                [
                    compo.rvecs[nonRefMarkers].flatten(),
                    compo.tvecs[nonRefMarkers].flatten(),
                    crvecs.flatten(),
                    ctvecs.flatten(),
                ]
            )

        def cost_function(X, nonRefMarkers, p2do, locs, compo, poseBatch):
            """
            Least square optimization function.
            """
            expand_inputs(X, nonRefMarkers, compo, poseBatch)
            p2dflat = np.concatenate(self.project_composite(index=-1,
                                                            locs=locs))
            return (p2dflat - p2do).flatten().astype(np.float64)

        # ACTUAL OPTIMIZATION
        if len(self.composites) == 0:
            self.graph_calibration()
        self.composites.append(self.composites[from_index].copy())
        self.poseBatches.append(self.poseBatches[from_index].copy())
        self.history.append("optim")
        compo = self.composites[-1]
        poseBatch = self.poseBatches[-1]
        locs = self.markers_on_images()
        p2do = self.observed_points_2D()
        centralMarker = self.centralMarker
        nonRefMarkers = ((compo.ids == centralMarker) == False)
        args = (nonRefMarkers, p2do, locs, compo, poseBatch,)
        X0 = flatten_inputs(nonRefMarkers, compo, poseBatch)
        self.t0 = time.time()
        t = threading.Thread(target=self.print_elapsed_time)
        self._run_flag = True
        t.start()
        sol = optimize.least_squares(
            cost_function, X0,
            method="lm",
            args=args,
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.e-10)
        self._run_flag = False
        t.join()
        expand_inputs(sol.x, nonRefMarkers, compo, poseBatch)
        return compo, poseBatch, sol

    def projected_points(self):
        """
        Returns the reprojection
        """
        data = self.data_graph.set_index(
            [("image", "fname"), ("markers", "label"), ("poses", "")])
        data.index.names = ["image", "marker", "pose"]
        data = data.p2d.swaplevel(0, 1, axis=1).stack()
        data.columns = pd.MultiIndex.from_product([("observed",), ("x", "y")])
        for i in range(len(self.composites)):
            p2d = self.project_composite(index=i)
            p2d = np.concatenate(np.concatenate(p2d))
            key = "{0}_{1}".format(self.history[i], i)
            data[(key, "x")] = p2d[:, 0]
            data[(key, "y")] = p2d[:, 1]
        return data

    def get_reprojection_errors(self):
        data = self.projected_points()
        data_out = data.observed
        data_out.columns = pd.MultiIndex.from_product(
            [("observed",), ("x", "y")])
        keys = [k for k in data.columns.levels[0] if k is not "observed"]

        for key in keys:
            df = data[key] - data.observed
            data_out[key, "x"] = df.x.values
            data_out[key, "y"] = df.y.values
        data_out.drop('observed', inplace=True, axis=1)
        return data_out

    def plot_reprojection_errors(self,
                                 global_plot=True,
                                 individual_plots=True,
                                 marker_cycle="x+1234",
                                 color_cycle="bgmcy",
                                 plot_type="classic"):
        """
        Plots reprojected points.
        """
        data = self.projected_points()
        keys = [k for k in data.columns.levels[0] if k is not "observed"]
        if global_plot:
            if plot_type == "classic":
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_aspect("equal")
                plt.title("Reprojection Errors")
                plt.plot(data.observed.x.values,
                         data.observed.y.values,
                         color="r",
                         marker="o",
                         linestyle="None",
                         label="observed",
                         mfc="none")

                for i in range(len(keys)):
                    plt.plot(data[keys[i]].x.values,
                             data[keys[i]].y.values,
                             color=color_cycle[i % len(color_cycle)],
                             marker=marker_cycle[i % len(marker_cycle)],
                             linestyle="None",
                             label=keys[i])
                plt.legend(loc="best")
                plt.grid()
                plt.ylabel("pixels")
                plt.xlabel("pixels")

                fig = plt.figure()
                ax2 = fig.add_subplot(1, 1, 1)
                plt.title("Reprojection Residuals")
                ax2.set_aspect("equal")
                for i in range(len(keys)):
                    plt.plot(
                        data[keys[i]].x.values - data.observed.x.values,
                        data[keys[i]].y.values - data.observed.y.values,
                        color=color_cycle[i % len(color_cycle)],
                        marker=marker_cycle[i % len(marker_cycle)],
                        linestyle="None",
                        label="{0} - observed".format(keys[i]))
                plt.ylabel("pixels")
                plt.xlabel("pixels")
                plt.grid()
                plt.legend()
                plt.tight_layout()
                plt.show()
            if plot_type == "interactive":
                import plotly.express as px
                import plotly.graph_objects as go
                dim = len(data)
                marker_cycle = ['x-thin-open',
                                'cross-thin-open']
                color_cycle = ['blue', 'green', ' magenta', 'cyan', 'yellow']
                imgs, markers, poses, corners = zip(
                    *data.index.values.tolist())
                d_obs = dict({"x": data.observed.x.values,
                              "y": data.observed.y.values,
                              "kind": ["obs"] * dim,
                              "pose": poses,
                              "image": imgs,
                              "marker": markers,
                              "corner": corners,
                              })
                df_obs = pd.DataFrame(data=d_obs)

                df_list = []
                for i in range(len(keys)):
                    imgs, markers, poses, corners = zip(
                        *data[keys[i]].index.values.tolist())
                    dic = dict({"x": data[keys[i]].x.values,
                                "y": data[keys[i]].y.values,
                                "kind": [keys[i]] * dim,
                                "pose": poses,
                                "image": imgs,
                                "marker": markers,
                                "corner": corners, })
                    df = pd.DataFrame(data=dic)
                    df_list.append(df)

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=df_obs['x'], y=df_obs['y'],
                               mode='markers',
                               marker_symbol='circle-open',
                               marker=dict(color='red'),
                               name="observed",
                               text="Pose: " + df_obs["pose"].astype(str) +
                               "<br>Image: " + df_obs["image"].astype(str) +
                               "<br>Marker: " + df_obs["marker"].astype(str) +
                               "<br>Corner: " + df_obs["corner"].astype(str),
                               ))
                Nl = len(df_list)
                pbar = tqdm(range(Nl))
                for ii in pbar:
                    pbar.set_description("Plotting: {0}".format(keys[ii]))
                    fig.add_trace(
                        go.Scatter(x=df_list[ii]['x'], y=df_list[ii]['y'],
                                   mode='markers',
                                   marker_symbol=marker_cycle[ii % len(
                                       marker_cycle)],
                                   marker=dict(
                                       color=color_cycle[ii % len(color_cycle)]),
                                   name=keys[ii],
                                   text="Pose: " + df_list[ii]["pose"].astype(str) +
                                   "<br>Image: " + df_list[ii]["image"].astype(str) +
                                   "<br>Marker: " + df_list[ii]["marker"].astype(str) +
                                   "<br>Corner: " +
                                   df_list[ii]["corner"].astype(str),
                                   ))

                fig.update_layout(
                    title="Reprojection Errors",
                    font=dict(color='#000000'),
                    showlegend=True,
                    xaxis_title="pixels",
                    yaxis_title="pixels",)
                fig.update_xaxes(showline=True,
                                 linecolor='black',
                                 showgrid=True,
                                 gridcolor='#000000')
                fig.update_yaxes(showline=True,
                                 linecolor='black',
                                 showgrid=True,
                                 gridcolor='#000000')
                fig.show()

                fig = go.Figure()
                for ii in pbar:
                    fig.add_trace(
                        go.Scatter(x=df_list[ii]['x'] - df_obs['x'],
                                   y=df_list[ii]['y'] - df_obs['y'],
                                   mode='markers',
                                   name="{0} - observed".format(keys[ii]),
                                   marker_symbol=marker_cycle[ii % len(
                                       marker_cycle)],
                                   marker=dict(
                                       color=color_cycle[ii % len(color_cycle)]),
                                   text="Pose: " + df_list[ii]["pose"].astype(str) +
                                   "<br>Image: " + df_list[ii]["image"].astype(str) +
                                   "<br>Marker: " + df_list[ii]["marker"].astype(str) +
                                   "<br>Corner: " +
                                   df_list[ii]["corner"].astype(str),
                                   ))

                fig.update_layout(
                    title="Reprojection Residuals",
                    font=dict(color='#000000'),
                    showlegend=True,
                    xaxis_title="pixels",
                    yaxis_title="pixels",)
                fig.update_xaxes(showline=True,
                                 linecolor='black',
                                 showgrid=True,
                                 gridcolor='#000000')
                fig.update_yaxes(showline=True,
                                 linecolor='black',
                                 showgrid=True,
                                 gridcolor='#000000')
                fig.show()
        if individual_plots:
            # IMAGE PER IMAGE PLOT
            marker_cycle = "x+1234"
            color_cycle = "bgmcy"
            quadLoc = np.array([0, 1, 2, 3, 0])
            for image_ind, imData in data.groupby(level=0):
                imagePath = self.data_graph_img.loc[image_ind].image.path.values.astype(str)[
                    0]
                fname = self.data_graph_img.loc[image_ind].image.fname.values.astype(str)[
                    0]
                rgb = io.imread(imagePath + fname)
                if len(rgb.shape) == 2:
                    frame = rgb
                else:
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                plt.figure()
                plt.imshow(frame, cmap=mpl.cm.gray, interpolation="nearest")
                plt.title("{0}".format(imagePath))

                plt.grid()
                plt.colorbar()
                olabel = "observed"
                clabels = copy.copy(keys)
                for marker, markerData in imData.groupby(level=1):
                    plt.plot(
                        markerData.observed.x.values[quadLoc],
                        markerData.observed.y.values[quadLoc],
                        "r-",
                        label=olabel,
                    )
                    olabel = None
                    for i in range(len(keys)):
                        plt.plot(
                            markerData[keys[i]].x.values[quadLoc],
                            markerData[keys[i]].y.values[quadLoc],
                            color_cycle[i % len(color_cycle)] + "-",
                            label=clabels[i],
                        )
                        clabels[i] = None
                plt.legend(loc="best")

            plt.show()

    def filter_reprojection_errors(self, criterion_pixel=0.5, inplace=False):
        data = self.get_reprojection_errors()
        keys = [k for k in data.columns.levels[0] if k.startswith('optim')]

        locs = data.loc[data[keys[-1]].values < criterion_pixel].index.get_level_values(
            'pose').unique().values.tolist()
        if inplace:
            self.data_graph = self.data_graph.loc[locs].sort_index()
            return locs, self.data_graph
        else:
            return locs, self.data_graph.loc[locs]

    def print_elapsed_time(self):
        while self._run_flag:
            print("optimizing\t elapsed time : {0}\ts".format(round(time.time() - self.t0, 3)),
                  end='\r')

    def delete_entry(self, index=-1):
        del self.history[index]
        del self.composites[index]
        del self.poseBatches[index]


class ImageBatchCompositePose(gbu.utils.ImageBatch):
    """
    A classe to estimate the pose of a composite on an image batch.

    Args:
    * composite: a composite instance who's pose has to be estimated.
    * reference: if not None, has to be composite instance used a the reference
    for pose estimation.
    """

    def __init__(self, composites, history=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.composites = composites
        self.history = history

        if 'directory' in kwargs:
            self.load_image_batch(directory=kwargs['directory'])
            self.detect_markers(plot_markers=False, enforce=False)
            self.estimate_composite_pose(
                method="gbu", planar=True, verbose=True)

    def estimate_composite_pose(self, planar=True, verbose=True, noweight=False, *args, **kwargs):
        """
        Estimates the pose of the composite and stores into the poseBatch
        attribute. Extra args. are passed to estimatePose.
        """
        composites = self.composites
        ids3Ddic = {cid: c.ids for cid, c in composites.items()}
        points3Ddic = {cid: c.p3d() for cid, c in composites.items()}
        group = self.data_detect.groupby("images")
        iids = sorted(list(group.groups.keys()))
        pbar = tqdm(iids)
        poses = {cid: {"iids": [], "rvecs": [], "tvecs": []}
                 for cid in composites.keys()}
        for iid in pbar:
            if verbose:
                pbar.set_description(
                    "Estimating composite pose on image: {0}".format(iid))
            idata = group.get_group(iid)
            ids2D = [i for i in idata.markers.label.values]
            points2D = idata.p2d.values.reshape(len(ids2D), 4, 2)
            weights = 1 - \
                (gbu.utils.calculate_inner_angles(
                    idata).angular_deviation.abs_max.values / 45)
            weights = np.repeat(weights, 4)
            if noweight:
                weights = np.ones_like(weights)
            for cid in composites.keys():

                if planar:
                    success, rvec, tvec, sol = gbu.core.estimate_pose_composite_planar(
                        ids2D=ids2D,
                        points2D=points2D,
                        ids3D=ids3Ddic[cid],
                        points3D=points3Ddic[cid],
                        camera_matrix=self.camera_matrix,
                        distortion_coefficients=self.distortion_coefficients,
                        weights=weights)
                else:
                    success, rvec, tvec, sol = gbu.core.estimate_pose_composite(
                        ids2D=ids2D,
                        points2D=points2D,
                        ids3D=ids3Ddic[cid],
                        points3D=points3Ddic[cid],
                        camera_matrix=self.camera_matrix,
                        distortion_coefficients=self.distortion_coefficients,
                        weights=weights)
                if success:
                    poses[cid]["iids"].append(iid)
                    poses[cid]["rvecs"].append(rvec)
                    poses[cid]["tvecs"].append(tvec)

        poseBatches = {}
        for cid in composites.keys():
            pb = gbu.calibration.PoseBatch(ids=poses[cid]["iids"],
                                           rvecs=poses[cid]["rvecs"],
                                           tvecs=poses[cid]["tvecs"],
                                           label=cid)

            poseBatches[cid] = pb
            if not cid in set(self.history):
                self.history.append(cid)
        self.poseBatches = poseBatches

    def project_composite(self, key=None, locs=None):
        """
        Projects the child composite for each image. The result can be compared
        directly with data.
        """
        composite = self.composites[key]
        pose = self.poseBatches[key]
        rvecs = pose.rvecs
        tvecs = pose.tvecs
        p2d = composite.project(
            rvecs, tvecs, self.camera_matrix, self.distortion_coefficients
        )
        if locs is None:
            locs = self.markers_on_images(key=key)
        p2d = [p2d[i][locs[i]] for i in range(len(rvecs))]
        return p2d

    def markers_on_images(self, key=None):
        """
        Returns the location of individual markers on the composite in each
        image.
        """
        locs = gbu.calibration.markers_on_images(index=key,
                                                 data_detect=self.data_detect,
                                                 poseBatches=self.poseBatches,
                                                 composites=self.composites)
        return locs

    def projected_points(self):
        """
        Returns the reprojection
        """
        data_detect = self.data_detect.copy()
        data_detect.reset_index('detects', inplace=True)
        data = data_detect.set_index(
            [("images"), ("markers", "label"), ("detects")])
        data.index.names = [('image'), ('marker'), ('detect')]
        data = data.p2d.swaplevel(0, 1, axis=1).stack()
        data.columns = pd.MultiIndex.from_product([("observed",), ("x", "y")])
        for key in self.composites.keys():
            p2d = self.project_composite(key=key)
            p2d = np.concatenate(np.concatenate(p2d))
            data[(key, "x")] = p2d[:, 0]
            data[(key, "y")] = p2d[:, 1]
        return data

    def get_reprojection_errors(self):
        data = self.projected_points()
        data_out = data.observed
        data_out.columns = pd.MultiIndex.from_product(
            [("observed",), ("x", "y")])
        keys = [k for k in data.columns.levels[0] if k is not "observed"]

        for i in range(len(keys)):
            df = data[keys[i]] - data.observed
            data_out[(keys[i], "x")] = df.x.values
            data_out[(keys[i], "y")] = df.y.values
        data_out.drop('observed', inplace=True, axis=1)
        return data_out

    def plot_reprojection_errors(self,
                                 global_plot=True,
                                 individual_plots=True,
                                 marker_cycle="x+1234",
                                 color_cycle="bgmcy",
                                 plot_type="classic"):
        """
        Plots reprojected points.
        """
        data = self.projected_points()
        keys = [k for k in data.columns.levels[0] if k is not "observed"]
        if global_plot:
            if plot_type == "classic":
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.set_aspect("equal")
                plt.title("Reprojection Errors")
                plt.plot(data.observed.x.values,
                         data.observed.y.values,
                         color="r",
                         marker="o",
                         linestyle="None",
                         label="observed",
                         mfc="none")

                for i in range(len(keys)):
                    plt.plot(data[keys[i]].x.values,
                             data[keys[i]].y.values,
                             color=color_cycle[i % len(color_cycle)],
                             marker=marker_cycle[i % len(marker_cycle)],
                             linestyle="None",
                             label=keys[i])
                plt.legend(loc="best")
                plt.grid()
                plt.ylabel("pixels")
                plt.xlabel("pixels")

                fig = plt.figure()
                ax2 = fig.add_subplot(1, 1, 1)
                plt.title("Reprojection Residuals")
                ax2.set_aspect("equal")
                for i in range(len(keys)):
                    plt.plot(
                        data[keys[i]].x.values - data.observed.x.values,
                        data[keys[i]].y.values - data.observed.y.values,
                        color=color_cycle[i % len(color_cycle)],
                        marker=marker_cycle[i % len(marker_cycle)],
                        linestyle="None",
                        label="{0} - observed".format(keys[i]))
                plt.ylabel("pixels")
                plt.xlabel("pixels")
                plt.grid()
                plt.legend()
                plt.tight_layout()
                plt.show()
            if plot_type == "interactive":
                import plotly.express as px
                import plotly.graph_objects as go
                dim = len(data)
                marker_cycle = ['x-thin-open',
                                'cross-thin-open']
                color_cycle = ['blue', 'green', ' magenta', 'cyan', 'yellow']
                imgs, markers, detects, corners = zip(
                    *data.index.values.tolist())
                d_obs = dict({"x": data.observed.x.values,
                              "y": data.observed.y.values,
                              "kind": ["obs"] * dim,
                              "detect": detects,
                              "image": self.data_img.image.fname.loc[np.array(imgs)],
                              "marker": markers,
                              "corner": corners,
                              })
                df_obs = pd.DataFrame(data=d_obs)

                df_list = []
                for i in range(len(keys)):
                    imgs, markers, detects, corners = zip(
                        *data[keys[i]].index.values.tolist())
                    dic = dict({"x": data[keys[i]].x.values,
                                "y": data[keys[i]].y.values,
                                "kind": [keys[i]] * dim,
                                "detect": detects,
                                "image": self.data_img.image.fname.loc[np.array(imgs)],
                                "marker": markers,
                                "corner": corners, })
                    df = pd.DataFrame(data=dic)
                    df_list.append(df)

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(x=df_obs['x'], y=df_obs['y'],
                               mode='markers',
                               marker_symbol='circle-open',
                               marker=dict(color='red'),
                               name="observed",
                               text="Image: " + df_obs["image"].astype(str) +
                               "<br>Detect: " + df_obs["detect"].astype(str) +
                               "<br>Marker: " + df_obs["marker"].astype(str) +
                               "<br>Corner: " + df_obs["corner"].astype(str),
                               ))
                Nl = len(df_list)
                pbar = tqdm(range(Nl))
                for ii in pbar:
                    pbar.set_description("Plotting: {0}".format(keys[ii]))
                    fig.add_trace(
                        go.Scatter(x=df_list[ii]['x'], y=df_list[ii]['y'],
                                   mode='markers',
                                   marker_symbol=marker_cycle[ii % len(
                                       marker_cycle)],
                                   marker=dict(
                                       color=color_cycle[ii % len(color_cycle)]),
                                   name=keys[ii],
                                   text="<br>Image: " + df_list[ii]["image"].astype(str) +
                                   "<br>Detect: " + df_list[ii]["detect"].astype(str) +
                                   "<br>Marker: " + df_list[ii]["marker"].astype(str) +
                                   "<br>Corner: " +
                                   df_list[ii]["corner"].astype(str),
                                   ))

                fig.update_layout(
                    title="Reprojection Errors",
                    font=dict(color='#000000'),
                    showlegend=True,
                    xaxis_title="pixels",
                    yaxis_title="pixels",)
                fig.update_xaxes(showline=True,
                                 linecolor='black',
                                 showgrid=True,
                                 gridcolor='#000000')
                fig.update_yaxes(showline=True,
                                 linecolor='black',
                                 showgrid=True,
                                 gridcolor='#000000')
                fig.show()

                fig = go.Figure()
                for ii in pbar:
                    fig.add_trace(
                        go.Scatter(x=df_list[ii]['x'] - df_obs['x'],
                                   y=df_list[ii]['y'] - df_obs['y'],
                                   mode='markers',
                                   name="{0} - observed".format(keys[ii]),
                                   marker_symbol=marker_cycle[ii % len(
                                       marker_cycle)],
                                   marker=dict(
                                       color=color_cycle[ii % len(color_cycle)]),
                                   text="<br>Image: " + df_list[ii]["image"].astype(str) +
                                   "<br>Detect: " + df_list[ii]["detect"].astype(str) +
                                   "<br>Marker: " + df_list[ii]["marker"].astype(str) +
                                   "<br>Corner: " +
                                   df_list[ii]["corner"].astype(str),
                                   ))

                fig.update_layout(
                    title="Reprojection Residuals",
                    font=dict(color='#000000'),
                    showlegend=True,
                    xaxis_title="pixels",
                    yaxis_title="pixels",)
                fig.update_xaxes(showline=True,
                                 linecolor='black',
                                 showgrid=True,
                                 gridcolor='#000000')
                fig.update_yaxes(showline=True,
                                 linecolor='black',
                                 showgrid=True,
                                 gridcolor='#000000')
                fig.show()
        if individual_plots:
            # IMAGE PER IMAGE PLOT
            marker_cycle = "x+1234"
            color_cycle = "bgmcy"
            quadLoc = np.array([0, 1, 2, 3, 0])
            for image_ind, imData in data.groupby(level=0):
                imagePath = self.data_graph_img.loc[image_ind].image.path.values.astype(str)[
                    0]
                fname = self.data_graph_img.loc[image_ind].image.fname.values.astype(str)[
                    0]
                rgb = io.imread(imagePath + fname)
                if len(rgb.shape) == 2:
                    frame = rgb
                else:
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                plt.figure()
                plt.imshow(frame, cmap=mpl.cm.gray, interpolation="nearest")
                plt.title("{0}".format(imagePath))

                plt.grid()
                plt.colorbar()
                olabel = "observed"
                clabels = copy.copy(keys)
                for marker, markerData in imData.groupby(level=1):
                    plt.plot(
                        markerData.observed.x.values[quadLoc],
                        markerData.observed.y.values[quadLoc],
                        "r-",
                        label=olabel,
                    )
                    olabel = None
                    for i in range(len(keys)):
                        plt.plot(
                            markerData[keys[i]].x.values[quadLoc],
                            markerData[keys[i]].y.values[quadLoc],
                            color_cycle[i % len(color_cycle)] + "-",
                            label=clabels[i],
                        )
                        clabels[i] = None
                plt.legend(loc="best")

            plt.show()

    def filter_reprojection_errors(self, criterion_pixel=1, key=None, inplace=False):
        data = self.get_reprojection_errors()
        if key is None:
            key = [keys for keys in self.composites.keys()][-1]
        locs2kill = data.loc[(abs(data[key]).values > criterion_pixel)].index.get_level_values(
            'detect').unique().values.tolist()

        locs = set(self.data_detect.index.values) - set(locs2kill)

        if inplace:
            self.data_detect = self.data_detect.loc[locs].sort_index()
            return locs, self.data_detect
        else:
            return locs, self.data_detect.loc[locs]

    def draw_xyz_composite(self, key=None, *args, **kwargs):
        if key is None:
            key = [keys for keys in self.poseBatches.keys()][-1]
        data_img = self.data_img.copy()
        data_img.columns = data_img.columns.droplevel(level=2)
        poseBatch = self.poseBatches[key].as_dataframe()
        poseBatch.index.names = ["images"]
        data_xyz = gbu.utils.merge_dataFrames(poseBatch, data_img)
        self.draw_xyz(data_xyz, *args, **kwargs)

    def plot_3D_composite(self, keys=None):

        if keys is None:
            keys = [keys for keys in self.composites.keys()]

        import numpy as np
        import plotly.graph_objects as go

        for key in keys:
            # x, y, z = zip(*self.composites[key].tvecs)

            fig = go.Figure(data=[go.Scatter3d(
                x=self.composites[key].as_dataframe(
                ).tvec.x.values.astype(np.float64),
                y=self.composites[key].as_dataframe(
                ).tvec.y.values.astype(np.float64),
                z=self.composites[key].as_dataframe(
                ).tvec.z.values.astype(np.float64),
                mode='markers',
                marker=dict(
                    size=12,
                    # color=self.composites[key].as_dataframe().tvec.z.values.astype(
                    #     np.float64),                # set color to an array/list of desired values
                    # colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )
            )])
            fig.add_trace(
                go.Scatter3d(
                    x=self.composites[key].p3d().reshape(-1, 3)[:, 0],
                    y=self.composites[key].p3d().reshape(-1, 3)[:, 1],
                    z=self.composites[key].p3d().reshape(-1, 3)[:, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        # color=self.composites[key].as_dataframe().tvec.z.values.astype(
                        #     np.float64),                # set color to an array/list of desired values
                        # colorscale='Viridis',   # choose a colorscale
                        opacity=0.8
                    )
                ))

        fig.show()

    def geometrical_features(self,
                             kind='point',
                             key_target='target',
                             key_ref='ref',
                             key_new_target='new_target'):
        """

        """
        needed_keys = [key_target, key_ref]
        keys = [key for key in needed_keys if key in self.composites.keys()]

        if len(keys) > 0:
            if len(keys) == 1:
                new_compo_target = gbu.calibration.composite_geometrical_feature(
                    compo_target=self.composites[key_target],
                    targetPoseBatch=self.poseBatches[key_target],
                    refPoseBatch=None,
                    kind=kind)

            if len(keys) == 2:
                new_compo_target = gbu.calibration.composite_geometrical_feature(
                    compo_target=self.composites[key_target],
                    targetPoseBatch=self.poseBatches[key_target],
                    refPoseBatch=self.poseBatches[key_ref],
                    kind=kind)

            self.composites.update({key_new_target: new_compo_target})
            self.estimate_composite_pose(method="gbu")
            return new_compo_target
        else:
            print("Composites keywords wrong !")
            return None

    def change_reference_frame(self, cid_in: str, cid_out: str):
        Ni = len(self.poseBatches[cid_in].ids)
        rvecs = np.zeros([Ni, 3])
        tvecs = np.zeros([Ni, 3])
        for i in range(Ni):
            rvecs[i], tvecs[i] = gbu.utils.change_reference(R10=self.poseBatches[cid_out].rvecs[i],
                                                            T10=self.poseBatches[cid_out].tvecs[i],
                                                            R20=self.poseBatches[cid_in].rvecs[i],
                                                            T20=self.poseBatches[cid_in].tvecs[i])
        return rvecs, tvecs


class MultiImageBatchCompositePose():

    def __init__(self,
                 composites,
                 parameters,
                 out_directory="./output/",
                 camera_matrix=np.zeros(3),
                 distortion_coefficients=np.zeros(5),
                 *args, **kwargs):

        self.composites = composites
        self.parameters = parameters
        self.out_directory = out_directory
        self.camera_matrix = camera_matrix
        self.distorsion_coefficients = distortion_coefficients
        self.kwargs = kwargs
        self.entries = {}

    def add_batch(self, directory: str, batch_name: str):
        batch = gbu.calibration.ImageBatchCompositePose(composites=self.composites,
                                                        parameters=self.parameters,
                                                        out_directory=self.out_directory,
                                                        camera_matrix=self.camera_matrix,
                                                        distortion_coefficients=self.distorsion_coefficients,
                                                        directory=directory
                                                        )
        self.entries.update({batch_name: batch})


class CompositeMetricCalibration():
    """
    A subclass of ImageBatchCompositePose class for composite marker metric calibration.
    """

    def __init__(self, composite,
                 parameters=None,
                 output_directory="./_outputs/",
                 camera_matrix=np.zeros(3),
                 distortion_coefficients=np.zeros(3),
                 *args, **kwargs):

        self.composite = composite
        self.output_directory = output_directory
        self.camera_matrix = camera_matrix
        self.distorsion_coefficients = distortion_coefficients
        self.kwargs = kwargs

        if parameters is None:
            self.default_aruco_parameters()
        else:
            self.parameters = parameters

        self.targets = []

    def default_aruco_parameters(self):
        self.parameters = aruco.DetectorParameters_create()
        self.parameters.cornerRefinementMethod = 3
        self.parameters.cornerRefinementWinSize = 5
        self.parameters.cornerRefinementMaxIterations = 100

    def load_target_batchs(self,
                           path_start_position="./image_start_folder/",
                           path_end_position="./image_end_folder/",
                           format=".tif",
                           target_dist=1e-2):
        """[summary]

        Args:
            path_start_position (str): path to start position image batch folder. Defaults to "./image_start_folder/".
            path_end_position (str): path to end position image batch folder. Defaults to "./image_end_folder/".
            target_dist ([float]): [composite distance between the 2 batchs in meter]. Defaults to 1e-2.
        """
        nb_ref = len((
            [f for f in os.listdir(path_start_position) if f.endswith(format)]))
        nb_pos = len((
            [f for f in os.listdir(path_end_position) if f.endswith(format)]))

        dic = {"path_start": path_start_position,
               "path_end": path_end_position,
               "nb_ref": nb_ref,
               "nb_pos": nb_pos,
               "target_dist": target_dist}

        self.targets.append(dic)

    def calibrate(self):

        targets = self.targets
        compo = self.composite
        parameters = self.parameters
        camera_matrix = self.camera_matrix
        distorsion_coefficients = self.distorsion_coefficients

        def distance_2_batches(compo, path_0, path_1):
            batch_0 = gbu.calibration.ImageBatchCompositePose(
                composites={"target": compo},
                directory=path_0,
                parameters=parameters,
                camera_matrix=camera_matrix,
                distortion_coefficients=distorsion_coefficients)

            batch_1 = gbu.calibration.ImageBatchCompositePose(
                composites={"target": compo},
                directory=path_1,
                parameters=parameters,
                camera_matrix=camera_matrix,
                distortion_coefficients=distorsion_coefficients)

            pos_0 = batch_0.poseBatches['target'].tvecs
            pos_1 = batch_1.poseBatches['target'].tvecs

            if len(pos_0) > len(pos_1):
                pos_0 = pos_0[:len(pos_1)]
            else:
                pos_1 = pos_1[:len(pos_0)]

            dist = np.linalg.norm(pos_0 - pos_1, axis=1)

            return dist

        def cost(k):
            Nt = len(targets)
            res = np.zeros([targets[0]['nb_ref'], Nt])
            compo_opt = compo.copy()
            compo_opt.dimensions = compo.dimensions * k[0]
            compo_opt.tvecs = compo.tvecs * k[0]
            for i in range(Nt):
                res[:, i] = targets[i]['target_dist'] - distance_2_batches(compo=compo_opt,
                                                                           path_0=targets[i]['path_start'],
                                                                           path_1=targets[i]['path_end'])

            return np.array(res).flatten()

        k0 = np.ones(3)
        self.sol = optimize.least_squares(cost, k0, method='lm', xtol=1.e-12,
                                          ftol=1.e-12,
                                          gtol=1.e-10)
        print(self.sol)
        print("Optimal Marker dimension size : {0} [mm]".format(
            self.sol.x[0] * self.composite.dimensions[0] * 1e3))

        self.composite_metric = set_composite_metric(
            compo, marker_coeff=self.sol.x[0])

        return self.composite_metric
