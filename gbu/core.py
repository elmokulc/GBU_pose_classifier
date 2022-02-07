# Filename: <core.py>
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
# GBU/CORE
################################################################################

import copy
import json

import cv2
################################################################################
# IMPORTS
import numpy as np
import pandas as pd
from cv2 import aruco
from scipy import optimize

import gbu

################################################################################


################################################################################
# MARKER DETECTION
def get_corners(dimension=1.0):
    """
    Returns a square marker's corners.
    """
    if len(np.shape(dimension)) == 0:
        d = np.array([dimension])
    else:
        d = np.array(dimension)
    c = np.array(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]]
    )
    return d[:, None, None] * c


def detect_markers(
    frame, dictionary=aruco.DICT_6X6_250, parameters=None, drop_duplicates=True
):
    """
    Detects markers using the opencv/Aruco dedicated function.

    Inputs:
    * frame: a grayscale 2D array.
    * dictionary: a aruco dictionary instance.
    * parameters: detector parameters.
    * drop_duplicates: if True, markers detected several times on a single frame
      are dropped.

    Outpus:
    * ids: a N int array containing the ids of the N markers.
    * corners: a Nx4x2 containing the pixel coordinates of the 4 corners of each
      N markers.
    """
    if parameters is None:
        parameters = aruco.DetectorParameters_create()
        parameters.cornerRefinementMethod = 1
        parameters.cornerRefinementWinSize = 2
        parameters.cornerRefinementMaxIterations = 10

    corners, ids, rejected = aruco.detectMarkers(
        frame, dictionary=aruco.Dictionary_get(dictionary), parameters=parameters
    )
    if len(corners) != 0:
        # Remove second dimension which is useless
        ids = ids.flatten()
        corners = np.array(corners).squeeze(axis=1)
        if drop_duplicates:
            unique, index, counts = np.unique(
                ids, return_index=True, return_counts=True
            )
            loc = index[counts == 1]
            corners = corners[loc]
            ids = ids[loc]
            return True, ids, corners
    else:
        return False, None, None


def marker_area(corners):
    """
    Calculates the area of the markers in pixels**2.

    Inputs:
        * corners: Nm x 4 x 2 float array. Can be used directly after detect_markers.

    Outputs:
        * Nm float array of areas.
    """
    Nc = len(corners)
    areas = np.zeros(Nc)
    for i in range(Nc):
        c = corners[i]
        c = c[1:] - c[0]
        areas[i] = (np.linalg.det(c[:-1]) + np.linalg.det(c[1:])) / 2.
    return areas


def marker_angles(corners, deviation=False, abs_max=False):
    """
    Calculates the 8 inner angles between the edges and diagonals of the markers in pixels**2.

    Inputs:
        * corners: Nm x 4 x 2 float array. Can be used directly after detect_markers.

    Outputs:
        * Nm x 8 float array of angles in degrees.
        * deviation: (boolean) if True, angles are expressed as deviation vs optimal value.
        * abs_max: (boolean) if True, only the maximum absolute value is kept.
    """
    Nc = len(corners)
    angles = np.zeros((Nc, 8))
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
        B = corners[:, v[0]]
        A = corners[:, v[1]]
        C = corners[:, v[2]]
        AB = B - A
        AC = C - A
        AB /= np.linalg.norm(AB, axis=1)[:, None]
        AC /= np.linalg.norm(AC, axis=1)[:, None]
        angles[:, i] = -np.degrees(np.arcsin(np.cross(AB, AC)))
    if deviation:
        angles -= 45.
    if abs_max:
        angles = np.abs(angles).max(axis=1)
    return angles


def _cost_pnp(X, p3d, p2dt, weights, camera_matrix, distortion_coefficients):
    """
    Reprojection error cost function used in customized solve PNP.
    """
    rvec, tvec = X.reshape(2, 3)
    proj = cv2.projectPoints(
        p3d, rvec, tvec, camera_matrix, distortion_coefficients)
    p2dp = proj[0].reshape(-1, 2)
    return ((p2dt - p2dp)  # .T * weights
            ).flatten()


def estimate_pose(
    points2D,
    points3D,
    camera_matrix=np.eye(3),
    distortion_coefficients=np.zeros(5),
    method="gbu",
    weights=1.0,
    rvec0=np.array([0.0, 0.0, 0.0]),
    tvec0=np.array([0.0, 0.0, 1.0]),
    **kwargs
):
    """
    Estimates the pose.

    Inputs:
    * points2D: 2D positions of marker corners on a frame.
    * points3D: 3D positions of markers contained in a composite marker.
    * camera_matrix(3x3 array_like): camera matrix.
    * distortion_coefficients (array-like): distortion coeffcients.
    * method: can be "cv2" for the standard cv2.solvePNP method or "gbu" for
      the custom method based on the sicpy Levenberg-Marquardt solver. This second method can take extra arguments.
    * weights: float (default 1.) or array with shape (N,) where N is len(ids2D) (gbu only).
    * rvec0: 1d array with 3 elements, intial guess for rvec (gbu only).
    * tvec0: 1d array with 3 elements, intial guess for tvec (gbu only).
    ** kwargs: extra arguments passed to scipy.optimize.least_squares (gbu only).
    """
    if method == "cv2":
        ret, rvec, tvec = cv2.solvePnP(
            np.concatenate(points3D),
            np.concatenate(points2D),
            camera_matrix,
            distortion_coefficients,
        )
        if ret:
            return True, rvec.flatten(), tvec.flatten(), None
        else:
            return False, None, None, None
    if method == "gbu":
        dt = np.float64
        p3d = points3D.astype(dt)
        p2dt = points2D.astype(dt)
        weights = np.array(weights)
        if len(weights.shape) != 0:
            weights = weights.astype(dt)
        X0 = np.concatenate([rvec0, tvec0]).astype(dt)
        sol = optimize.least_squares(
            _cost_pnp,
            X0,
            args=(p3d, p2dt, weights, camera_matrix,
                  distortion_coefficients),
            method="lm",
            **kwargs
        )
        if not sol.success:
            return False, np.ones(3) * np.nan, np.ones(3) * np.nan, sol
        srvec, stvec = sol.x.reshape(2, 3)
        angle = np.linalg.norm(srvec) % (2 * np.pi)
        if angle >= np.pi:
            angle -= 2. * np.pi
        srvec = srvec / np.linalg.norm(srvec)
        srvec = srvec * angle
        s = np.sign(stvec[2])
        srvec *= s
        stvec *= s
        return True, srvec, stvec, sol
    else:
        print("Composite object can't be found !")
        return False, np.ones(3) * np.nan,  np.ones(3) * np.nan, None


def estimate_pose_planar(*args, **kwargs):
    """
    Planar pose estimation.

    Inputs: See estimate_pose_single.
    Outputs:
    * status (int): 0=full success, 1=fail on second estimation, 2=total failure
    * rvec0
    * tvec0
    * rvec1
    * tvec1
    * sol0
    * sol1
    """
    kwargs["method"] = "gbu"
    success0, rvec0, tvec0, sol0 = estimate_pose(*args, **kwargs)
    if success0:
        Pt = np.outer(tvec0, tvec0.T) / np.dot(tvec0, tvec0)
        rvec1_guess = np.dot((2 * Pt - np.eye(3)), rvec0)
        kwargs["rvec0"] = rvec1_guess
        kwargs["tvec0"] = tvec0
        success1, rvec1, tvec1, sol1 = estimate_pose(*args, **kwargs)

        if success1:
            return 0, rvec0, tvec0, rvec1, tvec1, sol0, sol1
        else:
            return 1, rvec0, tvec0, np.ones(3) * np.nan, np.ones(3) * np.nan, sol0, None
    else:
        return 2, np.ones(3) * np.nan, np.ones(3) * np.nan, np.ones(3) * np.nan, np.ones(3) * np.nan, None, None


def estimate_pose_composite(ids2D,
                            points2D,
                            ids3D,
                            points3D,
                            camera_matrix,
                            distortion_coefficients,
                            rvec0=np.array([0.0, 0.0, 0.0]),
                            tvec0=np.array([0.0, 0.0, 1.0]),
                            weights=1.0):

    inter, loc3D, loc2D = np.intersect1d(
        ids3D, ids2D, return_indices=True)

    p3d = points3D[loc3D].reshape(-1, 3)
    p2d = points2D[loc2D].reshape(-1, 2)
    # print("ids2D", np.array(ids2D).shape)
    # print("ids3D", np.array(ids3D).shape)
    # print("points2D whitout reshape", points2D[loc1D].shape)
    # print("points2D", points2D[loc1D].reshape(-1, 2).shape)
    # print("p3d whitout reshape", p3d.shape)
    # print("p3d", p3d.reshape(-1, 3).shape)

    success, rvec, tvec, sol = estimate_pose(
        points2D=p2d,
        points3D=p3d,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        method="gbu",
        rvec0=rvec0,
        tvec0=tvec0,
        weights=weights)

    return success, rvec, tvec, sol


def estimate_pose_composite_planar(ids2D,
                                   points2D,
                                   ids3D,
                                   points3D,
                                   camera_matrix,
                                   distortion_coefficients,
                                   weights=1.0):
    success0, rvec0, tvec0, sol0 = estimate_pose_composite(
        ids2D,
        points2D,
        ids3D,
        points3D,
        camera_matrix,
        distortion_coefficients)
    if success0:
        Pt = np.outer(tvec0, tvec0.T) / np.dot(tvec0, tvec0)
        rvec1_guess = np.dot((2 * Pt - np.eye(3)), rvec0)

        success1, rvec1, tvec1, sol1 = estimate_pose_composite(
            ids2D=ids2D,
            points2D=points2D,
            ids3D=ids3D,
            points3D=points3D,
            camera_matrix=camera_matrix,
            distortion_coefficients=distortion_coefficients,
            rvec0=rvec1_guess,
            tvec0=tvec0,
            weights=weights)

        if success1 and sol0.cost < sol1.cost or not success1:
            return True, rvec0, tvec0, sol0
        else:
            return True, rvec1, tvec1, sol1
    else:
        return False, np.ones(3) * np.nan, np.ones(3) * np.nan, None


def marker_area(points2D):
    """
    Calculates the area of each markers in pixels squared.

    Inputs:
    * points2D: Nx4x2 array.

    Outputs:
    areas: N array
    """
    p = points2D[:, 1:] - points2D[:, :1]
    return (np.linalg.det(p[:, :-1]) + np.linalg.det(p[:, 1:])) / 2


################################################################################

################################################################################
# COMPOSITE MARKER


class Container:
    """
    Container meta class.
    """

    def copy(self):
        """
        Returns a copy of the instance.
        """
        return copy.deepcopy(self)


def save_composite(composite, path=None):
    """
    Returns a composite as a JSON string. If path
    is not None, saves it to path.
    """
    out = {'ids': composite.ids.tolist()}
    out["rvecs"] = composite.rvecs.tolist()
    out["tvecs"] = composite.tvecs.tolist()
    out["dimensions"] = composite.dimensions.tolist()
    out["label"] = composite.label
    outs = json.dumps(out, indent=2)
    if path is not None:
        open(path, "w").write(outs)
    return outs


def load_composite(path):
    """
    Loads a composite JSON file.
    """
    dic = json.loads(open(path).read())
    return CompositeMarker(**dic)


class CompositeMarker(Container):
    """
    COMPOSITE MARKER
    A class to manage composite markers.
    """

    def __init__(self, ids, dimensions, rvecs, tvecs, label=None, *args, **kwargs):
        self.ids = np.array(ids)
        self.dimensions = np.array(dimensions)
        self.rvecs = np.array(rvecs)
        self.tvecs = np.array(tvecs)
        self.label = label

    def __repr__(self):
        return "<Gbu Composite Marker w. {0} markers>".format(len(self.ids))

    def __str__(self):
        return "Gbu Composite Marker\n" + self.as_dataframe().__str__()

    def as_dataframe(self):
        """
        Returns the content of the composite marker as a pandas.DataFrame.
        """
        ddimensions = pd.DataFrame(
            index=self.ids,
            columns=pd.MultiIndex.from_product([["dimension"], [""]]),
            data=self.dimensions,
        )
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
        data = pd.concat([ddimensions, drvecs, dtvecs, ], axis=1)
        data.index.name = "marker"
        return data

    def p3d(self):
        """
        Returns the 3D points of the marker in its own frame.
        """
        Nm = len(self.ids)
        p3di = get_corners(self.dimensions)
        p3dc = np.zeros_like(p3di)
        rvecs = self.rvecs
        tvecs = self.tvecs
        for i in range(Nm):
            p3dc[i] = gbu.utils.apply_RT(p3di[i], rvecs[i], tvecs[i])
        return p3dc

    def apply_RT(self, R=np.zeros(3), T=np.zeros(3), inplace=False):
        """
        Applies a RT transformation to the composite marker.
        """
        out = self.copy() if not inplace else self
        R = np.array(R)
        T = np.array(T)
        rvecs = out.rvecs
        tvecs = out.tvecs
        for i in range(len(rvecs)):
            rvecs[i], tvecs[i] = gbu.utils.compose_RT(
                rvecs[i], tvecs[i], R, T)
        if not inplace:
            return out

    def project(
        self,
        rvecs=np.zeros(3),
        tvecs=np.zeros(3),
        cameraMatrix=np.eye(3),
        distortionCoefficients=np.zeros(5),
    ):
        """
        Projects the composite on a camera with given pose (rvec, tvec).
        Multiple Ni poses can be used simultanerously, in this case, rvec and
        rvec must have Ni x 3 shape.
        """
        if len(rvecs.shape) == 1:
            rvecs = rvecs.reshape(1, 3)
            tvecs = tvecs.reshape(1, 3)
            concat = True
        p3d = self.p3d()  # 3D corners
        Ni = len(rvecs)
        Nm = len(p3d)
        p3dconc = np.concatenate(p3d)
        p2dp = np.zeros((Ni, Nm, 4, 2))
        for im in range(Ni):
            projectedPoints = cv2.projectPoints(
                p3dconc, rvecs[im], tvecs[im], cameraMatrix, distortionCoefficients
            )[0]
            p2dp[im] = projectedPoints.reshape(Nm, 4, 2)
        return p2dp

    def estimate_pose(self, ids2D, points2D, cameraMatrix=np.eye(3), distortionCoefficients=np.zeros(5), *args, **kwargs):
        """
        Estimates the pose of the composite using detected markers.
        """
        return estimate_pose_composite(
            ids2D=ids2D,
            points2D=points2D,
            ids3D=self.ids.copy(),
            points3D=self.p3d(),
            camera_matrix=cameraMatrix,
            distortion_coefficients=distortionCoefficients,
            *args,
            **kwargs,
        )

    def save(self, path=None):
        """
        Returns a composite as a JSON string. If path
        is not None, saves it to path.
        """
        return save_composite(self, path)

    def set_marker_reference(self, mk_ref=None, inplace=False):
        """
        Sets a given marker as the reference of the composite. The associated
        RT transfomation is calculated and applied.
        """
        Rr = np.array(self.as_dataframe().rvec.loc[mk_ref])
        Tr = np.array(self.as_dataframe().tvec.loc[mk_ref])
        Rri, Tri = gbu.utils.invert_RT(R=Rr, T=Tr)
        return self.apply_RT(R=Rri, T=Tri, inplace=inplace)
