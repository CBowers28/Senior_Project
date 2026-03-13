"""
gaze_mapper.py
--------------
Homography-based gaze mapping.

Uses detected marker centers (pixel) vs known world positions (cm)
to compute a perspective homography H: pixel -> world.

Requires minimum 4 visible markers for a valid homography.
With more markers visible the solve is overdetermined and more stable.

Gaze (norm_pos) is converted to pixels then transformed via H.
"""

import cv2
import numpy as np

SMOOTH    = 0.15  # EMA alpha on homography matrix
MIN_MARKERS = 4   # minimum markers needed for homography


class GazeMapper:

    def __init__(self, world_markers, camera_matrix=None, dist_coeffs=None):
        self._world_markers = world_markers  # { id: (wx, wy, wz) }
        self._world_centers = {
            mid: np.array([wx, wy], dtype=np.float64)
            for mid, (wx, wy, *_) in world_markers.items()
        }

        self._H              = None
        self._active_markers = 0
        self._pose_source    = "none"

    def update_pose(self, detected: dict) -> bool:
        """
        Compute pixel->world homography from detected marker centers.
        detected: { marker_id: corners (4x2 float) }
        """
        known = {mid: corners for mid, corners in detected.items()
                 if mid in self._world_centers}

        if len(known) < MIN_MARKERS:
            self._H              = None
            self._active_markers = len(known)
            self._pose_source    = "none"
            return False

        # Use marker centers as correspondences
        img_pts   = np.array([known[mid].mean(axis=0) for mid in known],
                             dtype=np.float64)
        world_pts = np.array([self._world_centers[mid] for mid in known],
                             dtype=np.float64)

        H, mask = cv2.findHomography(
            img_pts.reshape(-1, 1, 2),
            world_pts.reshape(-1, 1, 2),
            cv2.RANSAC, 5.0
        )

        if H is None:
            return False

        # EMA smoothing
        if self._H is not None:
            H = (1 - SMOOTH) * self._H + SMOOTH * H

        self._H              = H
        self._active_markers = len(known)
        self._pose_source    = "multi"
        return True

    def project_gaze(self, norm_x, norm_y, frame_w, frame_h):
        """
        Project norm_pos gaze to world coords (cm) via homography.
        norm_pos convention: origin bottom-left, Y up → flip Y for pixels.
        """
        if self._H is None:
            return None

        px = norm_x * frame_w
        py = (1.0 - norm_y) * frame_h  # flip Y

        pt       = np.array([[[px, py]]], dtype=np.float64)
        world_pt = cv2.perspectiveTransform(pt, self._H)

        wx = float(world_pt[0][0][0])
        wy = float(world_pt[0][0][1])
        return wx, wy

    @property
    def has_pose(self):
        return self._H is not None

    @property
    def has_homography(self):
        return self._H is not None

    @property
    def active_marker_count(self):
        return self._active_markers

    @property
    def pose_source(self):
        return self._pose_source