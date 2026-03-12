import cv2
import numpy as np


class Renderer:

    WORLD_W = 800
    WORLD_H = 600
    MARGIN  = 60

    COL_MARKER_VISIBLE = (0, 220, 100)
    COL_MARKER_FIXED   = (60, 60, 120)
    COL_GAZE           = (0, 0, 255)
    COL_GRID           = (35, 35, 35)
    COL_TEXT           = (180, 180, 180)
    COL_BOARD          = (60, 60, 60)
    BADGE_MULTI        = (0, 200, 80)
    BADGE_NONE         = (60, 60, 60)

    def __init__(self, world_markers: dict):
        self._world_markers = world_markers
        xs = [p[0] for p in world_markers.values()]
        ys = [p[1] for p in world_markers.values()]
        self._wx_min, self._wx_max = min(xs), max(xs)
        self._wy_min, self._wy_max = min(ys), max(ys)

    def _world_to_panel(self, wx, wy):
        draw_w = self.WORLD_W - 2 * self.MARGIN
        draw_h = self.WORLD_H - 2 * self.MARGIN
        scale  = min(draw_w / max(self._wx_max - self._wx_min, 1),
                     draw_h / max(self._wy_max - self._wy_min, 1))
        px = int(self.MARGIN + (wx - self._wx_min) * scale)
        # Flip Y so the world Y=0 (bottom) maps to the panel bottom
        py = int(self.WORLD_H - self.MARGIN - (wy - self._wy_min) * scale)
        return px, py

    def render(self, gaze_world, has_homography, active_markers,
               total_logged, pose_source="none", visible_marker_ids=None):

        if visible_marker_ids is None:
            visible_marker_ids = set()

        panel = np.full((self.WORLD_H, self.WORLD_W, 3), (20, 20, 20), dtype=np.uint8)

        for gx in range(0, self.WORLD_W, 80):
            cv2.line(panel, (gx, 0), (gx, self.WORLD_H), self.COL_GRID, 1)
        for gy in range(0, self.WORLD_H, 80):
            cv2.line(panel, (0, gy), (self.WORLD_W, gy), self.COL_GRID, 1)

        # Create outline of the board
        corners = [
            self._world_to_panel(self._wx_min, self._wy_min),
            self._world_to_panel(self._wx_max, self._wy_min),
            self._world_to_panel(self._wx_max, self._wy_max),
            self._world_to_panel(self._wx_min, self._wy_max),
        ]
        cv2.polylines(panel, [np.array(corners, np.int32).reshape(-1, 1, 2)],
                      True, self.COL_BOARD, 1)

        # Markers — these are always shown and will be green if visible
        for mid, (wx, wy, *_) in self._world_markers.items():
            px, py  = self._world_to_panel(wx, wy)
            visible = mid in visible_marker_ids
            col     = self.COL_MARKER_VISIBLE if visible else self.COL_MARKER_FIXED
            thick   = 2 if visible else 1
            cv2.rectangle(panel, (px-10, py-10), (px+10, py+10), col, thick)
            cv2.putText(panel, f"M{mid}", (px+14, py+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        # Gaze dot
        if gaze_world is not None and has_homography:
            gx, gy = self._world_to_panel(*gaze_world[:2])
            gx = int(np.clip(gx, 5, self.WORLD_W - 5))
            gy = int(np.clip(gy, 5, self.WORLD_H - 5))
            cv2.circle(panel, (gx, gy), 12, self.COL_GAZE, -1)
            cv2.circle(panel, (gx, gy), 16, (255, 255, 255), 1)
            cv2.putText(panel,
                        f"({gaze_world[0]:.1f}, {gaze_world[1]:.1f}) cm",
                        (gx + 18, gy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COL_GAZE, 1)

        # Status bar
        badge_col = self.BADGE_MULTI if pose_source == "multi" else self.BADGE_NONE
        cv2.rectangle(panel, (0, self.WORLD_H - 28),
                      (self.WORLD_W, self.WORLD_H), (15, 15, 15), -1)
        cv2.putText(panel,
                    f"Markers: {active_markers}/4  |  "
                    f"Pose: {pose_source.upper()}  |  Logged: {total_logged}",
                    (10, self.WORLD_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_col, 1)
        cv2.putText(panel, "WORLD PLANE", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COL_TEXT, 1)

        return panel