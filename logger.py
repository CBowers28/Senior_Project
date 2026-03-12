import csv
import os
from datetime import datetime


class GazeLogger:

    HEADERS = ["timestamp", "norm_x", "norm_y",
               "world_x", "world_y", "confidence", "markers_visible"]

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(log_dir, f"gaze_log_{ts}.csv")
        self._f    = open(self._path, "w", newline="")
        self._w    = csv.writer(self._f)
        self._w.writerow(self.HEADERS)
        self._count = 0
        print(f"[logger] Logging to {self._path}")

    def log(self, timestamp, norm_x, norm_y, world_x, world_y,
            confidence, markers_visible):
        self._w.writerow([
            f"{timestamp:.4f}",
            f"{norm_x:.4f}",
            f"{norm_y:.4f}",
            f"{world_x:.4f}" if world_x is not None else "",
            f"{world_y:.4f}" if world_y is not None else "",
            f"{confidence:.3f}",
            markers_visible,
        ])
        self._count += 1

    def close(self):
        self._f.close()
        print(f"[logger] Saved {self._count} gaze points → {self._path}")