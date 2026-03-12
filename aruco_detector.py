import cv2
import numpy as np
import zmq

# Create AprilTag / ArUco detector with predefined dictionary
_aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
_aruco_params = cv2.aruco.DetectorParameters()

# Adjust detection sensitivity settings
_aruco_params.minMarkerPerimeterRate = 0.05
_aruco_params.minCornerDistanceRate  = 0.05
_aruco_params.minDistanceToBorder    = 3

# Create detector object
_detector = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)

MIN_MARKER_AREA = 800  # Minimum marker size in pixels


def recv_frame(sub: zmq.Socket, flags: int = 0) -> tuple:
    import msgpack

    # Receive metadata and image bytes from ZMQ socket
    meta_bytes = sub.recv(flags=flags)
    meta = msgpack.unpackb(meta_bytes, raw=False)
    img_bytes  = sub.recv(flags=flags)

    fmt = meta.get("format", "jpeg")
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)

    # Decode frame depending on format
    if fmt == "jpeg":
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    elif fmt in ("bgr", "rgb"):
        h, w  = meta["height"], meta["width"]
        frame = img_array.reshape((h, w, 3))
        if fmt == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    elif fmt == "gray":
        h, w  = meta["height"], meta["width"]
        frame = cv2.cvtColor(img_array.reshape((h, w)), cv2.COLOR_GRAY2BGR)

    elif fmt in ("yuv422", "yuyv"):
        h, w  = meta["height"], meta["width"]
        frame = cv2.cvtColor(
            img_array.reshape((h, w, 2)),
            cv2.COLOR_YUV2BGR_YUY2
        )

    else:
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Return None if decode failed
    if frame is None:
        return None, None

    return meta, frame


def detect_markers(frame: np.ndarray) -> dict:
    # Detect markers in frame
    corners, ids, _ = _detector.detectMarkers(frame)

    result = {}

    if ids is None:
        return result

    # Filter markers by area size
    for i, corner in enumerate(corners):
        pts = corner[0]

        if cv2.contourArea(pts) < MIN_MARKER_AREA:
            continue

        result[int(ids[i][0])] = pts.astype(np.float64)

    return result


def draw_markers(frame: np.ndarray, detected: dict) -> np.ndarray:
    # Draw detected markers on a copy of the frame
    out = frame.copy()

    for mid, pts in detected.items():

        # Draw box around marker
        cv2.polylines(
            out,
            [pts.astype(np.int32).reshape(-1, 1, 2)],
            True,
            (0, 255, 0),
            2
        )

        # Draw center point
        center = pts.mean(axis=0).astype(int)
        cv2.drawMarker(
            out,
            tuple(center),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            16,
            2
        )

        # Draw marker ID text
        cv2.putText(
            out,
            f"ID:{mid}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return out