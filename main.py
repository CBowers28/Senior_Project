import time
import zmq
import msgpack
import cv2
import os

from world_config import load_world_markers, load_camera_intrinsics, default_intrinsics
from aruco_detector import recv_frame, detect_markers, draw_markers
from gaze_mapper import GazeMapper
from renderer import Renderer
from logger import GazeLogger

PUPIL_REMOTE_ADDR = "tcp://localhost:50020"
XML_PATH          = "world_markers.xml"
INTRINSICS_PATH   = "~/pupil_capture_settings/world.intrinsics"
LOG_DIR           = "gaze_logs"
WORLD_WINDOW      = "World Plane"
SCENE_WINDOW      = "Scene Camera"
CONFIDENCE_MIN    = 0.6


def draw_gaze_on_scene(frame, norm_x, norm_y, frame_w, frame_h):
    px  = int(norm_x * frame_w)
    py  = int((1.0 - norm_y) * frame_h)
    out = frame.copy()
    cv2.circle(out, (px, py), 14, (0, 0, 255), -1)
    cv2.circle(out, (px, py), 18, (255, 255, 255), 2)
    cv2.putText(out, f"({norm_x:.3f}, {norm_y:.3f})",
                (px + 20, py + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out


def main():
    world_markers = load_world_markers(XML_PATH)

    intrinsics_path = os.path.expanduser(INTRINSICS_PATH)
    K, D = (load_camera_intrinsics(intrinsics_path)
            if os.path.exists(intrinsics_path)
            else default_intrinsics())

    print("[main] Connecting to Pupil Capture...")
    ctx    = zmq.Context()
    remote = ctx.socket(zmq.REQ)
    remote.connect(PUPIL_REMOTE_ADDR)
    remote.send_string("SUB_PORT")
    sub_port = remote.recv_string()
    print(f"[main] Subscriber port: {sub_port}")

    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://127.0.0.1:{sub_port}")
    sub.subscribe(b"frame.world")
    sub.subscribe(b"gaze")
    print("[main] Subscribed to frame.world + gaze")

    mapper   = GazeMapper(world_markers)
    renderer = Renderer(world_markers)
    logger   = GazeLogger(LOG_DIR)

    latest_scene       = None
    latest_scene_raw   = None
    latest_gaze_world  = None
    latest_norm        = None
    visible_marker_ids = set()
    frame_w, frame_h   = 1280, 720
    total_logged       = 0

    print("[main] Running — press 'q' to quit\n")

    try:
        while True:
            try:
                while True:
                    topic = sub.recv_string(flags=zmq.NOBLOCK)

                    if topic.startswith("frame.world"):
                        meta, frame = recv_frame(sub)
                        if frame is None:
                            continue
                        frame_h, frame_w = frame.shape[:2]
                        detected = detect_markers(frame)
                        mapper.update_pose(detected, frame_h)
                        visible_marker_ids = set(detected.keys())
                        latest_scene = draw_markers(frame, detected)

                        if latest_norm is not None and mapper.has_homography:
                            latest_scene_raw = draw_gaze_on_scene(
                                latest_scene,
                                latest_norm[0], latest_norm[1],
                                frame_w, frame_h)
                        else:
                            latest_scene_raw = latest_scene

                        if not mapper.has_homography:
                            latest_gaze_world = None

                    elif topic.startswith("gaze"):
                        payload = sub.recv(flags=zmq.NOBLOCK)
                        msg     = msgpack.unpackb(payload, raw=False)

                        if "norm_pos" not in msg:
                            continue

                        norm_x, norm_y = msg["norm_pos"]
                        confidence     = msg.get("confidence", 0.0)
                        timestamp      = msg.get("timestamp", time.time())

                        if confidence < CONFIDENCE_MIN:
                            continue

                        latest_norm = (norm_x, norm_y)
                        world_pos   = mapper.project_gaze(norm_x, norm_y,
                                                          frame_w, frame_h)
                        latest_gaze_world = world_pos if mapper.has_homography else None

                        wx = world_pos[0] if world_pos else None
                        wy = world_pos[1] if world_pos else None
                        logger.log(timestamp, norm_x, norm_y, wx, wy,
                                   confidence, mapper.active_marker_count)

                        if world_pos:
                            total_logged += 1

                    else:
                        while sub.getsockopt(zmq.RCVMORE):
                            sub.recv(flags=zmq.NOBLOCK)

            except zmq.Again:
                pass

            ui = renderer.render(
                gaze_world         = latest_gaze_world,
                has_homography     = mapper.has_homography,
                active_markers     = mapper.active_marker_count,
                total_logged       = total_logged,
                pose_source        = mapper.pose_source,
                visible_marker_ids = visible_marker_ids,
            )
            cv2.imshow(WORLD_WINDOW, ui)

            if latest_scene_raw is not None:
                cv2.imshow(SCENE_WINDOW, latest_scene_raw)
            elif latest_scene is not None:
                cv2.imshow(SCENE_WINDOW, latest_scene)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[main] Interrupted")

    finally:
        logger.close()
        sub.close()
        remote.close()
        ctx.term()
        cv2.destroyAllWindows()
        print("[main] Clean shutdown")


if __name__ == "__main__":
    main()