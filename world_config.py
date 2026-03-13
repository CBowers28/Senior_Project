import xml.etree.ElementTree as ET
import numpy as np


def load_world_markers(xml_path: str) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    markers = {}
    for elem in root.findall("marker"):
        mid     = int(elem.get("id"))
        world_x = float(elem.get("world_x"))
        world_y = float(elem.get("world_y"))
        world_z = float(elem.get("world_z", 0.0))
        markers[mid] = (world_x, world_y, world_z)
    print(f"[world_config] Loaded {len(markers)} markers from {xml_path}")
    for mid, pos in markers.items():
        print(f"  Marker {mid}: ({pos[0]}, {pos[1]}) cm")
    return markers


def load_camera_intrinsics(path: str):
    try:
        import msgpack
        with open(path, "rb") as f:
            data = msgpack.unpackb(f.read(), raw=False)
        key = list(data.keys())[0]
        intrinsics = data[key]
        K = np.array(intrinsics["camera_matrix"], dtype=np.float64)
        D = np.array(intrinsics["dist_coefs"],    dtype=np.float64)
        print(f"[world_config] Loaded intrinsics from {path}")
        return K, D
    except FileNotFoundError:
        return default_intrinsics()


def default_intrinsics():
    K = np.array([
        [910.0,   0.0, 640.0],
        [  0.0, 910.0, 360.0],
        [  0.0,   0.0,   1.0],
    ], dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    return K, D