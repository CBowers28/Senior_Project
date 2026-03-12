# Gaze Mapper - Real-Time Eye Tracking to World Coordinates

**Contributors: Christopher Bowers, Sanjeev Kamath, Grant Collins, Cody Coyle**

A Python-based eye-tracking system that maps gaze coordinates from a Pupil Capture eye tracker to real-world 2D coordinates using ArUco marker detection and perspective homography.

## Overview

This project enables real-time gaze point estimation on a physical scene by:

1. **Receiving eye tracking data** from Pupil Capture via ZMQ networking
2. **Detecting ArUco markers** in the world camera feed
3. **Computing perspective homography** to transform normalized gaze coordinates to world 2D coordinates
4. **Rendering live visualization** of detected markers and gaze points
5. **Logging gaze data** to CSV with confidence metrics

## Project Structure

- **`main.py`** - Main application loop; connects to Pupil Capture, processes frames, maps gaze, and handles visualization
- **`aruco_detector.py`** - ArUco marker detection using OpenCV with configurable detection parameters
- **`gaze_mapper.py`** - Homography-based gaze mapping; transforms pixel/normalized coordinates to world coordinates
- **`renderer.py`** - Renders a 2D world plane visualization showing marker positions and gaze point
- **`logger.py`** - CSV-based gaze logging with timestamp and confidence tracking
- **`world_config.py`** - Configuration loader for world markers (from XML) and camera intrinsics
- **`world_markers.xml`** - XML file defining ArUco marker IDs and their real-world 2D positions (in cm)
- **`Aruco_Marker_Detection.py`** - Helper/alternative marker detection module
- **`Pupil_data_loader.py`** - Helper for loading Pupil Capture data
- **`Pupil_data_loader.py`** - Gaze data loader for offline analysis

## Requirements

- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- ZMQ (`zmq`, `msgpack`)
- Pupil Capture running locally or on a networked machine

## Setup

1. **Configure world markers**: Edit `world_markers.xml` to define your ArUco marker positions in the real world.
   ```xml
   <marker id="123" world_x="10.0" world_y="15.5" world_z="0.0"/>
   ```

2. **Camera intrinsics** (optional): Place camera calibration file at `~/pupil_capture_settings/world.intrinsics`, or the system will use default intrinsics.

3. **Start Pupil Capture** on `localhost:50020` or update `PUPIL_REMOTE_ADDR` in `main.py`.

## Usage

```bash
python main.py
```

The application will display:
- **World Plane window**: 2D visualization of marker locations and the current gaze point
- **Scene Camera window**: Live camera feed with detected markers highlighted

Press **`q`** to quit.

## How It Works

### Marker Detection
ArUco markers (AprilTag 36h11) are detected in each frame using OpenCV's detector. Only markers with sufficient area and valid geometry are tracked.

### Gaze Mapping
The system computes a perspective homography from detected marker centers (pixel space) to known world positions (cm). This requires a minimum of 4 visible markers. With more visible markers, the solve becomes overdetermined and more robust.

- Normalized gaze coordinates (0-1) are converted to pixel coordinates based on frame resolution
- The homography matrix projects these into world 2D space
- Gaze points are only logged when confidence exceeds the threshold (default: 0.6)

### Data Logging
All gaze samples are logged to `gaze_logs/gaze_log_YYYYMMDD_HHMMSS.csv` with columns:
- `timestamp` - Pupil Capture timestamp
- `norm_x, norm_y` - Normalized gaze coordinates
- `world_x, world_y` - World 2D coordinates (or empty if homography unavailable)
- `confidence` - Eye tracking confidence
- `markers_visible` - Number of detected markers

## Configuration

Key parameters in `main.py`:
- `CONFIDENCE_MIN` - Minimum gaze confidence to log (default: 0.6)
- `PUPIL_REMOTE_ADDR` - Pupil Capture server address
- `XML_PATH` - Path to world markers configuration
- `INTRINSICS_PATH` - Path to camera calibration file

## Status Indicators

The world plane visualization shows:
- **Green circles**: Currently visible markers (with IDs)
- **Blue circles**: Configured markers not currently visible
- **Red square**: Current gaze point (when homography is valid)
- **Stats panel**: Active marker count, total logged points, pose source

---

**Senior Design Project** - Spring 2026
