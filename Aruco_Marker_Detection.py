import cv2
import numpy as np

print("Accessing Pupil Core scene camera with ArUco detection...")
print("Make sure Pupil Capture is NOT running")
print()

camera_idx = 0  # Change this if needed (try 0, 1, 2)

cap = cv2.VideoCapture(camera_idx, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print(f"Cannot open camera {camera_idx}")
    exit()

# Set the Full HD resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# Verify settings
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Camera settings:")
print(f"  Resolution: {actual_width}x{actual_height}")
print(f"  FPS: {actual_fps}")
print(f"  Press 'q' to quit")
print()

# Initialize ArUco detector with stricter parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()

# Make detection more conservative to reduce false positives
aruco_params.minMarkerPerimeterRate = 0.08  # Increase from default 0.03
aruco_params.minCornerDistanceRate = 0.05
aruco_params.minDistanceToBorder = 3

detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Minimum marker area in pixels² (adjust this value based on your markers)
MIN_MARKER_AREA = 1000  # Start with this, increase if still getting false positives

print("ArUco detector initialized (DICT_APRILTAG_36h11)")
print(f"Minimum marker area: {MIN_MARKER_AREA} pixels²")
print("Looking for markers...")
print()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(frame)

    # Filter markers by size
    if ids is not None and len(ids) > 0:
        filtered_corners = []
        filtered_ids = []

        for i, corner in enumerate(corners):
            # Calculate marker area
            marker_area = cv2.contourArea(corner[0])

            # Debug: print all detected markers with their areas
            # Uncomment this line to see what's being detected:
            # print(f"Marker {ids[i][0]}: area = {marker_area:.0f} pixels²")

            # Only keep markers above the minimum area
            if marker_area >= MIN_MARKER_AREA:
                filtered_corners.append(corner)
                filtered_ids.append(ids[i])

        # Only process if we have valid markers after filtering
        if len(filtered_ids) > 0:
            filtered_corners = tuple(filtered_corners)
            filtered_ids = np.array(filtered_ids)

            # Draw detected markers on the frame
            cv2.aruco.drawDetectedMarkers(frame, filtered_corners, filtered_ids)

            # Print detected marker IDs
            detected_ids = filtered_ids.flatten().tolist()
            print(f"Detected markers: {detected_ids}")

            # Draw marker IDs on frame
            for i, corner in enumerate(filtered_corners):
                # Get the center of the marker
                center = corner[0].mean(axis=0).astype(int)
                marker_id = filtered_ids[i][0]

                # Draw the ID text
                cv2.putText(frame, f"ID: {marker_id}",
                            tuple(center),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2)

    cv2.imshow('Pupil Scene Camera - ArUco Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()