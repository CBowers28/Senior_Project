import zmq
import msgpack
import time

print("Starting Pupil Service with full gaze pipeline...")
print("Make sure:")
print("  1. Pupil Core is plugged in")
print("  2. Calibration was completed previously in Pupil Capture")
print()

try:
    # Create ZMQ context and connect it to the Pupil Service
    ctx = zmq.Context()
    pupil_remote = ctx.socket(zmq.REQ)
    pupil_remote.connect('tcp://localhost:50020')

    print("[1/6] Connected to Pupil Service")


    # function for convenience
    def send_recv_notification(n):
        pupil_remote.send_string(f"notify.{n['subject']}", flags=zmq.SNDMORE)
        pupil_remote.send(msgpack.dumps(n))
        return pupil_remote.recv_string()


    # Start eye process 0
    print("[2/6] Starting eye process 0 (right eye)...")
    n = {'subject': 'eye_process.should_start.0', 'eye_id': 0, 'args': {}}
    print(f"     {send_recv_notification(n)}")

    # Start eye process 1
    print("[3/6] Starting eye process 1 (left eye)...")
    n = {'subject': 'eye_process.should_start.1', 'eye_id': 1, 'args': {}}
    print(f"     {send_recv_notification(n)}")

    # Wait for eye processes
    time.sleep(3)

    # Start the calibration plugin (loads previous calibration)
    print("[4/6] Loading calibration...")
    n = {'subject': 'start_plugin', 'name': 'Calibration_Choreography', 'args': {}}
    print(f"     {send_recv_notification(n)}")

    time.sleep(1)

    # Start the gaze mapper (converts pupil data to gaze)
    print("[5/6] Starting gaze mapper...")
    n = {'subject': 'start_plugin', 'name': 'Gaze_Mapper_2D', 'args': {}}
    print(f"     {send_recv_notification(n)}")

    time.sleep(2)

    # Get the subscriber port
    print("[6/6] Getting subscriber port...")
    pupil_remote.send_string('SUB_PORT')
    sub_port = pupil_remote.recv_string()
    print(f"     Subscriber port: {sub_port}")

    # Subscribe to gaze
    sub_socket = ctx.socket(zmq.SUB)
    sub_socket.connect(f'tcp://127.0.0.1:{sub_port}')
    sub_socket.subscribe(b'gaze')

    print()
    print("✓ Full gaze pipeline initialized!")
    print("✓ Streaming gaze data... (Press Ctrl+C to stop)")
    print()

    time.sleep(0.5)

    msg_count = 0
    no_data_count = 0

    while True:
        try:
            topic = sub_socket.recv_string(flags=zmq.NOBLOCK)
            payload = sub_socket.recv()
            message = msgpack.unpackb(payload, raw=False)

            msg_count += 1
            no_data_count = 0

            if 'norm_pos' in message:
                x, y = message['norm_pos']
                confidence = message.get('confidence', 0)
                print(f"[{msg_count}] Gaze: x={x:.3f}, y={y:.3f}, confidence={confidence:.2f}")

        except zmq.Again:
            no_data_count += 1

            if no_data_count == 100:
                print("⚠ No gaze data yet...")
                print("   Trying to subscribe to pupil data instead to verify eye tracking works...")
                # Also try subscribing to pupil data to debug
                sub_socket.subscribe(b'pupil')

            time.sleep(0.01)
            continue

except KeyboardInterrupt:
    print("\n\nStopping...")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
finally:
    print("Cleaning up...")
    try:
        sub_socket.close()
        pupil_remote.close()
        ctx.term()
    except:
        pass
    print("Disconnected")