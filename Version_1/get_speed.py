from pymavlink import mavutil
import time

# Connect to PX4 (replace with your connection details)
connection_string = 'udp:localhost:14540'  # Change if needed
master = mavutil.mavlink_connection(connection_string)

# Wait for a heartbeat to confirm connection
master.wait_heartbeat()
print("Connected to PX4.")

# Request data stream to get speed information
master.mav.request_data_stream_send(master.target_system, master.target_component,
                                    mavutil.mavlink.MAV_DATA_STREAM_ALL, 1, 1)

# Non-blocking mode: Check for data every 0.1 seconds
while True:
    # Poll for new messages without blocking
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)

    if msg:
        # Extract speed (in meters per second)
        speed_x = msg.vx / 100  # Convert from cm/s to m/s
        speed_y = msg.vy / 100
        speed_z = msg.vz / 100

        # Calculate the overall speed
        speed = (speed_x**2 + speed_y**2 + speed_z**2)**0.5

        # Print speed on the same line
        print(f"Speed (m/s): {speed:.2f}", end='\r')

    # Sleep for a short time to avoid 100% CPU usage
    time.sleep(0.1)
