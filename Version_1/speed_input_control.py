from pymavlink import mavutil
import time

# Connect to PX4
connection_string = 'udp:localhost:14540'
master = mavutil.mavlink_connection(connection_string)

# Wait for heartbeat
master.wait_heartbeat()
print("Connected to PX4!")

def change_speed(speed):
    """
    Send DO_CHANGE_SPEED command
    speed: ground speed in m/s
    """
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
        0,  # confirmation
        1,  # ground speed
        speed,  # speed in m/s
        -1,  # no change to throttle
        0,  # absolute speed
        0, 0, 0  # unused parameters
    )
    print(f"Speed change command sent: {speed} m/s")

try:
    while True:
        new_speed = input("Enter new speed (m/s) or 'q' to quit: ")
        
        if new_speed.lower() == 'q':
            break
            
        try:
            speed_value = float(new_speed)
            change_speed(speed_value)
            
            # Get current ground speed feedback
            msg = master.recv_match(type='VFR_HUD', blocking=True, timeout=1)
            if msg:
                print(f"Current ground speed: {msg.groundspeed:.2f} m/s")
                
        except ValueError:
            print("Please enter a valid number")
            
except KeyboardInterrupt:
    print("\nProgram ended by user")

print("Program completed")
