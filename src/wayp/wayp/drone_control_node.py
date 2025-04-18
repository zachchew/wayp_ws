#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw, VelocityBodyYawspeed
import numpy as np
import math
import time
from std_msgs.msg import Int32MultiArray

class DroneController(Node):
    def __init__(self):
        super().__init__('drone_controller')
        
        # Create subscription to robot pose and visible markers
        self.pose_subscription = self.create_subscription(
            PoseStamped,
            '/robot_pose',
            self.pose_callback,
            10)
        
        self.visible_markers_subscription = self.create_subscription(
            Int32MultiArray,
            '/visible_marker_ids',
            self.visible_markers_callback,
            10)
        
        # Initialize drone state variables
        self.current_pose = None
        self.last_valid_pose = None
        self.last_pose_time = 0
        self.visible_markers = []
        self.mission_state = "INIT"
        self.target_waypoint = 0
        self.waypoint_reached = False
        self.hover_start_time = 0
        self.drone = None
        self.dead_reckoning_active = False
        self.last_velocity_command = [0.0, 0.0, 0.0, 0.0]  # [forward, right, down, yaw_rate]
        self.pose_confidence = 1.0  # 1.0 = full confidence, 0.0 = no confidence
        
        # Marker positions in world frame (FRD)
        self.marker_positions = [
            [0.5, 5.0, 1.45],  # Marker 1
            [5.0, 4.5, 1.45],  # Marker 2
            [4.5, 0.0, 1.45],  # Marker 3
            [0.0, 0.5, 1.45]   # Marker 4
        ]
        
        # Waypoints between markers (for navigation when no markers are visible)
        self.intermediate_waypoints = self.generate_intermediate_waypoints()
        
        # Start the drone control loop
        asyncio.ensure_future(self.run())
        
        # Create timer for dead reckoning updates
        self.dead_reckoning_timer = self.create_timer(0.1, self.dead_reckoning_callback)
        
        self.get_logger().info('Drone controller initialized')
    
    def generate_intermediate_waypoints(self):
        """Generate waypoints between markers for navigation without visual markers"""
        waypoints = []
        num_points = 3  # Number of intermediate points between markers
        
        for i in range(len(self.marker_positions)):
            start = self.marker_positions[i]
            end = self.marker_positions[(i + 1) % len(self.marker_positions)]
            
            # Calculate intermediate points
            for j in range(1, num_points + 1):
                t = j / (num_points + 1)
                point = [
                    start[0] + t * (end[0] - start[0]),
                    start[1] + t * (end[1] - start[1]),
                    1.5  # Fixed height
                ]
                waypoints.append(point)
        
        return waypoints
    
    def visible_markers_callback(self, msg):
        """Callback for visible marker IDs"""
        self.visible_markers = msg.data
        
        # Update confidence based on marker visibility
        if len(self.visible_markers) > 0:
            self.dead_reckoning_active = False
            self.pose_confidence = 1.0
        else:
            # If no markers are visible, start decreasing confidence
            if not self.dead_reckoning_active and self.current_pose is not None:
                self.dead_reckoning_active = True
                self.last_valid_pose = self.current_pose
                self.last_pose_time = time.time()
    
    def pose_callback(self, msg):
        """Callback for robot pose updates"""
        # Only update current pose if we have markers visible
        if len(self.visible_markers) > 0:
            self.current_pose = msg
            self.last_valid_pose = msg
            self.last_pose_time = time.time()
            self.dead_reckoning_active = False
            self.pose_confidence = 1.0
        
        # Check if we've reached the target waypoint
        if self.mission_state == "FLYING_TO_MARKER" and self.current_pose is not None:
            # Calculate distance to target position
            marker_pos = self.marker_positions[self.target_waypoint]
            
            # Calculate vector pointing from marker to drone's desired position (0.5m away)
            marker_to_drone_direction = self.calculate_approach_vector(self.target_waypoint)
            target_position = [
                marker_pos[0] + marker_to_drone_direction[0] * 0.5,
                marker_pos[1] + marker_to_drone_direction[1] * 0.5,
                1.5  # Fixed height of 1.5m
            ]
            
            # Calculate distance to target
            current_pos = [
                self.current_pose.pose.position.x,
                self.current_pose.pose.position.y,
                self.current_pose.pose.position.z
            ]
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(current_pos, target_position)))
            
            if distance < 0.2:  # Within 20cm of target
                self.waypoint_reached = True
    
    def dead_reckoning_callback(self):
        """Update position estimate when no markers are visible using dead reckoning"""
        if self.dead_reckoning_active and self.last_valid_pose is not None:
            # Calculate time since last valid pose
            elapsed_time = time.time() - self.last_pose_time
            
            # Decrease confidence over time
            self.pose_confidence = max(0.0, 1.0 - (elapsed_time / 5.0))  # Confidence goes to 0 after 5 seconds
            
            # If we've been in dead reckoning too long, don't update the pose
            if elapsed_time > 10.0:  # After 10 seconds, stop dead reckoning
                self.get_logger().warn("Dead reckoning timeout - no markers detected for too long")
                return
                
            # Create a new pose message based on last velocity command and elapsed time
            estimated_pose = PoseStamped()
            estimated_pose.header = self.last_valid_pose.header
            estimated_pose.header.stamp = self.get_clock().now().to_msg()
            
            # Extract yaw from last valid pose
            q = self.last_valid_pose.pose.orientation
            yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
            
            # Estimate new position based on velocity commands
            forward_vel, right_vel, down_vel, _ = self.last_velocity_command
            
            # Convert body velocities to world frame
            dx = forward_vel * math.cos(yaw) - right_vel * math.sin(yaw)
            dy = forward_vel * math.sin(yaw) + right_vel * math.cos(yaw)
            dz = down_vel
            
            # Update position
            estimated_pose.pose.position.x = self.last_valid_pose.pose.position.x + dx * elapsed_time
            estimated_pose.pose.position.y = self.last_valid_pose.pose.position.y + dy * elapsed_time
            estimated_pose.pose.position.z = self.last_valid_pose.pose.position.z + dz * elapsed_time
            
            # Keep the same orientation
            estimated_pose.pose.orientation = self.last_valid_pose.pose.orientation
            
            # Update current pose with estimate
            self.current_pose = estimated_pose
            
            self.get_logger().info(f"Dead reckoning active - confidence: {self.pose_confidence:.2f}")
    
    def calculate_approach_vector(self, waypoint_idx):
        """Calculate unit vector pointing from marker to desired drone position"""
        # For simplicity, we'll use vectors pointing toward the center of the room
        # This ensures the drone faces the marker when positioned 0.5m away
        marker_pos = self.marker_positions[waypoint_idx]
        room_center = [2.5, 2.5, 1.5]  # Approximate center of the room
        
        # Vector from marker to room center
        vector = [room_center[0] - marker_pos[0], room_center[1] - marker_pos[1]]
        
        # Normalize to unit vector
        magnitude = math.sqrt(vector[0]**2 + vector[1]**2)
        if magnitude > 0:
            return [vector[0]/magnitude, vector[1]/magnitude]
        else:
            return [0, 0]
    
    async def run(self):
        """Main drone control loop"""
        # Connect to the drone
        self.drone = System()
        await self.drone.connect(system_address="udp://:14540")  # Adjust connection string as needed
        
        self.get_logger().info("Waiting for drone connection...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                self.get_logger().info("Drone connected!")
                break
        
        # Set initial mission state
        self.mission_state = "INIT"
        
        # Start the mission state machine
        while True:
            if self.mission_state == "INIT":
                await self.initialize_drone()
            elif self.mission_state == "TAKEOFF":
                await self.takeoff()
            elif self.mission_state == "HOVER_INITIAL":
                await self.hover(5)
            elif self.mission_state == "FLYING_FORWARD":
                await self.fly_forward()
            elif self.mission_state == "FLYING_TO_MARKER":
                await self.fly_to_marker()
            elif self.mission_state == "TURN_RIGHT":
                await self.turn_right()
            elif self.mission_state == "HOVER_AFTER_TURN":
                await self.hover(5)
            elif self.mission_state == "SEARCH_FOR_MARKER":
                await self.search_for_marker()
            elif self.mission_state == "MISSION_COMPLETE":
                await self.land()
                break
            
            # Small delay to prevent CPU overuse
            await asyncio.sleep(0.1)
    
    async def initialize_drone(self):
        """Initialize the drone for flight"""
        self.get_logger().info("Initializing drone...")
        
        # Check if drone has a valid position estimate
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_local_position_ok:
                break
            self.get_logger().info("Waiting for position estimate...")
            await asyncio.sleep(1)
        
        # Arm the drone
        self.get_logger().info("Arming drone...")
        await self.drone.action.arm()
        
        # Transition to takeoff state
        self.mission_state = "TAKEOFF"
    
    async def takeoff(self):
        """Take off and prepare for offboard control"""
        self.get_logger().info("Taking off...")
        
        # Set initial setpoint before enabling offboard mode
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(0.0, 0.0, -1.5, 0.0)  # -1.5m in NED frame is 1.5m above ground
        )
        
        # Start offboard mode
        try:
            await self.drone.offboard.start()
        except OffboardError as error:
            self.get_logger().error(f"Starting offboard mode failed with error: {error}")
            await self.drone.action.land()
            return
        
        # Wait for drone to reach takeoff altitude
        async for position in self.drone.telemetry.position():
            if -position.relative_altitude_m >= 1.4:  # Using >= for safety
                break
            await asyncio.sleep(0.1)
        
        self.get_logger().info("Takeoff complete")
        self.mission_state = "HOVER_INITIAL"
        self.hover_start_time = time.time()
    
    async def hover(self, duration):
        """Hover in place for specified duration"""
        current_time = time.time()
        
        if current_time - self.hover_start_time >= duration:
            self.get_logger().info(f"Completed {duration}s hover")
            
            if self.mission_state == "HOVER_INITIAL":
                self.mission_state = "FLYING_FORWARD"
            elif self.mission_state == "HOVER_AFTER_TURN":
                self.target_waypoint = (self.target_waypoint + 1) % 4
                
                if self.target_waypoint == 0 and self.mission_state != "HOVER_INITIAL":
                    # We've completed the full cycle of 4 markers
                    self.mission_state = "MISSION_COMPLETE"
                else:
                    self.mission_state = "FLYING_FORWARD"
        
        # Maintain current position during hover
        if self.current_pose is not None:
            # Use position control with confidence-based gains
            x = self.current_pose.pose.position.x
            y = self.current_pose.pose.position.y
            z = self.current_pose.pose.position.z
            
            # Extract yaw from quaternion
            q = self.current_pose.pose.orientation
            yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
            
            # Use position control when confidence is high, velocity when low
            if self.pose_confidence > 0.5:
                await self.drone.offboard.set_position_ned(
                    PositionNedYaw(x, y, z, yaw)
                )
                self.last_velocity_command = [0.0, 0.0, 0.0, 0.0]
            else:
                # In low confidence, just try to maintain altitude with zero velocity
                await self.drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                )
                self.last_velocity_command = [0.0, 0.0, 0.0, 0.0]
    
    async def fly_forward(self):
        """Fly forward until a marker is detected"""
        self.get_logger().info("Flying forward to find marker...")
        
        if self.current_pose is not None:
            # Get current position
            current_x = self.current_pose.pose.position.x
            current_y = self.current_pose.pose.position.y
            current_z = self.current_pose.pose.position.z
            
            # Extract yaw from quaternion
            q = self.current_pose.pose.orientation
            yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
            
            # If we have markers visible, transition to marker approach
            if len(self.visible_markers) > 0:
                self.mission_state = "FLYING_TO_MARKER"
                self.waypoint_reached = False
                return
                
            # No markers visible, use velocity control with dead reckoning
            forward_speed = 0.5 * self.pose_confidence  # Reduce speed as confidence decreases
            
            # Calculate forward direction based on yaw (in FRD, forward is along X axis)
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(forward_speed, 0.0, 0.0, 0.0)
            )
            self.last_velocity_command = [forward_speed, 0.0, 0.0, 0.0]
            
            # If confidence gets too low, switch to search pattern
            if self.pose_confidence < 0.2:
                self.mission_state = "SEARCH_FOR_MARKER"
    
    async def search_for_marker(self):
        """Execute a search pattern to find markers when confidence is low"""
        self.get_logger().info("Searching for markers...")
        
        # If we found markers, go back to normal operation
        if len(self.visible_markers) > 0:
            self.mission_state = "FLYING_TO_MARKER"
            self.waypoint_reached = False
            return
            
        # Simple search pattern: rotate slowly in place
        await self.drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, 0.0, 15.0)  # Rotate at 15 deg/s
        )
        self.last_velocity_command = [0.0, 0.0, 0.0, 15.0]
        
        # After a full rotation, try moving to next expected marker position
        await asyncio.sleep(2.0)  # Search for 2 seconds before trying next approach
        
        # If we still don't have markers, try to navigate to the next marker using dead reckoning
        if len(self.visible_markers) == 0:
            # Calculate next marker position
            next_marker = self.marker_positions[self.target_waypoint]
            
            # If we have a current pose (even low confidence), try to navigate there
            if self.current_pose is not None:
                current_pos = [
                    self.current_pose.pose.position.x,
                    self.current_pose.pose.position.y,
                    self.current_pose.pose.position.z
                ]
                
                # Vector to next marker
                dx = next_marker[0] - current_pos[0]
                dy = next_marker[1] - current_pos[1]
                
                # Calculate heading to marker
                target_yaw = np.arctan2(dy, dx)
                
                # Extract current yaw
                q = self.current_pose.pose.orientation
                current_yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
                
                # Yaw difference
                yaw_diff = target_yaw - current_yaw
                if yaw_diff > np.pi:
                    yaw_diff -= 2*np.pi
                elif yaw_diff < -np.pi:
                    yaw_diff += 2*np.pi
                    
                # Set velocity to move toward marker
                yaw_rate = 0.5 * yaw_diff  # Proportional control for yaw
                forward_speed = 0.3  # Slow forward speed during search
                
                await self.drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(forward_speed, 0.0, 0.0, yaw_rate)
                )
                self.last_velocity_command = [forward_speed, 0.0, 0.0, yaw_rate]
            
            # Go back to flying forward after search attempt
            self.mission_state = "FLYING_FORWARD"
    
    async def fly_to_marker(self):
        """Fly to position 0.5m away from the marker"""
        if not self.waypoint_reached:
            self.get_logger().info(f"Flying to marker {self.target_waypoint + 1}...")
            
            if self.current_pose is not None:
                # If we lost markers during approach, switch to search
                if len(self.visible_markers) == 0:
                    self.mission_state = "SEARCH_FOR_MARKER"
                    return
                
                # Calculate target position (0.5m away from marker)
                marker_pos = self.marker_positions[self.target_waypoint]
                approach_vector = self.calculate_approach_vector(self.target_waypoint)
                
                target_x = marker_pos[0] + approach_vector[0] * 0.5
                target_y = marker_pos[1] + approach_vector[1] * 0.5
                target_z = -1.5  # 1.5m above ground in FRD (negative Z is up)
                
                # Calculate yaw to face the marker
                dx = marker_pos[0] - target_x
                dy = marker_pos[1] - target_y
                target_yaw = np.arctan2(dy, dx)
                
                # Use position control with high confidence, velocity with low
                if self.pose_confidence > 0.5:
                    await self.drone.offboard.set_position_ned(
                        PositionNedYaw(target_x, target_y, target_z, target_yaw)
                    )
                    self.last_velocity_command = [0.0, 0.0, 0.0, 0.0]
                else:
                    # Calculate direction to target
                    current_x = self.current_pose.pose.position.x
                    current_y = self.current_pose.pose.position.y
                    
                    # Vector to target in world frame
                    dx = target_x - current_x
                    dy = target_y - current_y
                    
                    # Extract current yaw
                    q = self.current_pose.pose.orientation
                    current_yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
                    
                    # Transform to body frame
                    forward = dx * np.cos(current_yaw) + dy * np.sin(current_yaw)
                    right = -dx * np.sin(current_yaw) + dy * np.cos(current_yaw)
                    
                    # Calculate yaw rate
                    yaw_diff = target_yaw - current_yaw
                    if yaw_diff > np.pi:
                        yaw_diff -= 2*np.pi
                    elif yaw_diff < -np.pi:
                        yaw_diff += 2*np.pi
                    yaw_rate = 0.5 * yaw_diff  # Proportional control
                    
                    # Set velocity command
                    await self.drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(forward, right, 0.0, yaw_rate)
                    )
                    self.last_velocity_command = [forward, right, 0.0, yaw_rate]
        else:
            self.get_logger().info(f"Reached marker {self.target_waypoint + 1}")
            self.mission_state = "TURN_RIGHT"
    
    async def turn_right(self):
        """Turn 90 degrees to the right"""
        self.get_logger().info("Turning right 90 degrees...")
        
        if self.current_pose is not None:
            # Get current position
            current_x = self.current_pose.pose.position.x
            current_y = self.current_pose.pose.position.y
            current_z = self.current_pose.pose.position.z
            
            # Extract current yaw from quaternion
            q = self.current_pose.pose.orientation
            current_yaw = np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))
            
            # Calculate new yaw (90 degrees clockwise in FRD)
            new_yaw = current_yaw - np.pi/2
            if new_yaw < -np.pi:
                new_yaw += 2*np.pi
            
            # Use position control with high confidence, velocity with low
            if self.pose_confidence > 0.5:
                await self.drone.offboard.set_position_ned(
                    PositionNedYaw(current_x, current_y, current_z, new_yaw)
                )
                self.last_velocity_command = [0.0, 0.0, 0.0, 0.0]
            else:
                # Just command a right turn with velocity
                await self.drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, -45.0)  # Turn right at 45 deg/s
                )
                self.last_velocity_command = [0.0, 0.0, 0.0, -45.0]
            
            # Wait for turn to complete (simple delay approach)
            await asyncio.sleep(2)
            
            self.mission_state = "HOVER_AFTER_TURN"
            self.hover_start_time = time.time()
    
    async def land(self):
        """Land the drone and end the mission"""
        self.get_logger().info("Mission complete. Landing...")
        
        # Switch from offboard to land mode
        await self.drone.offboard.stop()
        await self.drone.action.land()
        
        self.get_logger().info("Mission completed successfully!")

def main(args=None):
    rclpy.init(args=args)
    
    # Create the ROS2 node
    drone_controller = DroneController()
    
    # Set up the event loop
    loop = asyncio.get_event_loop()
    
    try:
        # Run the asyncio event loop and ROS2 spin in parallel
        loop.run_until_complete(
            asyncio.gather(
                rclpy.spin(drone_controller),
            )
        )
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        rclpy.shutdown()

if __name__ == '__main__':
    main()
