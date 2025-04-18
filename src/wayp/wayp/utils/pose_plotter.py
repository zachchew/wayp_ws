import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time

class PoseConsistencyPlotter:
    def __init__(self, buffer_size=100):
        # Initialize data buffers
        self.buffer_size = buffer_size
        self.timestamps = deque(maxlen=buffer_size)
        self.x_values = deque(maxlen=buffer_size)
        self.y_values = deque(maxlen=buffer_size)
        self.z_values = deque(maxlen=buffer_size)
        self.qx_values = deque(maxlen=buffer_size)
        self.qy_values = deque(maxlen=buffer_size)
        self.qz_values = deque(maxlen=buffer_size)
        self.qw_values = deque(maxlen=buffer_size)
        
        # Start time reference
        self.start_time = time.time()
        
        # Create figure and subplots
        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 8))
        plt.ion()  # Interactive mode on
        
    def add_pose(self, pose):
        """
        Add a new pose measurement to the plot
        
        Args:
            pose: A PoseStamped message or dictionary with position and orientation
        """
        current_time = time.time() - self.start_time
        
        # Extract position and orientation from pose
        if hasattr(pose, 'pose'):  # If it's a PoseStamped message
            self.x_values.append(pose.pose.position.x)
            self.y_values.append(pose.pose.position.y)
            self.z_values.append(pose.pose.position.z)
            self.qx_values.append(pose.pose.orientation.x)
            self.qy_values.append(pose.pose.orientation.y)
            self.qz_values.append(pose.pose.orientation.z)
            self.qw_values.append(pose.pose.orientation.w)
        else:  # If it's a dictionary
            self.x_values.append(pose['x'])
            self.y_values.append(pose['y'])
            self.z_values.append(pose['z'])
            self.qx_values.append(pose['qx'])
            self.qy_values.append(pose['qy'])
            self.qz_values.append(pose['qz'])
            self.qw_values.append(pose['qw'])
        
        self.timestamps.append(current_time)
        
    def update_plot(self):
        """Update the plot with current data"""
        # Clear previous plots
        self.axs[0].clear()
        self.axs[1].clear()
        
        # Plot position data
        self.axs[0].plot(list(self.timestamps), list(self.x_values), 'r-', label='X')
        self.axs[0].plot(list(self.timestamps), list(self.y_values), 'g-', label='Y')
        self.axs[0].plot(list(self.timestamps), list(self.z_values), 'b-', label='Z')
        self.axs[0].set_ylabel('Position (m)')
        self.axs[0].set_title('Robot Pose Consistency - Position')
        self.axs[0].legend()
        self.axs[0].grid(True)
        
        # Plot orientation data
        self.axs[1].plot(list(self.timestamps), list(self.qx_values), 'r-', label='Qx')
        self.axs[1].plot(list(self.timestamps), list(self.qy_values), 'g-', label='Qy')
        self.axs[1].plot(list(self.timestamps), list(self.qz_values), 'b-', label='Qz')
        self.axs[1].plot(list(self.timestamps), list(self.qw_values), 'k-', label='Qw')
        self.axs[1].set_ylabel('Orientation (quaternion)')
        self.axs[1].set_xlabel('Time (s)')
        self.axs[1].set_title('Robot Pose Consistency - Orientation')
        self.axs[1].legend()
        self.axs[1].grid(True)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def save_plot(self, filename='pose_consistency.png'):
        """Save the current plot to a file"""
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        
    def calculate_pose_statistics(self):
        """Calculate statistical measures of pose consistency"""
        if len(self.x_values) < 2:
            return None
            
        # Calculate standard deviations
        x_std = np.std(list(self.x_values))
        y_std = np.std(list(self.y_values))
        z_std = np.std(list(self.z_values))
        
        # Calculate mean absolute differences between consecutive measurements
        x_diffs = np.abs(np.diff(list(self.x_values)))
        y_diffs = np.abs(np.diff(list(self.y_values)))
        z_diffs = np.abs(np.diff(list(self.z_values)))
        
        x_mean_diff = np.mean(x_diffs)
        y_mean_diff = np.mean(y_diffs)
        z_mean_diff = np.mean(z_diffs)
        
        # Calculate quaternion stability (using dot product)
        quat_stability = []
        quats = np.array([list(self.qx_values), list(self.qy_values), 
                          list(self.qz_values), list(self.qw_values)]).T
        
        for i in range(1, len(quats)):
            dot_product = np.abs(np.sum(quats[i] * quats[i-1]))
            # Dot product of unit quaternions is close to 1 if they are similar
            quat_stability.append(dot_product)
        
        quat_stability_mean = np.mean(quat_stability) if quat_stability else 0
        
        stats = {
            'position_std': {'x': x_std, 'y': y_std, 'z': z_std},
            'position_stability': {'x': x_mean_diff, 'y': y_mean_diff, 'z': z_mean_diff},
            'orientation_stability': quat_stability_mean
        }
        
        return stats
