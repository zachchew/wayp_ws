# import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from collections import deque, defaultdict
import time
import threading
import io
from PIL import Image

class ArUcoDetectionRatePlotter:
    def __init__(self, marker_ids=[1, 2, 3, 4], buffer_size=100, update_interval=1.0):
        # Initialize data buffers
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.marker_ids = marker_ids
        self.timestamps = deque(maxlen=buffer_size)
        
        # Track detection status for each marker
        self.marker_detections = {}
        for marker_id in marker_ids:
            self.marker_detections[marker_id] = deque(maxlen=buffer_size)
        
        # Track overall detection rate
        self.any_marker_detected = deque(maxlen=buffer_size)
        
        # Track multiple marker detection rates
        self.num_markers_detected = deque(maxlen=buffer_size)
        
        # Track detection counts in time windows
        self.detection_windows = defaultdict(int)
        self.window_size = 10  # seconds
        self.window_counts = defaultdict(lambda: defaultdict(int))
        
        # Start time reference
        self.start_time = time.time()
        
        # Create figure and subplots (do this once)
        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # Lock for thread safety
        self.data_lock = threading.Lock()
        
        # Flag to indicate if plot needs updating
        self.plot_dirty = True
        
        # Current plot image
        self.current_plot_image = None
        
        # Start the update thread
        self.running = True
        self.update_thread = threading.Thread(target=self._update_plot_thread)
        self.update_thread.daemon = True
        self.update_thread.start()
        
    def add_detection(self, visible_ids):
        """Add a new detection measurement to the plot"""
        with self.data_lock:
            current_time = time.time() - self.start_time
            self.timestamps.append(current_time)
            
            # Record which markers were detected
            any_detected = False
            for marker_id in self.marker_ids:
                detected = marker_id in visible_ids
                self.marker_detections[marker_id].append(1 if detected else 0)
                any_detected = any_detected or detected
                
                # Update window counts
                window_key = int(current_time / self.window_size)
                self.window_counts[window_key][marker_id] += 1 if detected else 0
                self.window_counts[window_key]['total'] += 1
            
            # Record if any marker was detected
            self.any_marker_detected.append(1 if any_detected else 0)
            
            # Record how many markers were detected
            self.num_markers_detected.append(len(visible_ids))
            
            # Mark plot as needing update
            self.plot_dirty = True
    
    def _update_plot_thread(self):
        """Thread function to update the plot periodically"""
        while self.running:
            try:
                if self.plot_dirty:
                    self._render_plot_to_image()
                    self.plot_dirty = False
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error updating plot: {e}")
    
    def _render_plot_to_image(self):
        """Render the plot to an image buffer (thread-safe)"""
        if len(self.timestamps) < 2:
            return
            
        # Create a new figure for this update to avoid GUI operations
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # Set axes spine colors to black and line width thinner
        for ax in axs:
            for spine in ax.spines.values():
                spine.set_color('black')
                spine.set_linewidth(0.7)  # slightly thinner
            ax.grid(True, linewidth=0.7, color='black', alpha=0.3)  # thinner grid lines
        
        with self.data_lock:
            # Define specific colors for each marker
            marker_colors = {
                1: 'red',
                2: 'blue',
                3: 'green',
                4: 'purple'
            }
            
            # Plot individual marker detection rates
            for marker_id in self.marker_ids:
                if len(self.marker_detections[marker_id]) > 0:
                    # Calculate rolling average (5-point window)
                    window_size = min(5, len(self.marker_detections[marker_id]))
                    rolling_avg = np.convolve(
                        list(self.marker_detections[marker_id]), 
                        np.ones(window_size)/window_size, 
                        mode='valid'
                    )
                    
                    # Plot with assigned color based on marker ID
                    x_values = list(self.timestamps)[-len(rolling_avg):]
                    color = marker_colors.get(marker_id, 'black')  # Default to black if marker ID not in dict
                    axs[0].plot(x_values, rolling_avg, color=color, label=f'Marker {marker_id}')
            
            # Plot overall detection rate
            if len(self.any_marker_detected) > 0:
                window_size = min(5, len(self.any_marker_detected))
                overall_avg = np.convolve(
                    list(self.any_marker_detected), 
                    np.ones(window_size)/window_size, 
                    mode='valid'
                )
                x_values = list(self.timestamps)[-len(overall_avg):]
                axs[0].plot(x_values, overall_avg, 'k-', 
                        linewidth=2, label='Any marker')
            
            axs[0].set_ylabel('Detection Rate (rolling avg)')
            axs[0].set_title('ArUco Marker Detection Success Rate')
            axs[0].legend()
            axs[0].set_ylim(-0.05, 1.05)
            
            # Plot number of markers detected in brown
            if len(self.num_markers_detected) > 0:
                axs[1].plot(list(self.timestamps), 
                        list(self.num_markers_detected), color='brown')
                axs[1].set_ylabel('Number of Markers Detected')
                axs[1].set_xlabel('Time (s)')
                axs[1].set_title('Multiple Marker Detection')
                axs[1].set_ylim(-0.5, len(self.marker_ids) + 0.5)
                
                # Add horizontal lines for reference
                for i in range(1, len(self.marker_ids) + 1):
                    axs[1].axhline(y=i, color='gray', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # Save to buffer instead of displaying
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # Store the image data
        self.current_plot_image = Image.open(buf)
        
        # Close the figure to free memory
        plt.close(fig)

    
    def get_current_plot_image(self):
        """Get the current plot as a PIL Image (thread-safe)"""
        return self.current_plot_image
    
    def save_plot(self, filename='aruco_detection_rate.png'):
        """Save the current plot to a file"""
        if self.current_plot_image:
            self.current_plot_image.save(filename)
            print(f"Plot saved to {filename}")
        else:
            print("No plot available to save")
    
    def calculate_statistics(self):
        """Calculate statistical measures of detection success (thread-safe)"""
        with self.data_lock:
            if len(self.timestamps) < 2:
                return None
                
            stats = {
                'overall_detection_rate': np.mean(list(self.any_marker_detected)) if self.any_marker_detected else 0,
                'marker_detection_rates': {},
                'avg_markers_detected': np.mean(list(self.num_markers_detected)) if self.num_markers_detected else 0,
                'window_detection_rates': {}
            }
            
            # Calculate per-marker detection rates
            for marker_id in self.marker_ids:
                if len(self.marker_detections[marker_id]) > 0:
                    stats['marker_detection_rates'][marker_id] = np.mean(
                        list(self.marker_detections[marker_id]))
                else:
                    stats['marker_detection_rates'][marker_id] = 0
                    
            # Calculate window-based detection rates
            for window_key in sorted(self.window_counts.keys()):
                window_stats = {}
                total_frames = self.window_counts[window_key]['total']
                if total_frames > 0:
                    for marker_id in self.marker_ids:
                        detections = self.window_counts[window_key][marker_id]
                        window_stats[marker_id] = detections / total_frames
                stats['window_detection_rates'][window_key * self.window_size] = window_stats
                
            return stats
    
    def stop(self):
        """Stop the update thread"""
        self.running = False
        if self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
