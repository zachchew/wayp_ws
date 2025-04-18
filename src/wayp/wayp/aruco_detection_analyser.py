import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
import time
import cv2
import numpy as np

from wayp.utils.detection_rate_plotter import ArUcoDetectionRatePlotter

class ArUcoDetectionAnalyser(Node):
    def __init__(self):
        super().__init__('aruco_detection_analyser')
        
        # Parameters
        self.declare_parameter('marker_ids', [1, 2, 3, 4])
        self.declare_parameter('buffer_size', 200)
        self.declare_parameter('update_interval', 1.0)
        
        marker_ids = self.get_parameter('marker_ids').value
        buffer_size = self.get_parameter('buffer_size').value
        update_interval = self.get_parameter('update_interval').value
        
        # Create subscription to visible marker IDs
        self.visible_markers_sub = self.create_subscription(
            Int32MultiArray,
            '/visible_marker_ids',
            self.visible_markers_callback,
            10)
        
        # Initialize plotter
        self.plotter = ArUcoDetectionRatePlotter(
            marker_ids=marker_ids,
            buffer_size=buffer_size,
            update_interval=update_interval
        )
        
        # Create timer for logging statistics
        self.stats_timer = self.create_timer(10.0, self.log_statistics)
        
        # Create timer for displaying the plot in the main thread
        self.display_timer = self.create_timer(0.5, self.display_plot)
        
        self.get_logger().info('ArUco detection analyser started')
        
    def visible_markers_callback(self, msg):
        """Process visible marker IDs"""
        visible_ids = msg.data
        self.plotter.add_detection(visible_ids)
        
    def log_statistics(self):
        """Log detection statistics periodically"""
        stats = self.plotter.calculate_statistics()
        if stats:
            self.get_logger().info(f"Overall detection rate: {stats['overall_detection_rate']:.2f}")
            
            marker_rates = stats['marker_detection_rates']
            for marker_id, rate in marker_rates.items():
                self.get_logger().info(f"Marker {marker_id} detection rate: {rate:.2f}")
                
            self.get_logger().info(f"Average markers detected: {stats['avg_markers_detected']:.2f}")
    
    def display_plot(self):
        """Display the plot in the main thread"""
        img = self.plotter.get_current_plot_image()
        if img is not None:
            # Convert PIL image to OpenCV format
            cv_img = np.array(img)
            cv_img = cv_img[:, :, ::-1]  # RGB to BGR
            
            # Display using OpenCV
            cv2.imshow("ArUco Detection Rate", cv_img)
            cv2.waitKey(1)
        
    def on_shutdown(self):
        """Save the final plot and statistics on shutdown"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.plotter.save_plot(f'aruco_detection_rate_{timestamp}.png')
        
        # Save statistics to file
        stats = self.plotter.calculate_statistics()
        if stats:
            with open(f'aruco_detection_stats_{timestamp}.txt', 'w') as f:
                f.write(f"Overall detection rate: {stats['overall_detection_rate']:.4f}\n")
                f.write(f"Average markers detected: {stats['avg_markers_detected']:.4f}\n\n")
                
                f.write("Per-marker detection rates:\n")
                for marker_id, rate in stats['marker_detection_rates'].items():
                    f.write(f"Marker {marker_id}: {rate:.4f}\n")
                
                f.write("\nTime window detection rates:\n")
                for window_time, window_stats in stats['window_detection_rates'].items():
                    f.write(f"Window {window_time:.1f}s - {window_time + self.plotter.window_size:.1f}s:\n")
                    for marker_id, rate in window_stats.items():
                        if marker_id != 'total':
                            f.write(f"  Marker {marker_id}: {rate:.4f}\n")
                    f.write("\n")
        
        self.plotter.stop()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = ArUcoDetectionAnalyser()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
