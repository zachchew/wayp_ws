import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

from wayp.utils.pose_plotter import PoseConsistencyPlotter

class PoseConsistencyAnalyser(Node):
    def __init__(self):
        super().__init__('pose_consistency_analyser')
        
        # Create pose subscriber
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/robot_pose',
            self.pose_callback,
            10)
        
        # Initialize plotter
        self.plotter = PoseConsistencyPlotter(buffer_size=200)
        
        # Create timer for updating plot
        self.timer = self.create_timer(0.5, self.update_plot)
        
        self.get_logger().info('Pose consistency analyser started')
        
    def pose_callback(self, msg):
        # Add new pose to plotter
        self.plotter.add_pose(msg)
        
    def update_plot(self):
        # Update the plot
        self.plotter.update_plot()
        
        # Calculate and log statistics
        stats = self.plotter.calculate_pose_statistics()
        if stats:
            self.get_logger().info(f"Position std dev: X={stats['position_std']['x']:.4f}, Y={stats['position_std']['y']:.4f}, Z={stats['position_std']['z']:.4f}")
            self.get_logger().info(f"Orientation stability: {stats['orientation_stability']:.4f}")
        
    def on_shutdown(self):
        # Save the final plot
        self.plotter.save_plot('final_pose_consistency.png')

def main(args=None):
    rclpy.init(args=args)
    node = PoseConsistencyAnalyser()
    
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
