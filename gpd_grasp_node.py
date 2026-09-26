#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, Pose
import sensor_msgs_py.point_cloud2 as pc2
import subprocess, tempfile, os, struct

GPD_DETECT = '/home/lab/workspace/ros2_kortex_ws/src/gpd/build/detect_grasps'
GPD_CFG    = '/home/lab/workspace/ros2_kortex_ws/src/gpd/cfg/ros_eigen_params.cfg'

class GPDGraspNode(Node):
    def __init__(self):
        super().__init__('gpd_grasp_node')
        self.pub = self.create_publisher(PoseArray, '/grasp_poses', 10)
        self.create_subscription(PointCloud2, '/detection_cloud', self._cb, 10)
        self.get_logger().info('GPD Grasp Node started')

    def _cb(self, msg):
        pts = list(pc2.read_points(msg, field_names=['x','y','z'], skip_nans=True))
        if len(pts) < 50:
            self.get_logger().warn(f'Too few points: {len(pts)}')
            return
        self.get_logger().info(f'Running GPD on {len(pts)} points...')
        tmp = tempfile.NamedTemporaryFile(suffix='.pcd', delete=False)
        header = f'# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts)}\nDATA binary\n'
        tmp.write(header.encode())
        for p in pts:
            tmp.write(struct.pack('fff', float(p[0]), float(p[1]), float(p[2])))
        tmp.close()
        try:
            result = subprocess.run([GPD_DETECT, GPD_CFG, tmp.name],
                capture_output=True, text=True, timeout=30)
            self.get_logger().info(result.stdout[-500:])
            pa = PoseArray()
            pa.header.frame_id = msg.header.frame_id
            pa.header.stamp = self.get_clock().now().to_msg()
            for line in result.stdout.split('\n'):
                if 'Position' in line:
                    parts = line.split()
                    try:
                        p = Pose()
                        p.position.x = float(parts[-3])
                        p.position.y = float(parts[-2])
                        p.position.z = float(parts[-1])
                        p.orientation.w = 1.0
                        pa.poses.append(p)
                    except Exception:
                        pass
            if pa.poses:
                self.get_logger().info(f'Publishing {len(pa.poses)} grasps')
                self.pub.publish(pa)
        except Exception as e:
            self.get_logger().error(f'GPD failed: {e}')
        finally:
            os.unlink(tmp.name)

def main():
    rclpy.init()
    rclpy.spin(GPDGraspNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
