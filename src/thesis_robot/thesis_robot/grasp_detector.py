import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, Pose
import numpy as np
import open3d as o3d
import struct

class GraspDetector(Node):
    def __init__(self):
        super().__init__('grasp_detector')
        self.create_subscription(
            PointCloud2, '/bottle_pointcloud', self.callback, 10)
        self.grasp_pub = self.create_publisher(PoseArray, '/grasp_poses', 10)
        self.get_logger().info('Grasp detector started')

    def pointcloud2_to_numpy(self, msg):
        points = []
        for i in range(msg.width):
            offset = i * msg.point_step
            x, y, z = struct.unpack_from('fff', msg.data, offset)
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                points.append([x, y, z])
        return np.array(points)

    def callback(self, msg):
        pts = self.pointcloud2_to_numpy(msg)
        if len(pts) < 100:
            self.get_logger().warn('Too few points for grasp detection')
            return

        # Build Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # Downsample
        pcd = pcd.voxel_down_sample(voxel_size=0.003)

        # Remove table plane (RANSAC)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.005, ransac_n=3, num_iterations=1000)
        bottle_pcd = pcd.select_by_index(inliers, invert=True)

        if len(bottle_pcd.points) < 50:
            self.get_logger().warn('Too few points after plane removal')
            return

        # Estimate normals
        bottle_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.01, max_nn=30))

        pts_arr = np.asarray(bottle_pcd.points)
        norms = np.asarray(bottle_pcd.normals)

        # Bottle center and dimensions
        center = pts_arr.mean(axis=0)
        z_min = pts_arr[:, 2].min()
        z_max = pts_arr[:, 2].max()
        grasp_height = z_min + (z_max - z_min) * 0.4  # 40% up from bottom

        # Generate grasp candidates around bottle
        grasp_poses = PoseArray()
        grasp_poses.header = msg.header

        num_grasps = 8
        for i in range(num_grasps):
            angle = 2 * np.pi * i / num_grasps
            approach_x = np.cos(angle)
            approach_y = np.sin(angle)

            # Grasp position: approach from side at grasp height
            grasp_x = center[0] + approach_x * 0.12
            grasp_y = center[1] + approach_y * 0.12
            grasp_z = grasp_height

            # Orientation: gripper points toward bottle center
            # z-axis = approach direction (toward bottle)
            z_ax = np.array([-approach_x, -approach_y, 0.0])
            x_ax = np.array([0.0, 0.0, 1.0])
            y_ax = np.cross(z_ax, x_ax)

            # Build rotation matrix
            R = np.column_stack([x_ax, y_ax, z_ax])

            # Convert to quaternion
            q = self.rotation_matrix_to_quaternion(R)

            p = Pose()
            p.position.x = float(grasp_x)
            p.position.y = float(grasp_y)
            p.position.z = float(grasp_z)
            p.orientation.x = float(q[0])
            p.orientation.y = float(q[1])
            p.orientation.z = float(q[2])
            p.orientation.w = float(q[3])
            grasp_poses.poses.append(p)

        self.grasp_pub.publish(grasp_poses)
        self.get_logger().info(
            f'Published {len(grasp_poses.poses)} grasp candidates | '
            f'Bottle center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) | '
            f'Grasp height: {grasp_height:.3f}m')

    def rotation_matrix_to_quaternion(self, R):
        trace = R[0,0] + R[1,1] + R[2,2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2,1] - R[1,2]) * s
            y = (R[0,2] - R[2,0]) * s
            z = (R[1,0] - R[0,1]) * s
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
        return [x, y, z, w]

def main():
    rclpy.init()
    rclpy.spin(GraspDetector())

if __name__ == '__main__':
    main()
