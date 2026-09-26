import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

CORRECT_ORDER = [
    "finger_joint", "joint_1", "joint_2", "joint_3",
    "joint_4", "joint_5", "joint_6", "joint_7"
]

class JointStateRemapper(Node):
    def __init__(self):
        super().__init__("joint_state_remapper")
        self.sub = self.create_subscription(
            JointState, "/joint_states", self.callback, 10)
        self.pub = self.create_publisher(
            JointState, "/joint_states_fixed", 10)
        self.get_logger().info("Joint state remapper started")

    def callback(self, msg):
        joint_map = {}
        for i, name in enumerate(msg.name):
            joint_map[name] = {
                "pos": msg.position[i] if i < len(msg.position) else 0.0,
                "vel": msg.velocity[i] if i < len(msg.velocity) else 0.0,
                "eff": msg.effort[i]   if i < len(msg.effort)   else 0.0,
            }
        new_msg = JointState()
        new_msg.header = msg.header
        for name in CORRECT_ORDER:
            if name in joint_map:
                new_msg.name.append(name)
                new_msg.position.append(joint_map[name]["pos"])
                new_msg.velocity.append(joint_map[name]["vel"])
                new_msg.effort.append(joint_map[name]["eff"])
        self.pub.publish(new_msg)

def main():
    rclpy.init()
    rclpy.spin(JointStateRemapper())

if __name__ == "__main__":
    main()
