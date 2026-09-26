import sys, threading
sys.path.insert(0, '/home/lab/workspace/YOLO-3D/GreenArm/src/install/kinova_gen3/lib/python3.10/site-packages')
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2
from kinova_gen3.utilities import parseConnectionArguments, DeviceConnection

TIMEOUT = 60   # increased timeout to allow for slow movement

# ── Speed constraint (lower = slower) ────────────────────────────
# translation: metres/sec   rotation: degrees/sec
TRANSLATION_SPEED = 0.01   # 1 cm/s  (default is ~0.1)
ROTATION_SPEED    = 5.0    # 5 deg/s (default is ~15)

POSITIONS = [
    {"name": "Glass Position 1", "x": 0.267, "y": -0.076, "z": 0.240, "tx": -179.072, "ty": 5.621, "tz": 79.154},
    {"name": "Glass Position 2", "x": 0.292, "y": -0.102, "z": 0.217, "tx":  176.664, "ty": 5.199, "tz": 78.054},
    {"name": "Glass Position 3", "x": 0.269, "y": -0.094, "z": 0.221, "tx":  178.548, "ty": 4.646, "tz": 78.019},
    {"name": "Glass Position 4", "x": 0.249, "y": -0.098, "z": 0.217, "tx": -178.695, "ty": 5.499, "tz": 77.817},
]

def check_for_end_or_abort(e):
    def check(notification, e=e):
        if notification.action_event in [
                Base_pb2.ACTION_END,
                Base_pb2.ACTION_ABORT]:
            e.set()
    return check

def move_to_pose(base, pos):
    print(f"\n Moving to {pos['name']}...")
    print(f"  x={pos['x']}  y={pos['y']}  z={pos['z']}")
    print(f"  Tx={pos['tx']}  Ty={pos['ty']}  Tz={pos['tz']}")
    print(f"  Speed: {TRANSLATION_SPEED*100:.1f} cm/s  |  {ROTATION_SPEED} deg/s")

    # ── Use ConstrainedPose to enforce speed limits ───────────────
    action = Base_pb2.Action()
    action.name             = pos['name']
    action.application_data = ""

    constrained_pose = action.reach_pose

    # Target pose
    p           = constrained_pose.target_pose
    p.x         = pos['x']
    p.y         = pos['y']
    p.z         = pos['z']
    p.theta_x   = pos['tx']
    p.theta_y   = pos['ty']
    p.theta_z   = pos['tz']

    # Speed constraint
    speed                   = constrained_pose.constraint.speed
    speed.translation       = TRANSLATION_SPEED
    speed.orientation       = ROTATION_SPEED

    e      = threading.Event()
    handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions())

    base.ExecuteAction(action)
    finished = e.wait(TIMEOUT)
    base.Unsubscribe(handle)

    if finished:
        print(f"  Reached {pos['name']}!")
    else:
        print(f"  Timeout on {pos['name']}! (increase TIMEOUT if needed)")
    return finished

def main():
    args = parseConnectionArguments()
    with DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        mode = Base_pb2.ServoingModeInformation()
        mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(mode)

        print("="*50)
        print("  ROBOT ARM — GLASS POSITIONS (SLOW MODE)")
        print("="*50)
        print(f"  Total positions : {len(POSITIONS)}")
        print(f"  Translation speed: {TRANSLATION_SPEED*100:.1f} cm/s")
        print(f"  Rotation speed   : {ROTATION_SPEED} deg/s")
        print("  Press ENTER to go to next position")
        print("  Press Ctrl+C to quit")
        print("="*50)

        for i, pos in enumerate(POSITIONS):
            print(f"\n>>> Press ENTER to go to {pos['name']} "
                  f"({i+1}/{len(POSITIONS)}): ", end="", flush=True)
            try:
                input()
            except KeyboardInterrupt:
                print("\nAborted.")
                return
            move_to_pose(base, pos)

        print("\n All Glass Positions completed!")

if __name__ == "__main__":
    main()
