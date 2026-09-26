#!/usr/bin/env python3
"""
Calibration pose-diversity checker for Kinova Gen3 7DOF + RealSense D435I.
Eye-to-base: camera fixed, ChArUco board on EE.

Fully automatic — no keyboard input required.

Workflow:
  1. Run this script alongside the MoveIt2 calibration RViz panel.
  2. Move the arm to a new pose and hold still.
  3. After the arm is still for 1.5 s, this script auto-checks diversity.
       - GOOD → prints ">>> TAKE SAMPLE NOW <<<" — click Take Sample in RViz.
       - TOO SIMILAR → tells you what kind of motion is missing.
  4. Once you click Take Sample in RViz, move the arm to the next pose.
     The script detects movement and resets automatically.

Run:
  source ~/workspace/ros2_kortex_ws/install/setup.bash
  python3 ~/workspace/ros2_kortex_ws/calibration_helper.py
"""

import math

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

BASE_FRAME = "base_link"
EE_FRAME   = "end_effector_link"

STILL_DURATION_S   = 1.5   # seconds arm must be stationary before auto-check
STILL_TRANS_TOL_M  = 0.003 # metres — arm considered still below this
STILL_ROT_TOL_DEG  = 0.5   # degrees — arm considered still below this

MIN_TRANS_M        = 0.08  # minimum translation from closest sample to be GOOD
MIN_ROT_DEG        = 12.0  # minimum rotation  from closest sample to be GOOD

# After recording, arm must move this much before the next auto-record triggers.
# Prevents recording the same pose twice if the user is slow to move away.
RESET_TRANS_M      = 0.04
RESET_ROT_DEG      = 6.0


class CalibrationHelper(Node):
    def __init__(self):
        super().__init__("calibration_helper")
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.samples: list[list[float]] = []

        self._prev_pose       = None   # pose at previous tick
        self._still_since     = None   # time() when arm last became still
        self._last_recorded   = None   # pose that was last auto-recorded
        self._waiting_for_move = False # True after recording, waiting for arm to leave

        self.timer = self.create_timer(0.2, self._tick)  # 5 Hz

        self.get_logger().info(
            f"\nCalibration helper ready  ({BASE_FRAME} → {EE_FRAME})\n"
            "  Move the arm to a new pose and hold still.\n"
            "  When the pose is good, you will see  >>> TAKE SAMPLE NOW <<<\n"
            "  Then click Take Sample in the MoveIt2 RViz panel.\n"
        )

    # ------------------------------------------------------------------ #
    # TF helpers                                                           #
    # ------------------------------------------------------------------ #

    def _lookup(self) -> list[float] | None:
        try:
            t = self.tf_buffer.lookup_transform(BASE_FRAME, EE_FRAME, rclpy.time.Time())
            tr = t.transform.translation
            ro = t.transform.rotation
            return [tr.x, tr.y, tr.z, ro.x, ro.y, ro.z, ro.w]
        except Exception:
            return None

    @staticmethod
    def _trans_dist(a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    @staticmethod
    def _rot_dist_deg(a, b):
        dot = abs(a[3]*b[3] + a[4]*b[4] + a[5]*b[5] + a[6]*b[6])
        return math.degrees(2.0 * math.acos(min(1.0, dot)))

    # ------------------------------------------------------------------ #
    # Diversity                                                            #
    # ------------------------------------------------------------------ #

    def _closest(self, pose):
        """Return (min_dist_m, min_angle_deg) to the nearest existing sample."""
        if not self.samples:
            return float("inf"), float("inf")
        dists  = [self._trans_dist(pose, s)    for s in self.samples]
        angles = [self._rot_dist_deg(pose, s)  for s in self.samples]
        return min(dists), min(angles)

    def _diversity_report(self):
        n = len(self.samples)
        if n < 2:
            return ""
        total_d = total_a = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_d += self._trans_dist(self.samples[i], self.samples[j])
                total_a += self._rot_dist_deg(self.samples[i], self.samples[j])
                count += 1
        avg_d = total_d / count
        avg_a = total_a / count
        ok_d = "OK" if avg_d > 0.12 else "LOW — need more arm movement"
        ok_a = "OK" if avg_a > 20   else "LOW — need more wrist rotation"
        return (
            f"  Set diversity:  avg trans={avg_d:.3f} m ({ok_d})\n"
            f"                  avg rot  ={avg_a:.1f} deg ({ok_a})"
        )

    # ------------------------------------------------------------------ #
    # Main tick                                                            #
    # ------------------------------------------------------------------ #

    def _tick(self):
        pose = self._lookup()
        if pose is None:
            return

        tx, ty, tz = pose[:3]

        # --- detect movement ---
        if self._prev_pose is not None:
            moved_t = self._trans_dist(pose, self._prev_pose)
            moved_r = self._rot_dist_deg(pose, self._prev_pose)
            moving  = moved_t > STILL_TRANS_TOL_M or moved_r > STILL_ROT_TOL_DEG
        else:
            moving = True

        self._prev_pose = pose

        # --- if we just recorded, wait until the arm actually moves away ---
        if self._waiting_for_move:
            if self._last_recorded is not None:
                away_t = self._trans_dist(pose, self._last_recorded)
                away_r = self._rot_dist_deg(pose, self._last_recorded)
                if away_t > RESET_TRANS_M or away_r > RESET_ROT_DEG:
                    self._waiting_for_move = False
                    self._still_since = None
                    self.get_logger().info("Arm moved — watching for next good pose...")
            return  # don't evaluate diversity until arm has left the recorded pose

        # --- stillness timer ---
        now = self.get_clock().now().nanoseconds * 1e-9
        if moving:
            self._still_since = None
            return

        if self._still_since is None:
            self._still_since = now
            return

        if now - self._still_since < STILL_DURATION_S:
            return  # not still long enough yet

        # --- arm has been still long enough: evaluate diversity ---
        if not self.samples:
            self.get_logger().info(
                f"\n>>> TAKE SAMPLE NOW (sample 1) <<<\n"
                f"  EE: x={tx:.3f}  y={ty:.3f}  z={tz:.3f}\n"
                "  (first sample — any pose is fine)"
            )
            self._record(pose)
            return

        min_dist, min_angle = self._closest(pose)

        if min_dist > MIN_TRANS_M or min_angle > MIN_ROT_DEG:
            report = self._diversity_report()
            self.get_logger().info(
                f"\n{'='*50}\n"
                f">>> TAKE SAMPLE NOW (sample {len(self.samples)+1}) <<<\n"
                f"  EE: x={tx:.3f}  y={ty:.3f}  z={tz:.3f}\n"
                f"  Dist from closest: {min_dist:.3f} m   Angle: {min_angle:.1f} deg\n"
                f"{'='*50}"
            )
            self._record(pose)
        else:
            hint = self._movement_hint(min_dist, min_angle)
            self.get_logger().info(
                f"TOO SIMILAR — move more\n"
                f"  EE: x={tx:.3f}  y={ty:.3f}  z={tz:.3f}\n"
                f"  Closest sample: dist={min_dist:.3f} m   angle={min_angle:.1f} deg\n"
                f"  {hint}"
            )
            # reset still timer so it doesn't spam this message
            self._still_since = None

    def _record(self, pose):
        self.samples.append(pose)
        self._last_recorded   = pose
        self._waiting_for_move = True
        self._still_since      = None
        report = self._diversity_report()
        if report:
            self.get_logger().info(f"Auto-recorded sample {len(self.samples)}.\n{report}")
        else:
            self.get_logger().info(f"Auto-recorded sample {len(self.samples)}.")

    @staticmethod
    def _movement_hint(dist, angle):
        if dist <= MIN_TRANS_M and angle <= MIN_ROT_DEG:
            if dist < 0.04:
                return "Hint: move the arm at least 10 cm in any direction"
            else:
                return "Hint: rotate the wrist at least 15 degrees (tilt/roll the board)"
        if angle <= MIN_ROT_DEG:
            return "Hint: rotate the wrist — tilt or roll the board more"
        return "Hint: move the arm further from its current position"


# ------------------------------------------------------------------ #

def main():
    rclpy.init()
    node = CalibrationHelper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
