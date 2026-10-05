#!/usr/bin/env python3
"""
LLM Planner Node — the brain of the Audio-to-Action pipeline.

Subscribes to:
  /scene_snapshot   (std_msgs/String — JSON world model)
  /voice_command    (std_msgs/String — transcribed text)

Publishes to:
  /action_plan      (std_msgs/String — validated JSON plan)
  /planner_status   (std_msgs/String — human-readable status)

Your thesis contribution: LLM-based task decomposition grounded
in a live scene graph, with pre-execution safety validation.
"""
import rclpy, json, math, time, threading, ollama
from rclpy.node import Node
from std_msgs.msg import String

from thesis_robot import safety_geometry as sg

# ── SYSTEM PROMPT ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a task planner for a Kinova Gen3 7-DOF robotic arm.

SAFETY RULES — every action must obey ALL of these without exception:
  RULE 1: ALL z values >= {Z_FLOOR} at minimum (z is the wrist flange height;
          the fingertips hang ~0.21 m below it, so this keeps them above the table)
  RULE 2: approach_z in pick >= {Z_FLOOR} (approach from well above object)
  RULE 3: place z >= {Z_FLOOR} (place gently, never at table level)
  RULE 4: Speed always exactly 0.20 (20% of maximum — safety requirement)
  RULE 5: x must be in [0.10, 0.60]  y must be in [-0.35, 0.35]
  RULE 6: Never command picking a stale object (stale=true in world state)

AVAILABLE ACTIONS — use only these, no others:
  {"action":"move_to",      "x":float, "y":float, "z":float, "speed":0.2}
  {"action":"pick",         "object_id":"str", "approach_z":float}
  {"action":"place",        "x":float, "y":float, "z":float}
  {"action":"open_gripper"}
  {"action":"close_gripper"}
  {"action":"go_home"}
  {"action":"null_space_adjust", "objective":"clear_camera"}

WHICH ACTION FOR WHICH COMMAND:
  - "go to/near/over X", "move to X", "look at X" (no picking up) ->
    ONE move_to step. Use X's x/y from the world state, and
    z = max(X's z + 0.15, {Z_FLOOR}) (hover above it, don't descend onto it).
    Do NOT use pick for these — pick closes the gripper on the object,
    which is not what "go near" means.
  - "pick up X", "grab X", "get X" -> a pick step (approach_z>=0.15),
    optionally followed by place/open_gripper if the command also says
    where to put it down.
  - If the named object isn't in the world state below, or every
    instance of it is stale, say so in "reasoning" and return an empty
    plan rather than guessing a position.

WORKSPACE BOUNDS: x=[0.10, 0.60]  y=[-0.35, 0.35]  z=[{Z_FLOOR}, 0.50]

RESPOND WITH EXACTLY THIS JSON STRUCTURE — no other format:
{
  "reasoning": "step by step thinking about which objects, sequence, safety checks",
  "plan": [
    {"action": "...", ...},
    {"action": "...", ...}
  ]
}"""

# ── MOVE_TO HEIGHT CORRECTION ────────────────────────────────────────
# The system prompt tells the LLM to set move_to's z to "object z + 0.15"
# (hover above, don't descend onto it), but a 7B local model doesn't
# follow that arithmetic reliably -- it sometimes copies the object's raw
# z straight through, which is at/below table height and gets rejected
# by the floor check below. Since the LLM's x/y *are* reliably copied
# from the target object, re-derive z from the matched object ourselves
# instead of trusting the LLM's addition, rather than just rejecting and
# hoping a retry does better.
MOVE_TO_HOVER_M = 0.15
_XY_MATCH_TOL_M = 0.05
# z_floor comes from safety_geometry.flange_floor_z(): table height + the
# 0.215 m flange-to-fingertip reach + clearance. (An earlier fixed 0.12
# minimum here still put the fingertips at the table surface.)


def fix_move_to_heights(plan, scene, z_floor):
    """Mutates move_to steps in place: if a step's (x,y) matches a known
    scene object closely enough, override z to that object's z + hover
    offset (never below z_floor), regardless of what the LLM computed."""
    if not isinstance(plan, list):
        return plan
    for step in plan:
        if not isinstance(step, dict) or step.get('action') != 'move_to':
            continue
        if not (_is_number(step.get('x')) and _is_number(step.get('y'))):
            continue
        best_id, best_dist = None, _XY_MATCH_TOL_M
        for obj_id, obj in scene.items():
            if not isinstance(obj, dict) or obj.get('z') is None:
                continue
            dist = ((obj['x'] - step['x']) ** 2 + (obj['y'] - step['y']) ** 2) ** 0.5
            if dist < best_dist:
                best_id, best_dist = obj_id, dist
        if best_id is not None:
            step['z'] = max(scene[best_id]['z'] + MOVE_TO_HOVER_M, z_floor)
    return plan


# ── SAFETY VALIDATOR ─────────────────────────────────────────────────
def validate_plan(plan, scene, z_floor=0.08):
    """
    Scan entire plan for safety violations before first move.
    Returns (is_safe: bool, reason: str, warnings: list[str]).
    Warnings do not block the plan (e.g. picking a stale object).
    """
    warnings = []
    if not isinstance(plan, list) or len(plan) == 0:
        return False, "Plan is empty or not a list", warnings

    for i, step in enumerate(plan):
        if not isinstance(step, dict):
            return False, f"Step {i}: not an object ({step!r})", warnings
        act = step.get('action', '')

        # Every coordinate present must be a real number (null z is possible
        # now that the scene reports unknown heights as null)
        for k in ['x', 'y', 'z', 'approach_z']:
            if k in step and not _is_number(step[k]):
                return False, f"Step {i} ({act}): {k}={step[k]!r} is not a number", warnings

        # Z floor check
        for k in ['z', 'approach_z']:
            if k in step and step[k] < z_floor:
                return False, f"Step {i} ({act}): {k}={step[k]:.3f} violates floor {z_floor:.3f}", warnings

        # approach_z must be high enough
        if act == 'pick':
            if step.get('approach_z', 1.0) < max(0.15, z_floor):
                return False, f"Step {i}: approach_z={step.get('approach_z')} too low", warnings
            # Object must be named and exist in scene
            oid = step.get('object_id')
            if not oid:
                return False, f"Step {i}: pick has no object_id", warnings
            if oid not in scene:
                return False, f"Step {i}: object_id '{oid}' not in scene", warnings
            obj = scene[oid] if isinstance(scene[oid], dict) else {}
            if obj.get('z') is None:
                return False, f"Step {i}: '{oid}' has unknown height", warnings
            # Stale objects: warning only — arm_controller uses the last seen position
            if obj.get('stale', False):
                warnings.append(f"Step {i}: object '{oid}' is stale but proceeding")

        # Place Z check
        if act == 'place' and step.get('z', 1.0) < max(0.10, z_floor):
            return False, f"Step {i}: place z={step.get('z'):.3f} too low", warnings

        # Workspace bounds — x upper bound matches arm_controller_node.py's
        # clamp (0.60, widened 2026-09-30; see its comment for why)
        if 'x' in step and not (sg.PLAN_X_RANGE[0] <= step['x'] <= sg.PLAN_X_RANGE[1]):
            return False, f"Step {i}: x={step['x']:.3f} outside workspace", warnings
        if 'y' in step and not (sg.PLAN_Y_RANGE[0] <= step['y'] <= sg.PLAN_Y_RANGE[1]):
            return False, f"Step {i}: y={step['y']:.3f} outside workspace", warnings

    return True, f"Plan approved — {len(plan)} steps", warnings


def _is_number(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


class LLMPlannerNode(Node):

    def __init__(self):
        super().__init__('llm_planner')

        # Declare model as a ROS2 parameter — change without recompiling
        self.declare_parameter('model', 'qwen2.5:7b')
        # A hung Ollama must not leave is_planning stuck True (every later
        # command would be ignored as "already planning").
        self.declare_parameter('llm_timeout_s', 60.0)
        self.declare_parameter('max_scene_age_s', 3.0)
        self.model = self.get_parameter('model').value
        self.max_scene_age = float(self.get_parameter('max_scene_age_s').value)
        self.llm = ollama.Client(timeout=float(self.get_parameter('llm_timeout_s').value))
        self._scene_time = 0.0

        # ── state ────────────────────────────────────────────────────
        self.latest_scene    = {}
        self.latest_scene_raw = '{}'
        self.is_planning     = False   # prevent overlapping calls
        self._lock           = threading.Lock()

        # ── subscribers ──────────────────────────────────────────────
        self.create_subscription(
            String, '/scene_snapshot',
            self.scene_callback, 10)
        self.create_subscription(
            String, '/voice_command',
            self.command_callback, 10)

        # ── publishers ───────────────────────────────────────────────
        self.plan_pub   = self.create_publisher(String, '/action_plan',    10)
        self.status_pub = self.create_publisher(String, '/planner_status', 10)

        self.get_logger().info(
            f'LLMPlannerNode ready — model: {self.model}'
        )
        self._publish_status('READY — waiting for voice command')

    # ── CALLBACKS ────────────────────────────────────────────────────
    def scene_callback(self, msg):
        """Cache latest scene — runs at 5Hz, never blocks."""
        try:
            self.latest_scene     = json.loads(msg.data)
            self.latest_scene_raw = msg.data
            self._scene_time      = time.time()
        except json.JSONDecodeError:
            pass

    def command_callback(self, msg):
        """Trigger planning on voice command — run in separate thread."""
        command = msg.data.strip()
        if not command:
            return

        with self._lock:
            if self.is_planning:
                self.get_logger().warn(
                    'Already planning — ignoring new command. '
                    'Wait for current plan to complete.'
                )
                return
            self.is_planning = True

        # Run LLM call in background thread — keeps spin() responsive
        t = threading.Thread(
            target=self._plan_and_publish,
            args=(command,),
            daemon=True
        )
        t.start()

    # ── CORE PLANNING LOGIC ──────────────────────────────────────────
    def _plan_and_publish(self, command):
        """Called in background thread. Calls LLM, validates, publishes."""
        try:
            self.get_logger().info(f'Command: "{command}"')
            self._publish_status(f'PLANNING: {command}')

            geom, geom_err = sg.load_geometry()
            z_floor = sg.flange_floor_z(geom)
            if geom is None:
                self.get_logger().warn(
                    f'No table geometry ({geom_err}) — using conservative flange floor '
                    f'z={z_floor:.2f}. Record the table in the dashboard.')
            system_prompt = SYSTEM_PROMPT.replace('{Z_FLOOR}', f'{z_floor:.2f}')

            scene_age = time.time() - self._scene_time
            if self._scene_time and scene_age > self.max_scene_age:
                self._publish_status(f'ERROR: scene data is {scene_age:.0f}s old — is scene_graph_node running?')
                self.get_logger().warn(
                    f'Refusing to plan on a {scene_age:.1f}s-old scene snapshot '
                    f'(limit {self.max_scene_age:.0f}s).')
                return
            if not self.latest_scene:
                self._publish_status('ERROR: no scene data yet')
                self.get_logger().warn('No scene data — publish to /scene_snapshot first')
                return

            # Build prompt — only send reachable objects. Unreachable
            # background clutter (false-positive YOLO/color detections
            # outside the workspace, common with this scene's noise) was
            # blowing the full scene up past 40KB / ~11k tokens, which
            # overflows Ollama's default context window and made the model
            # return an empty plan with empty reasoning every time rather
            # than actually failing loudly. Validation below still checks
            # against the FULL scene, not this filtered view, so the plan
            # is never less safe — the LLM just can't reference something
            # it was never told the arm can't reach anyway.
            prompt_scene = {
                obj_id: obj for obj_id, obj in self.latest_scene.items()
                if obj.get('reachable') and not obj.get('stale', False)
            }
            if not prompt_scene:
                self.get_logger().warn(
                    f'No reachable objects in scene ({len(self.latest_scene)} '
                    f'total, 0 reachable) — sending the LLM an empty world state.')
            user_msg = (
                f"CURRENT WORLD STATE:\n"
                f"{json.dumps(prompt_scene, indent=2)}\n\n"
                f"ENGINEER COMMAND: \"{command}\"\n\n"
                f"Reason step by step, then output the JSON."
            )

            # Call LLM
            t0 = time.time()
            response = self.llm.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user',   'content': user_msg},
                ],
                format='json'
            )
            latency = time.time() - t0

            # Parse response
            raw = response['message']['content']
            parsed = json.loads(raw)
            plan      = parsed.get('plan', [])
            # Sanitise — ensure each step is a dict not a string
            plan = [
                json.loads(s) if isinstance(s, str) else s
                for s in plan
                if s  # skip empty
            ]
            reasoning = parsed.get('reasoning', '')

            self.get_logger().info(
                f'LLM response in {latency:.1f}s — '
                f'{len(plan)} steps | Reasoning: {reasoning[:80]}...'
            )

            # Re-derive move_to heights from the matched scene object rather
            # than trusting the LLM's "z + 0.15" arithmetic (see comment on
            # fix_move_to_heights) — then validate as usual.
            plan = fix_move_to_heights(plan, self.latest_scene, z_floor)
            ok, reason, warnings = validate_plan(plan, self.latest_scene, z_floor)
            for w in warnings:
                self.get_logger().warn(w)

            if not ok:
                self.get_logger().warn(f'Plan REJECTED: {reason}')
                self._publish_status(f'REJECTED: {reason}')
                return

            # Publish approved plan
            self.get_logger().info(f'Plan APPROVED: {reason}')
            self._publish_status(f'EXECUTING: {len(plan)} steps')

            out = String()
            out.data = json.dumps({
                'command':   command,
                'reasoning': reasoning,
                'plan':      plan,
                'latency_s': round(latency, 2),
                'model':     self.model,
            })
            self.plan_pub.publish(out)

            # Log each step
            for i, step in enumerate(plan):
                self.get_logger().info(f'  Step {i+1}: {json.dumps(step)}')

        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON parse error: {e}')
            self._publish_status(f'ERROR: JSON parse failed')
        except Exception as e:
            self.get_logger().error(f'Planning failed: {e}')
            self._publish_status(f'ERROR: {str(e)[:80]}')
        finally:
            with self._lock:
                self.is_planning = False
            self._publish_status('READY — waiting for voice command')

    # ── HELPERS ──────────────────────────────────────────────────────
    def _publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LLMPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
