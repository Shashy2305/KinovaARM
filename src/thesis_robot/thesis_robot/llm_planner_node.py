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

# ── SYSTEM PROMPT ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a task planner for a Kinova Gen3 7-DOF robotic arm.

SAFETY RULES — every action must obey ALL of these without exception:
  RULE 1: ALL z values >= 0.08 at minimum
  RULE 2: approach_z in pick >= 0.15 (approach from well above object)
  RULE 3: place z >= 0.10 (place gently, never at floor level)
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
    z = X's z + 0.15 (hover above it, don't descend onto it).
    Do NOT use pick for these — pick closes the gripper on the object,
    which is not what "go near" means.
  - "pick up X", "grab X", "get X" -> a pick step (approach_z>=0.15),
    optionally followed by place/open_gripper if the command also says
    where to put it down.
  - If the named object isn't in the world state below, or every
    instance of it is stale, say so in "reasoning" and return an empty
    plan rather than guessing a position.

WORKSPACE BOUNDS: x=[0.10, 0.60]  y=[-0.35, 0.35]  z=[0.08, 0.50]

RESPOND WITH EXACTLY THIS JSON STRUCTURE — no other format:
{
  "reasoning": "step by step thinking about which objects, sequence, safety checks",
  "plan": [
    {"action": "...", ...},
    {"action": "...", ...}
  ]
}"""

# ── SAFETY VALIDATOR ─────────────────────────────────────────────────
def validate_plan(plan, scene):
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
            if k in step and step[k] < 0.08:
                return False, f"Step {i} ({act}): {k}={step[k]:.3f} violates floor", warnings

        # approach_z must be high enough
        if act == 'pick':
            if step.get('approach_z', 1.0) < 0.15:
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
        if act == 'place' and step.get('z', 1.0) < 0.10:
            return False, f"Step {i}: place z={step.get('z'):.3f} too low", warnings

        # Workspace bounds — x upper bound matches arm_controller_node.py's
        # clamp (0.60, widened 2026-09-30; see its comment for why)
        if 'x' in step and not (0.10 <= step['x'] <= 0.60):
            return False, f"Step {i}: x={step['x']:.3f} outside workspace", warnings
        if 'y' in step and not (-0.35 <= step['y'] <= 0.35):
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
        self.model = self.get_parameter('model').value

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
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
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

            # Validate
            ok, reason, warnings = validate_plan(plan, self.latest_scene)
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
