import yaml

from thesis_robot import safety_geometry as sg

GEOM = {'table_top_z': -0.06, 'x': [-0.15, 0.75], 'y': [-0.80, 0.85], 'rear_wall_x': -0.20}


def test_flange_floor_keeps_fingertips_above_table():
    # The incident: flange z=0.12 with the gripper pointing down put the
    # fingertips (0.19 m further along the tool axis) at about -0.07.
    tip_for_old_target = 0.12 - sg.TCP_REACH_M
    assert tip_for_old_target < GEOM['table_top_z']
    floor = sg.flange_floor_z(GEOM)
    assert floor - sg.TCP_REACH_M > GEOM["table_top_z"] + sg.LINK_TABLE_CLEARANCE_M


def test_flange_floor_without_geometry_is_conservative():
    assert sg.flange_floor_z(None) == sg.FALLBACK_FLANGE_Z_MIN
    assert sg.FALLBACK_FLANGE_Z_MIN - sg.TCP_REACH_M > 0.0


def test_check_rejects_pad_below_table_and_behind_robot():
    ok, why = sg.check_link_positions({'left_inner_finger_pad': (0.2, 0.1, -0.07)}, GEOM)
    assert not ok and 'below the table' in why
    ok, why = sg.check_link_positions({'bracelet_link': (-0.3, 0.0, 0.4)}, GEOM)
    assert not ok and 'behind' in why
    ok, _ = sg.check_link_positions(
        {'left_inner_finger_pad': (0.2, 0.1, 0.02), 'bracelet_link': (0.2, 0.1, 0.2)}, GEOM)
    assert ok


def test_collision_boxes_match_geometry():
    slab, wall = sg.collision_boxes(GEOM)
    (sid, (dx, dy, dz), (cx, cy, cz)) = slab
    assert sid == 'table_slab'
    assert abs((cz + dz / 2.0) - GEOM['table_top_z']) < 1e-9      # top face == table top
    assert abs((cx - dx / 2.0) - sg.SLAB_X_MIN) < 1e-9             # clear of the robot base
    assert abs((cy - dy / 2.0) - GEOM['y'][0]) < 1e-9
    (wid, (wx, _, _), (wcx, _, _)) = wall
    assert wid == 'rear_safety_wall'
    assert abs((wcx + wx / 2.0) - GEOM['rear_wall_x']) < 1e-9       # front face at the wall plane


def test_geometry_from_points():
    pts = [(0.3, -0.4, -0.05), (0.3, 0.4, -0.048), (0.6, 0.0, -0.052), (0.5, 0.2, -0.05)]
    g, err = sg.geometry_from_points(pts)
    assert err is None
    assert abs(g['table_top_z'] - (-0.05 - sg.PAD_FRAME_ABOVE_TIP_M)) < 1e-3
    assert g['x'][0] <= sg.DEFAULT_FOOTPRINT_X[0] and g['y'][1] >= sg.DEFAULT_FOOTPRINT_Y[1]


def test_geometry_from_points_rejects_uneven_or_too_few():
    assert sg.geometry_from_points([(0, 0, 0), (0, 0, 0)])[1] is not None
    g, err = sg.geometry_from_points([(0.3, 0, -0.05), (0.3, 0.1, 0.10), (0.4, 0, -0.05)])
    assert g is None and 'redo' in err


def test_load_geometry_fails_closed(tmp_path):
    missing = tmp_path / 'nope.yaml'
    assert sg.load_geometry(str(missing))[0] is None
    bad = tmp_path / 'bad.yaml'
    bad.write_text('table_top_z: banana\n')
    assert sg.load_geometry(str(bad))[0] is None
    absurd = tmp_path / 'absurd.yaml'
    absurd.write_text(yaml.safe_dump({'table_top_z': 3.0, 'x': [0, 1], 'y': [0, 1]}))
    assert sg.load_geometry(str(absurd))[0] is None
    good = tmp_path / 'good.yaml'
    sg.save_geometry(dict(GEOM), str(good))
    g, err = sg.load_geometry(str(good))
    assert err is None and g['table_top_z'] == -0.06


def test_sample_indices_include_endpoints():
    assert sg.sample_indices(0) == []
    assert sg.sample_indices(5) == [0, 1, 2, 3, 4]
    idx = sg.sample_indices(1000, max_samples=40)
    assert idx[0] == 0 and idx[-1] == 999 and len(idx) <= 41
    assert idx == sorted(idx)


WS = {'x': (0.05, 0.60), 'y': (-0.55, 0.55), 'z': (-0.40, 2.0)}


def test_reachable_matches_planner_limits_not_just_the_workspace_box():
    assert sg.is_reachable(0.30, 0.10, -0.03, WS, -0.06)
    assert not sg.is_reachable(0.23, 0.43, -0.03, WS, -0.06)     # inside the box, outside the planner's |y|<=0.35
    assert not sg.is_reachable(0.05, 0.0, -0.03, WS, -0.06)      # x below the planner minimum
    assert not sg.is_reachable(0.30, 0.10, None, WS, -0.06)


def test_objects_below_the_table_are_not_reachable():
    # the live scene had cups at z=-0.28 with the table top at about -0.06
    assert not sg.is_reachable(0.34, 0.20, -0.28, WS, -0.06)
    assert sg.is_reachable(0.34, 0.20, -0.28, WS, None)           # no recorded table: cannot tell


def test_a_path_may_start_below_the_margin_but_not_go_lower():
    geom = {'table_top_z': -0.0125, 'x': [-0.15, 0.78], 'y': [-0.9, 0.85], 'rear_wall_x': -0.2}
    start = {'left_inner_finger_pad': (0.26, -0.24, 0.029), 'end_effector_link': (0.26, -0.24, 0.227)}
    floors = sg.start_floors(start, geom)
    assert floors['left_inner_finger_pad'] < 0.038 and abs(floors['end_effector_link'] - 0.0375) < 1e-9
    assert sg.check_link_positions(start, geom)[0] is False                        # the old rule: stuck
    assert sg.check_link_positions(start, geom, floors)[0]                         # first waypoint accepted
    lifting = {'left_inner_finger_pad': (0.26, -0.24, 0.10), 'end_effector_link': (0.26, -0.24, 0.30)}
    assert sg.check_link_positions(lifting, geom, floors)[0]
    sinking = {'left_inner_finger_pad': (0.26, -0.24, 0.020), 'end_effector_link': (0.26, -0.24, 0.227)}
    assert not sg.check_link_positions(sinking, geom, floors)[0]                   # lower than it started: refused
    # a link that began ABOVE the margin still has the full margin
    ok, _ = sg.check_link_positions({'left_inner_finger_pad': (0.26, -0.24, 0.03), 'end_effector_link': (0.26, -0.24, 0.3)}, geom, floors)
    assert ok is True
    ok, why = sg.check_link_positions({'end_effector_link': (0.26, -0.24, 0.03)}, geom, floors)
    assert not ok


def test_detections_on_the_robot_are_recognised_and_real_objects_are_not():
    chain = [(0, 0, 0), (0, 0, 0.16), (0.0, -0.1, 0.4), (0.3, -0.1, 0.5), (0.3, -0.1, 0.33)]     # base ... flange hovering at z 0.33
    fingers = [((0.3, -0.1, 0.33), (0.231, -0.115, 0.153)), ((0.3, -0.1, 0.33), (0.369, -0.085, 0.153))]
    assert sg.point_on_robot((0.231, -0.115, 0.129), chain, fingers)          # "bottle_04": the left finger (2026-10-05)
    assert sg.point_on_robot((0.3, -0.1, 0.40), chain, fingers)               # the wrist
    assert not sg.point_on_robot((0.304, -0.12, 0.01), chain, fingers)        # the mouse on the table under the gripper
    assert not sg.point_on_robot((0.54, -0.01, 0.045), chain, fingers)        # the mug on the other side
    assert not sg.point_on_robot((0.30, -0.10, 0.05), chain, fingers)         # an object grasped between the pads (6 cm from each)


def test_objects_behind_the_robot_or_off_the_table_are_not_in_the_table_region():
    geom = {'table_top_z': -0.0125, 'x': [-0.15, 0.784], 'y': [-0.899, 0.85], 'rear_wall_x': -0.2}
    assert sg.in_table_region(geom, 0.43, -0.5, 0.06)               # the mug, far left
    assert sg.in_table_region(geom, 0.32, 0.33, 0.10)               # the bottle
    assert sg.in_table_region(geom, 0.51, -0.29, 0.0)               # the mouse
    assert not sg.in_table_region(geom, -0.30, 0.39, -0.03)         # a mouse on the operator's desk behind the robot
    assert not sg.in_table_region(geom, -1.52, 0.81, -0.04)         # a cup across the room
    assert not sg.in_table_region(geom, 0.5, 1.2, 0.05)             # beyond the table's side edge
    assert not sg.in_table_region(geom, 0.4, 0.2, 0.9)              # far above the table (a lamp, a person's head)
    assert not sg.in_table_region(geom, 0.4, 0.2, -0.5)             # below it


def test_detections_in_the_carried_objects_volume_are_recognised():
    pad_mid = (0.37, 0.18, 0.12)                                   # fingers holding a mouse above the target
    assert sg.in_held_volume((0.367, 0.22, 0.087), pad_mid)         # "remote_00": the held mouse, 4 cm off
    assert sg.in_held_volume((0.332, 0.191, 0.09), pad_mid)         # "blue_obj_02"
    assert sg.in_held_volume((0.37, 0.18, -0.02), pad_mid)          # a bottle hanging below the pads
    assert not sg.in_held_volume((0.30, 0.31, 0.10), pad_mid)       # the real bottle, 14 cm away
    assert not sg.in_held_volume((0.37, 0.18, 0.4), pad_mid)        # high above
