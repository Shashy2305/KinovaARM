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
