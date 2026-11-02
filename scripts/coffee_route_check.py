# READ-ONLY IK search (used 2026-11-02): for each cup spot and side azimuth, can the arm pick the cup AND enter the coffee tray toward the same azimuth? Needs MoveIt (/compute_ik). Run: ROS_DOMAIN_ID=42 python3 scripts/coffee_route_check.py
import sys, math
import numpy as np, rclpy
sys.path.insert(0,'/mnt/ros_workspace/Shashproject/scripts'); sys.path.insert(0,'/mnt/ros_workspace/Shashproject/src/thesis_robot')
from angled_approach_study import JOINTS, Study, GRASP_QUAT_XYZW
from thesis_robot import grasp_options as go, motion_utils as mu, locations as locs, safety_geometry as sg
rclpy.init(); n=Study(); n.spin_until(lambda: n.js is not None and n.ik.service_is_ready(),10)
geom,_=sg.load_geometry(); margin=geom['table_top_z']+sg.LINK_TABLE_CLEARANCE_M
home=[math.radians(v) for v in (0.0,-20.0,179.9,-130.0,0.0,55.0,90.0)]
def ik(pos,quat,seed,tries=3):
    for _ in range(tries):
        s=n.solve_ik(list(pos),quat,seed)
        if s is not None: return s
def chain(pad_xy,pad_z,az,standoff,seed_hover,extra=()):
    """the side-pick chain (high, pre, grasp, lift) then extra poses; True if some wrist orientation works"""
    for flip in (False,True):
        quat=go.side_quat(az,flip)
        high,pre,grasp,a=go.side_poses(pad_xy,pad_z,az)
        pre=grasp-standoff*a; high=pre+np.array([0,0,go.HIGH_CLEAR_M])
        seed=seed_hover; ok=True; first=True
        for name,fl in (('high',high),('pre',pre),('grasp',grasp),('lift',grasp+np.array([0,0,0.10]))):
            s=ik(fl,quat,seed)
            if s is None: ok=False; break
            u=mu.unwrap_to_seed(s,seed_hover if first else seed)
            if u is None: ok=False; break
            if first and max(mu.joint_deltas(u,seed_hover))>mu.MAX_JOINT_DELTA_RAD: ok=False; break
            hs=n.pad_heights(s)
            if hs is None or min(hs[:2])<margin-0.002: ok=False; break
            seed=s; first=False
        if ok: return True
    return False
loc=locs.get('coffee_tray_grasp'); pad=loc['pad_midpoint']
AZS=(45.0,60.0,75.0,90.0,105.0,120.0,135.0)
def hover_for(xy):
    for _ in range(4):
        h=n.solve_ik([xy[0],xy[1],0.38],GRASP_QUAT_XYZW,home)
        if h: return h
tray={}
for az in AZS:
    best=None
    for so in (0.17,0.22):
        p=locs.approach_poses(dict(loc,approach_azimuth_deg=az,standoff_m=so)); gap,_w=locs.approach_clearance(p, locs.fixed_obstacle_rects())
        h=hover_for((p['high'][0],p['high'][1]))
        ok=bool(h) and chain((pad[0],pad[1]),pad[2],az,so,h)
        if ok and gap>=0.03: best=(so,gap); break
    tray[az]=best
    print(f'tray entry toward {az:5.0f}: ' + (f'OK with standoff {best[0]:.2f} (fingertips {best[1]*100:.1f} cm from the machine)' if best else 'no'), flush=True)
spots=[(x,y) for x in (0.30,0.35,0.40,0.45,0.50,0.55) for y in (-0.20,-0.10,0.0,0.10)]
print('pick spots (side grasp, standard 17 cm run-in) where the SAME azimuth also works at the tray:')
for xy in spots:
    h=hover_for(xy); good=[]
    if h:
        for az in AZS:
            if tray[az] and chain(xy,pad[2],az,go.STANDOFF_M,h): good.append(int(az))
    print(f'  cup at {xy}: {good}', flush=True)
rclpy.shutdown()
