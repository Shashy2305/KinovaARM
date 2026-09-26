% =========================================================================
%  Step4_PickAndPlace.m
%  Given a detected object in OAK-D camera frame,
%  computes the Kinova Gen3 EE target pose in base frame for pick-and-place.
%
%  Workflow:
%    1. OAK-D detects object → gives 3D point in camera frame (X,Y,Z in metres)
%    2. This script transforms it to base frame
%    3. Outputs: target EE position + quaternion for MoveIt2
%
%  You call this from MATLAB to generate the goal pose, then send it
%  to ROS2/MoveIt2 via the companion Python node (publish_goal_pose.py).
%
%  >> Step4_PickAndPlace
% =========================================================================

clear; clc;
CALIB_CONFIG;

fprintf('STEP 4: Pick-and-Place Transform\n\n');

% ── Load calibrated transforms ─────────────────────────────────────────────
pp_path = fullfile(RESULTS_DIR, 'pickplace_transforms.mat');
if ~isfile(pp_path)
    error('Run Step3_VerifyAndExport first to generate pickplace_transforms.mat');
end
load(pp_path, 'pickplace');

T_base_cam = pickplace.T_base_cam;

fprintf('Loaded T_base_cam (mean calib error: %.2f mm)\n\n', pickplace.mean_err_mm);

% =========================================================================
%  EXAMPLE USAGE
%  Replace the values below with real OAK-D detections!
% =========================================================================

% ── Simulated OAK-D object detection ──────────────────────────────────────
% These come from your OAK-D depth stream (in metres, camera frame).
% In your ROS2 system, subscribe to /global_camera/stereo/points and
% get the XYZ of the detected object centroid.
%
% Camera frame convention (standard optical):
%   X = right, Y = down, Z = forward (depth)

example_detections = [
%   X_cam   Y_cam   Z_cam   Label
    0.05,   0.10,   0.45,   "Red block";
   -0.08,   0.05,   0.38,   "Blue cube";
    0.00,  -0.02,   0.52,   "Target object";
];

fprintf('%-15s  %-30s  %-30s\n', 'Object', 'Camera frame (m)', 'Base frame (m)');
fprintf('%s\n', repmat('-', 1, 78));

goal_poses = [];   % store for export to ROS2

for i = 1:size(example_detections, 1)
    x_cam = double(example_detections(i, 1));
    y_cam = double(example_detections(i, 2));
    z_cam = double(example_detections(i, 3));
    label = example_detections(i, 4);

    % ── Transform to base frame ──────────────────────────────────────────
    p_cam  = [x_cam; y_cam; z_cam; 1];
    p_base = T_base_cam * p_cam;

    x_base = p_base(1);
    y_base = p_base(2);
    z_base = p_base(3);

    fprintf('%-15s  [%6.3f %6.3f %6.3f]  →  [%6.3f %6.3f %6.3f]\n', ...
        label, x_cam, y_cam, z_cam, x_base, y_base, z_base);

    % ── Target EE pose for pick ─────────────────────────────────────────
    % Approach from above: EE z-axis pointing down into the object
    % You may need to adjust this based on your gripper orientation
    z_approach_offset = 0.15;   % metres above object to approach
    grasp_height      = 0.00;   % extra z offset at grasp (tune per object)

    % Approach pose (above object)
    T_approach = build_top_down_pose(x_base, y_base, z_base + z_approach_offset);

    % Grasp pose (at object)
    T_grasp    = build_top_down_pose(x_base, y_base, z_base + grasp_height);

    % Convert to quaternion for MoveIt2
    q_approach = rotm2quat(T_approach(1:3,1:3));   % [qw qx qy qz]
    q_grasp    = rotm2quat(T_grasp(1:3,1:3));

    goal_poses(end+1).label      = label;
    goal_poses(end).approach_xyz = T_approach(1:3,4)';
    goal_poses(end).approach_q   = q_approach;
    goal_poses(end).grasp_xyz    = T_grasp(1:3,4)';
    goal_poses(end).grasp_q      = q_grasp;
end

fprintf('\n\n=== MoveIt2 Goal Poses (base_link frame) ===\n\n');

for i = 1:length(goal_poses)
    gp = goal_poses(i);
    fprintf('Object: %s\n', gp.label);
    fprintf('  APPROACH: x=%.4f y=%.4f z=%.4f | qw=%.4f qx=%.4f qy=%.4f qz=%.4f\n', ...
        gp.approach_xyz(1), gp.approach_xyz(2), gp.approach_xyz(3), ...
        gp.approach_q(1), gp.approach_q(2), gp.approach_q(3), gp.approach_q(4));
    fprintf('  GRASP   : x=%.4f y=%.4f z=%.4f | qw=%.4f qx=%.4f qy=%.4f qz=%.4f\n\n', ...
        gp.grasp_xyz(1), gp.grasp_xyz(2), gp.grasp_xyz(3), ...
        gp.grasp_q(1), gp.grasp_q(2), gp.grasp_q(3), gp.grasp_q(4));
end

% ── Save goal poses as JSON for publish_goal_pose.py ──────────────────────
write_goal_poses_json(goal_poses, RESULTS_DIR);

fprintf('Saved: %s/goal_poses.json\n', RESULTS_DIR);
fprintf('Next : ros2 run your_pkg publish_goal_pose.py\n\n');

% ── Visualise ─────────────────────────────────────────────────────────────
figure('Name', 'Pick-and-Place in Base Frame', 'Position', [100 100 800 600]);
hold on; grid on; axis equal;

% Draw camera position
cam_pos = T_base_cam(1:3,4);
scatter3(cam_pos(1), cam_pos(2), cam_pos(3), 200, 'b', 'filled', '^');
text(cam_pos(1), cam_pos(2), cam_pos(3)+0.05, 'OAK-D', 'FontSize', 10, 'Color', 'b');

% Draw robot base
scatter3(0, 0, 0, 200, 'k', 'filled', 's');
text(0, 0, -0.05, 'Robot Base', 'FontSize', 10);

% Draw object detections and approach arrows
colors = lines(length(goal_poses));
for i = 1:length(goal_poses)
    gp = goal_poses(i);
    scatter3(gp.grasp_xyz(1), gp.grasp_xyz(2), gp.grasp_xyz(3), ...
             120, colors(i,:), 'filled', 'o');
    text(gp.grasp_xyz(1)+0.02, gp.grasp_xyz(2), gp.grasp_xyz(3), ...
         char(gp.label), 'FontSize', 9, 'Color', colors(i,:));
    quiver3(gp.approach_xyz(1), gp.approach_xyz(2), gp.approach_xyz(3), ...
            gp.grasp_xyz(1)-gp.approach_xyz(1), ...
            gp.grasp_xyz(2)-gp.approach_xyz(2), ...
            gp.grasp_xyz(3)-gp.approach_xyz(3), ...
            0, 'Color', colors(i,:), 'LineWidth', 2, 'MaxHeadSize', 0.5);
end

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Pick-and-Place Targets in Kinova Base Frame');
view(45, 30);

saveas(gcf, fullfile(RESULTS_DIR, 'pickplace_visualization.png'));
fprintf('Saved visualization to %s/pickplace_visualization.png\n\n', RESULTS_DIR);


% =========================================================================
%  HELPER FUNCTIONS
% =========================================================================

function T = build_top_down_pose(x, y, z)
% Build a 4×4 pose with EE pointing straight down (Z-down approach).
% Adjust rotation if your gripper has a different default orientation.
    R = [1  0  0;     % EE x → world x
         0 -1  0;     % EE y → world -y  (gripper pointing down)
         0  0 -1];    % EE z → world -z
    T = eye(4);
    T(1:3,1:3) = R;
    T(1:3,  4) = [x; y; z];
end


function write_goal_poses_json(goal_poses, out_dir)
% Write goal poses to JSON so publish_goal_pose.py can read them.
    fid = fopen(fullfile(out_dir, 'goal_poses.json'), 'w');
    fprintf(fid, '[\n');
    for i = 1:length(goal_poses)
        gp = goal_poses(i);
        fprintf(fid, '  {\n');
        fprintf(fid, '    "label": "%s",\n', gp.label);
        fprintf(fid, '    "approach": {"x": %.6f, "y": %.6f, "z": %.6f, "qw": %.6f, "qx": %.6f, "qy": %.6f, "qz": %.6f},\n', ...
            gp.approach_xyz(1), gp.approach_xyz(2), gp.approach_xyz(3), ...
            gp.approach_q(1), gp.approach_q(2), gp.approach_q(3), gp.approach_q(4));
        fprintf(fid, '    "grasp":    {"x": %.6f, "y": %.6f, "z": %.6f, "qw": %.6f, "qx": %.6f, "qy": %.6f, "qz": %.6f}\n', ...
            gp.grasp_xyz(1), gp.grasp_xyz(2), gp.grasp_xyz(3), ...
            gp.grasp_q(1), gp.grasp_q(2), gp.grasp_q(3), gp.grasp_q(4));
        if i < length(goal_poses)
            fprintf(fid, '  },\n');
        else
            fprintf(fid, '  }\n');
        end
    end
    fprintf(fid, ']\n');
    fclose(fid);
end
