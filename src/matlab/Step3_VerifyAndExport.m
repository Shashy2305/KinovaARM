% =========================================================================
%  Step3_VerifyAndExport.m
%  Verifies calibration accuracy and exports new TF values for
%  robot_launch.py (base_link → global_camera_link).
%
%  Verification: project a 3D point (from camera) into base frame
%  using the calibrated T_base_cam, then compare to robot FK.
%
%  >> Step3_VerifyAndExport
% =========================================================================

clear; clc;
CALIB_CONFIG;

fprintf('STEP 3: Verification & Export\n\n');

% ── Load results ──────────────────────────────────────────────────────────
load(fullfile(RESULTS_DIR, 'calib_result.mat'), 'calib_result');
load(fullfile(RESULTS_DIR, 'corner_data.mat'),  'T_cam_board', 'T_base_ee', 'n_valid');

T_base_cam = calib_result.T_base_cam;
T_cam_base = inv(T_base_cam);   % camera frame → base frame (inverse)

fprintf('Calibrated T_base_cam:\n');
disp(T_base_cam);

% ── Verification: board origin in base frame ───────────────────────────────
% For each pose, compute board origin two ways:
%   Way 1 (camera): T_base_cam * T_cam_board * [0 0 0 1]'
%   Way 2 (robot) : T_base_ee  * T_ee_board  * [0 0 0 1]'
%   T_ee_board is approximately constant — estimate it as mean.

% Estimate T_ee_board from all poses
T_ee_board_list = zeros(4,4,n_valid);
for i = 1:n_valid
    T_ee_board_list(:,:,i) = inv(T_base_ee{i}) * T_base_cam * T_cam_board{i};
end
T_ee_board_mean = mean(T_ee_board_list, 3);

% Reprojection errors in 3D base frame
errs_3d = zeros(n_valid, 1);
board_pts_cam  = zeros(n_valid, 3);
board_pts_rob  = zeros(n_valid, 3);

for i = 1:n_valid
    % Camera path
    p_cam = T_base_cam * T_cam_board{i} * [0;0;0;1];
    % Robot path
    p_rob = T_base_ee{i} * T_ee_board_mean * [0;0;0;1];

    board_pts_cam(i,:) = p_cam(1:3)';
    board_pts_rob(i,:) = p_rob(1:3)';
    errs_3d(i) = norm(p_cam(1:3) - p_rob(1:3));
end

fprintf('3D Verification (board origin in base frame):\n');
fprintf('  Mean error : %.2f mm\n', mean(errs_3d)*1000);
fprintf('  Max  error : %.2f mm\n', max(errs_3d)*1000);
fprintf('  Std  error : %.2f mm\n\n', std(errs_3d)*1000);

if mean(errs_3d)*1000 < 5
    fprintf('  ✓ EXCELLENT calibration (< 5 mm)\n\n');
elseif mean(errs_3d)*1000 < 15
    fprintf('  ✓ GOOD calibration (< 15 mm) — suitable for pick-and-place\n\n');
else
    fprintf('  ✗ HIGH error (> 15 mm) — recollect data with more diverse poses\n\n');
end

% ── 3D error plot ─────────────────────────────────────────────────────────
figure('Name', '3D Verification', 'Position', [100 100 1100 500]);

subplot(1,2,1);
scatter3(board_pts_cam(:,1), board_pts_cam(:,2), board_pts_cam(:,3), ...
         60, 'b', 'filled'); hold on;
scatter3(board_pts_rob(:,1), board_pts_rob(:,2), board_pts_rob(:,3), ...
         60, 'r', '^');
for i = 1:n_valid
    plot3([board_pts_cam(i,1) board_pts_rob(i,1)], ...
          [board_pts_cam(i,2) board_pts_rob(i,2)], ...
          [board_pts_cam(i,3) board_pts_rob(i,3)], 'k--', 'LineWidth', 0.8);
end
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Board Origin: Camera (blue) vs Robot FK (red)');
legend('Camera path', 'Robot FK', 'Location', 'best');
grid on; axis equal; view(45, 30);

subplot(1,2,2);
bar(errs_3d * 1000, 'FaceColor', [0.2 0.5 0.8]);
hold on;
yline(mean(errs_3d)*1000, 'r--', 'LineWidth', 2, 'Label', ...
      sprintf('Mean=%.1fmm', mean(errs_3d)*1000));
xlabel('Pose index'); ylabel('3D error (mm)');
title('Calibration Verification Error per Pose');
grid on;

saveas(gcf, fullfile(RESULTS_DIR, 'verification_plot.png'));

% ── Compare with old TF ───────────────────────────────────────────────────
q_new = calib_result.q_wxyz;   % [qw qx qy qz]
t_new = calib_result.t_xyz;    % [x y z]

fprintf('=== CALIBRATION RESULTS ===\n\n');

fprintf('OLD (in robot_launch.py):\n');
fprintf('  x=%.5f  y=%.5f  z=%.5f\n', EXISTING_T_x, EXISTING_T_y, EXISTING_T_z);
fprintf('  qx=%.5f  qy=%.5f  qz=%.5f  qw=%.5f\n\n', ...
        EXISTING_Q_x, EXISTING_Q_y, EXISTING_Q_z, EXISTING_Q_w);

fprintf('NEW (freshly calibrated — use this!):\n');
fprintf('  x=%.5f  y=%.5f  z=%.5f\n', t_new(1), t_new(2), t_new(3));
fprintf('  qx=%.5f  qy=%.5f  qz=%.5f  qw=%.5f\n\n', ...
        q_new(2), q_new(3), q_new(4), q_new(1));

% ── Write updated robot_launch.py snippet ────────────────────────────────
launch_snippet = sprintf([...
'    calibration_tf = Node(\n'...
'        package="tf2_ros",\n'...
'        executable="static_transform_publisher",\n'...
'        name="calibration_tf_publisher",\n'...
'        output="log",\n'...
'        arguments=[\n'...
'            "--x",  "%.5f",\n'...
'            "--y",  "%.5f",\n'...
'            "--z",  "%.5f",\n'...
'            "--qx", "%.5f",\n'...
'            "--qy", "%.5f",\n'...
'            "--qz", "%.5f",\n'...
'            "--qw", "%.5f",\n'...
'            "--frame-id",       "base_link",\n'...
'            "--child-frame-id", "global_camera_link",\n'...
'        ],\n'...
'    )\n'], ...
    t_new(1), t_new(2), t_new(3), ...
    q_new(2), q_new(3), q_new(4), q_new(1));

snippet_path = fullfile(RESULTS_DIR, 'new_calibration_tf_snippet.txt');
fid = fopen(snippet_path, 'w');
fprintf(fid, '# ── Replace the calibration_tf Node in robot_launch.py ──\n');
fprintf(fid, '# Calibration date: %s\n', datestr(now));
fprintf(fid, '# Method: %s\n', calib_result.method);
fprintf(fid, '# Mean 3D error: %.2f mm\n\n', mean(errs_3d)*1000);
fprintf(fid, '%s', launch_snippet);
fclose(fid);

fprintf('Saved launch snippet: %s\n\n', snippet_path);
fprintf('=== COPY THIS INTO robot_launch.py ===\n\n');
fprintf('%s\n', launch_snippet);

% ── Save T_ee_board for pick-and-place ────────────────────────────────────
pickplace.T_base_cam   = T_base_cam;
pickplace.T_ee_board   = T_ee_board_mean;
pickplace.K            = K;
pickplace.mean_err_mm  = mean(errs_3d)*1000;
save(fullfile(RESULTS_DIR, 'pickplace_transforms.mat'), 'pickplace');

fprintf('Saved pick-and-place transforms: %s/pickplace_transforms.mat\n', RESULTS_DIR);
fprintf('Next : run >> Step4_PickAndPlace to test with a real object!\n\n');
