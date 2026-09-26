% =========================================================================
%  Step1_DetectCorners.m
%  Detects 7×10 inner checkerboard corners in every saved image.
%  Computes T_cam_to_board for each valid pose.
%  Saves: corner_data.mat
%
%  Run after collect_calib_data.py has finished.
%  >> Step1_DetectCorners
% =========================================================================

clear; clc; close all;
CALIB_CONFIG;       % load shared config

fprintf('STEP 1: Detecting Checkerboard Corners\n');
fprintf('Board pattern: %d x %d inner corners\n', BOARD_INNER_ROWS, BOARD_INNER_COLS);
fprintf('Square size  : %.1f mm\n\n', SQUARE_SIZE_MM);

% ── Collect image files ───────────────────────────────────────────────────
img_files = dir(fullfile(IMAGES_DIR, 'pose_*.png'));
if isempty(img_files)
    error('No images found in %s\nRun collect_calib_data.py first.', IMAGES_DIR);
end
n_images = length(img_files);
fprintf('Found %d images.\n\n', n_images);

% ── 3D object points for checkerboard (Z=0 plane) ─────────────────────────
% Pattern: BOARD_INNER_COLS columns, BOARD_INNER_ROWS rows
[cols_grid, rows_grid] = meshgrid(0:BOARD_INNER_COLS-1, 0:BOARD_INNER_ROWS-1);
obj_pts_3d = [cols_grid(:), rows_grid(:), zeros(BOARD_INNER_ROWS*BOARD_INNER_COLS, 1)] ...
             * SQUARE_SIZE_M;    % N×3 in metres

% ── Storage ──────────────────────────────────────────────────────────────
T_cam_board_all  = cell(n_images, 1);   % 4×4 homogeneous cam→board
T_base_ee_all    = cell(n_images, 1);   % 4×4 homogeneous base→EE
valid_mask       = false(n_images, 1);
reproj_errors    = zeros(n_images, 1);

% ── Process each image ────────────────────────────────────────────────────
fig = figure('Name', 'Corner Detection', 'Position', [100 100 900 550]);

for i = 1:n_images
    fname    = img_files(i).name;
    tag      = fname(1:end-4);           % 'pose_001'
    img_path = fullfile(IMAGES_DIR, fname);
    pose_path = fullfile(POSES_DIR, [tag '.txt']);

    % Load image
    img  = imread(img_path);
    gray = rgb2gray(img);

    % Load robot EE pose
    if ~isfile(pose_path)
        fprintf('[%d/%d] %s — SKIPPED (no pose file)\n', i, n_images, tag);
        continue;
    end
    T_base_ee = readmatrix(pose_path, 'CommentStyle', '#');
    % Handle header line that readmatrix might include
    if size(T_base_ee, 1) > 4
        T_base_ee = T_base_ee(end-3:end, :);
    end

    % Detect corners
    [corners, board_size] = detectCheckerboardPoints(gray);
    expected_board = [BOARD_INNER_ROWS, BOARD_INNER_COLS];

    if isempty(corners) || ~isequal(board_size, expected_board)
        fprintf('[%d/%d] %s — board NOT detected (got %dx%d, need %dx%d)\n', ...
            i, n_images, tag, board_size(1), board_size(2), ...
            expected_board(1), expected_board(2));
        continue;
    end

    % Estimate board pose via PnP (solvePnP equivalent in MATLAB)
    % undistort corners if needed (OAK-D does hardware undistortion → skip)
    image_pts = corners;   % N×2  [x, y] pixel coords

    % Use estimateWorldCameraPose (MATLAB Computer Vision Toolbox)
    cam_params = cameraIntrinsics([FX, FY], [CX, CY], [IMG_H, IMG_W]);

    % obj_pts for MATLAB: must be N×3, and MATLAB uses row vectors
    try
        [R_board, t_board, reproj_err] = estimateWorldCameraPose( ...
            image_pts, obj_pts_3d, cam_params, ...
            'MaxReprojectionError', 5.0);
    catch ME
        fprintf('[%d/%d] %s — PnP failed: %s\n', i, n_images, tag, ME.message);
        continue;
    end

    % Build T_cam_board  (camera → board)
    T_cam_board          = eye(4);
    T_cam_board(1:3,1:3) = R_board';      % estimateWorldCameraPose gives world→cam
    T_cam_board(1:3,  4) = -R_board' * t_board';

    % Store
    T_cam_board_all{i}  = T_cam_board;
    T_base_ee_all{i}    = T_base_ee;
    valid_mask(i)        = true;
    reproj_errors(i)     = reproj_err;

    % Visualise
    imshow(img, 'Parent', gca); hold on;
    plot(image_pts(:,1), image_pts(:,2), 'r+', 'MarkerSize', 6, 'LineWidth', 1.5);
    title(sprintf('[%d/%d] %s — reproj err = %.2f px', i, n_images, tag, reproj_err));
    drawnow;

    fprintf('[%d/%d] %s — OK  reproj=%.2f px  t_board=[%.3f %.3f %.3f] m\n', ...
        i, n_images, tag, reproj_err, T_cam_board(1,4), T_cam_board(2,4), T_cam_board(3,4));
end

close(fig);

% ── Summary ───────────────────────────────────────────────────────────────
n_valid = sum(valid_mask);
fprintf('\n----- Detection Summary -----\n');
fprintf('Valid poses   : %d / %d\n', n_valid, n_images);
fprintf('Mean reproj   : %.2f px\n', mean(reproj_errors(valid_mask)));
fprintf('Max  reproj   : %.2f px\n', max(reproj_errors(valid_mask)));

if n_valid < 10
    warning('Only %d valid poses! Need ≥ 10 (ideally 15-20) for good calibration.', n_valid);
end

% ── Filter and save ───────────────────────────────────────────────────────
T_cam_board = T_cam_board_all(valid_mask);
T_base_ee   = T_base_ee_all(valid_mask);
reproj_err_valid = reproj_errors(valid_mask);

os.makedirs(RESULTS_DIR, exist_ok=True)  % will use MATLAB mkdir below
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

save_path = fullfile(RESULTS_DIR, 'corner_data.mat');
save(save_path, 'T_cam_board', 'T_base_ee', 'reproj_err_valid', ...
     'n_valid', 'BOARD_INNER_ROWS', 'BOARD_INNER_COLS', 'SQUARE_SIZE_M', 'K');

fprintf('\nSaved: %s\n', save_path);
fprintf('Next  : run >> Step2_HandEyeCalib\n\n');
