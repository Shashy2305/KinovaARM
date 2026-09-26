% =========================================================================
%  Step2_HandEyeCalib.m
%  Solves the hand-eye calibration problem  AX = XB
%  for Eye-in-Hand (camera fixed, board on EE).
%
%  A_ij  = T_base_ee_i  * inv(T_base_ee_j)   ← relative EE motion
%  B_ij  = T_cam_board_i * inv(T_cam_board_j) ← relative board motion
%  X     = T_base_cam                          ← GOAL: camera in base frame
%
%  Methods implemented:
%    1. Tsai-Lenz  (fast, closed-form)
%    2. Park-Martin (SVD-based, often more robust)
%    3. MATLAB built-in estimateHandEye (if R2022b+)
%  Uses all three and picks the best by residual.
%
%  >> Step2_HandEyeCalib
% =========================================================================

clear; clc;
CALIB_CONFIG;

fprintf('STEP 2: Hand-Eye Calibration  (AX = XB)\n\n');

% ── Load data ─────────────────────────────────────────────────────────────
data_path = fullfile(RESULTS_DIR, 'corner_data.mat');
if ~isfile(data_path)
    error('corner_data.mat not found. Run Step1_DetectCorners first.');
end
load(data_path, 'T_cam_board', 'T_base_ee', 'n_valid');

fprintf('Loaded %d valid pose pairs.\n\n', n_valid);

% ── Build relative motion pairs (A_ij, B_ij) ──────────────────────────────
fprintf('Building %d relative motion pairs...\n', n_valid*(n_valid-1)/2);

A_list = {};   % relative EE motions
B_list = {};   % relative board-in-cam motions

for i = 1:n_valid
    for j = i+1:n_valid
        A_ij = T_base_ee{i} / T_base_ee{j};    % T_ee_i * inv(T_ee_j)
        B_ij = T_cam_board{i} / T_cam_board{j}; % T_cb_i * inv(T_cb_j)

        % Skip near-identity rotations (robot barely moved — bad conditioning)
        rot_angle_A = acos(min(1, max(-1, (trace(A_ij(1:3,1:3))-1)/2)));
        if rot_angle_A < deg2rad(5)
            continue;
        end

        A_list{end+1} = A_ij;
        B_list{end+1} = B_ij;
    end
end

n_pairs = length(A_list);
fprintf('Using %d pairs (rotation > 5°).\n\n', n_pairs);

if n_pairs < 3
    error('Too few useful pairs (%d). Ensure robot makes large orientation changes between poses.', n_pairs);
end

% =========================================================================
%  METHOD 1: Tsai-Lenz (closed-form rotation + translation)
% =========================================================================
fprintf('--- Method 1: Tsai-Lenz ---\n');
X_tsai = tsai_lenz(A_list, B_list);
res_tsai = compute_residual(X_tsai, A_list, B_list);
fprintf('Rotation residual    : %.4f rad\n', res_tsai.rot);
fprintf('Translation residual : %.4f m\n\n', res_tsai.trans);

% =========================================================================
%  METHOD 2: Park-Martin (SVD-based)
% =========================================================================
fprintf('--- Method 2: Park-Martin ---\n');
X_park = park_martin(A_list, B_list);
res_park = compute_residual(X_park, A_list, B_list);
fprintf('Rotation residual    : %.4f rad\n', res_park.rot);
fprintf('Translation residual : %.4f m\n\n', res_park.trans);

% =========================================================================
%  METHOD 3: MATLAB built-in estimateHandEye  (R2022b+)
% =========================================================================
X_matlab = [];
try
    % Prepare inputs: cell arrays of rotation matrices and translation vecs
    Ra = cellfun(@(T) T(1:3,1:3), A_list, 'UniformOutput', false);
    ta = cellfun(@(T) T(1:3,4),   A_list, 'UniformOutput', false);
    Rb = cellfun(@(T) T(1:3,1:3), B_list, 'UniformOutput', false);
    tb = cellfun(@(T) T(1:3,4),   B_list, 'UniformOutput', false);
    [R_x, t_x] = estimateHandEye(Ra, ta, Rb, tb);
    X_matlab = eye(4);
    X_matlab(1:3,1:3) = R_x;
    X_matlab(1:3,  4) = t_x;
    res_matlab = compute_residual(X_matlab, A_list, B_list);
    fprintf('--- Method 3: MATLAB estimateHandEye ---\n');
    fprintf('Rotation residual    : %.4f rad\n', res_matlab.rot);
    fprintf('Translation residual : %.4f m\n\n', res_matlab.trans);
catch
    fprintf('--- Method 3: estimateHandEye not available (needs R2022b+) ---\n\n');
    res_matlab.rot   = inf;
    res_matlab.trans = inf;
end

% ── Pick best result ───────────────────────────────────────────────────────
scores = [res_tsai.rot + res_tsai.trans, ...
          res_park.rot + res_park.trans, ...
          res_matlab.rot + res_matlab.trans];
[~, best_idx] = min(scores);

method_names = {'Tsai-Lenz', 'Park-Martin', 'estimateHandEye'};
X_list       = {X_tsai, X_park, X_matlab};
T_base_cam   = X_list{best_idx};

fprintf('=== Best result: %s ===\n\n', method_names{best_idx});

% ── Extract quaternion and translation ────────────────────────────────────
R_result = T_base_cam(1:3,1:3);
t_result = T_base_cam(1:3,  4);
q_result = rotm2quat(R_result);    % [qw qx qy qz]

fprintf('T_base_cam  (4×4):\n');
disp(T_base_cam);

fprintf('Translation  [x y z] (metres):\n');
fprintf('  x = %.6f\n  y = %.6f\n  z = %.6f\n\n', t_result(1), t_result(2), t_result(3));

fprintf('Quaternion  [qw qx qy qz]:\n');
fprintf('  qw = %.6f\n  qx = %.6f\n  qy = %.6f\n  qz = %.6f\n\n', ...
        q_result(1), q_result(2), q_result(3), q_result(4));

% ── Save ──────────────────────────────────────────────────────────────────
calib_result.T_base_cam    = T_base_cam;
calib_result.t_xyz         = t_result;
calib_result.q_wxyz        = q_result;
calib_result.method        = method_names{best_idx};
calib_result.residual_rot  = scores(best_idx);
calib_result.all_X         = struct('tsai', X_tsai, 'park', X_park);

save(fullfile(RESULTS_DIR, 'calib_result.mat'), 'calib_result');
fprintf('Saved: %s/calib_result.mat\n', RESULTS_DIR);
fprintf('Next : run >> Step3_VerifyAndExport\n\n');


% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function X = tsai_lenz(A_list, B_list)
% Tsai-Lenz closed-form hand-eye calibration.
    n = length(A_list);

    % --- Rotation ---
    M = zeros(3, 3);
    for k = 1:n
        Ra = A_list{k}(1:3,1:3);
        Rb = B_list{k}(1:3,1:3);
        aa = rotm2axang(Ra);   % [axis angle]
        ab = rotm2axang(Rb);
        % Skew-symmetric form for Tsai rotation
        alpha = aa(4);  a_hat = aa(1:3);
        beta  = ab(4);  b_hat = ab(1:3);
        a_prim = 2 * sin(alpha/2) * a_hat';
        b_prim = 2 * sin(beta/2)  * b_hat';
        M = M + b_prim * a_prim';
    end
    [U, ~, V] = svd(M);
    Rx = V * diag([1,1,det(V)*det(U)]) * U';

    % --- Translation ---
    C = zeros(3*n, 3);
    d = zeros(3*n, 1);
    for k = 1:n
        Ra = A_list{k}(1:3,1:3);
        ta = A_list{k}(1:3,4);
        tb = B_list{k}(1:3,4);
        C(3*k-2:3*k, :) = (Ra - eye(3));
        d(3*k-2:3*k)    = Rx * tb - ta;
    end
    tx = (C' * C) \ (C' * d);

    X = eye(4);
    X(1:3,1:3) = Rx;
    X(1:3,  4) = tx;
end


function X = park_martin(A_list, B_list)
% Park-Martin SVD-based hand-eye calibration.
    n = length(A_list);
    M = zeros(3,3);
    for k = 1:n
        Ra = A_list{k}(1:3,1:3);
        Rb = B_list{k}(1:3,1:3);
        M  = M + logm(Rb) * logm(Ra)';
    end
    [U,~,V] = svd(M);
    Rx = V * diag([1,1,det(V)*det(U)]) * U';

    % Translation — same as Tsai
    C = zeros(3*n, 3);
    d = zeros(3*n, 1);
    for k = 1:n
        Ra = A_list{k}(1:3,1:3);
        ta = A_list{k}(1:3,4);
        tb = B_list{k}(1:3,4);
        C(3*k-2:3*k, :) = (Ra - eye(3));
        d(3*k-2:3*k)    = Rx * tb - ta;
    end
    tx = (C' * C) \ (C' * d);

    X = eye(4);
    X(1:3,1:3) = Rx;
    X(1:3,  4) = tx;
end


function res = compute_residual(X, A_list, B_list)
% Mean rotation and translation residual over all pairs.
    n = length(A_list);
    rot_errs   = zeros(n,1);
    trans_errs = zeros(n,1);
    for k = 1:n
        AX = A_list{k} * X;
        XB = X * B_list{k};
        dR = AX(1:3,1:3)' * XB(1:3,1:3);
        rot_errs(k)   = norm(rotm2axang(dR) .* [1 1 1 1]);
        trans_errs(k) = norm(AX(1:3,4) - XB(1:3,4));
    end
    rot_errs(rot_errs > pi) = [];   % remove outliers
    res.rot   = mean(rot_errs);
    res.trans = mean(trans_errs);
end
