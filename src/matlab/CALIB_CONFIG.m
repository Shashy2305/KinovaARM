% =========================================================================
%  CALIB_CONFIG.m  —  Hand-Eye Calibration Configuration
%  Robot  : Kinova Gen3 (7-DOF) + Robotiq 2F-140
%  Camera : OAK-D Pro Wide (fixed, global mount)
%  Target : 8x11 checkerboard (7x10 inner corners), 15 mm squares
% =========================================================================

%% ── PATHS ─────────────────────────────────────────────────────────────────
CALIB_DATA_DIR = fullfile(getenv('HOME'), 'calib_data');   % ~/calib_data/
IMAGES_DIR     = fullfile(CALIB_DATA_DIR, 'images');       % pose_001.png ...
POSES_DIR      = fullfile(CALIB_DATA_DIR, 'poses');        % pose_001.txt ...
RESULTS_DIR    = fullfile(CALIB_DATA_DIR, 'results');      % output folder

%% ── CHECKERBOARD ──────────────────────────────────────────────────────────
BOARD_INNER_ROWS  = 7;      % inner corners  (8 squares → 7 corners)
BOARD_INNER_COLS  = 10;     % inner corners  (11 squares → 10 corners)
SQUARE_SIZE_MM    = 15.0;   % physical size of each square in mm
SQUARE_SIZE_M     = SQUARE_SIZE_MM / 1000;   % in metres

%% ── OAK-D INTRINSICS @ 640×400 ───────────────────────────────────────────
% Verified from EEPROM, scaled from 3840×2160 → 640×400
IMG_W = 640;
IMG_H = 400;
FX    = 426.1166343;
FY    = 425.9295924;
CX    = 315.2250461;
CY    = 205.3090416;

K = [FX,   0,  CX;
      0,  FY,  CY;
      0,   0,   1];      % 3×3 camera intrinsic matrix

DIST_COEFFS = zeros(1, 5);   % OAK-D undistortion is done in hardware

%% ── EXISTING TF (from robot_launch.py) ───────────────────────────────────
% base_link → global_camera_link  (old/initial guess)
EXISTING_T_x   =  0.03986;
EXISTING_T_y   =  0.35130;
EXISTING_T_z   =  0.24388;
EXISTING_Q_x   =  0.20970;
EXISTING_Q_y   = -0.16902;
EXISTING_Q_z   =  0.01846;
EXISTING_Q_w   =  0.96287;

fprintf('=== Hand-Eye Calibration Config Loaded ===\n');
fprintf('Data dir    : %s\n', CALIB_DATA_DIR);
fprintf('Board       : %dx%d inner corners, %.0fmm squares\n', ...
        BOARD_INNER_ROWS, BOARD_INNER_COLS, SQUARE_SIZE_MM);
fprintf('Camera K    : FX=%.2f  FY=%.2f  CX=%.2f  CY=%.2f\n', FX, FY, CX, CY);
fprintf('==========================================\n\n');
