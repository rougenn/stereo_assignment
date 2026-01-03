import os
import re
import glob
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Input folders with pairs (same stems 1..13 etc.)
LEFT_DIR  = os.path.join(PROJECT_ROOT, "rectified_photos", "left")
RIGHT_DIR = os.path.join(PROJECT_ROOT, "rectified_photos", "right")

# Stereo calibration YAML (OpenCV FileStorage), must contain: K1,D1,K2,D2,R,T
STEREO_YAML = os.path.join(PROJECT_ROOT, "rectified_photos", "stereo_calibration_opencv.yml")

# Output
OUT_DIR = os.path.join(PROJECT_ROOT, "classic_depth_outputs")

# If your inputs are ALREADY rectified (recommended for FoundationStereo),
# you can skip rectification maps and just run SGBM.
# True  = assume images already rectified (faster)
# False = apply undistort+rectify from STEREO_YAML (safer)
INPUTS_ALREADY_RECTIFIED = True

# If INPUTS_ALREADY_RECTIFIED=False, we will resize to this (must match calibration aspect).
# If True, we keep input size unless RESIZE_ANYWAY=True.
RESIZE_ANYWAY = False
TARGET_W = 1440
TARGET_H = 1920

# ---- StereoSGBM parameters ----
# minDisparity can stay 0 for most rectified pairs
MIN_DISP = 0

# must be divisible by 16 (16, 32, 64, 128, 256, ...)
# Bigger -> can capture larger depth range, but slower and noisier.
NUM_DISP = 320

# odd number >= 3 (3..11 usually). Bigger -> smoother, less detail.
BLOCK_SIZE = 11

# penalties (SGBM uses these)
# Typical: P1 = 8*cn*block^2, P2 = 32*cn*block^2
# cn = 1 for gray, 3 for color (we will use gray)
SGBM_P1 = None  # if None -> auto
SGBM_P2 = None  # if None -> auto

UNIQUENESS_RATIO = 7
SPECKLE_WINDOW_SIZE = 200
SPECKLE_RANGE = 1
DISP12_MAX_DIFF = 1

# Post-processing
APPLY_WLS_FILTER = True  # needs opencv-contrib (ximgproc). If True and available, improves quality.

# Saving
SAVE_DISP_NPY = True
SAVE_DEPTH_NPY = True
SAVE_PREVIEW_PNG = True  # disparity/depth preview images
# =========================


def numeric_key(path: str):
    base = os.path.basename(path)
    m = re.match(r"(\d+)", base)
    return int(m.group(1)) if m else 10**9


def list_images(folder: str):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files, key=numeric_key)


def read_stereo_yaml(path: str):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open stereo yaml: {path}")

    K1 = fs.getNode("K1").mat()
    D1 = fs.getNode("D1").mat()
    K2 = fs.getNode("K2").mat()
    D2 = fs.getNode("D2").mat()
    R  = fs.getNode("R").mat()
    T  = fs.getNode("T").mat()
    # optional precomputed rectification
    R1 = fs.getNode("R1").mat()
    R2 = fs.getNode("R2").mat()
    P1 = fs.getNode("P1").mat()
    P2 = fs.getNode("P2").mat()
    Q  = fs.getNode("Q").mat()
    fs.release()

    if any(x is None for x in [K1, D1, K2, D2, R, T]) or (np.size(K1) == 0):
        raise RuntimeError("Stereo YAML must contain K1,D1,K2,D2,R,T")

    K1 = np.asarray(K1, dtype=np.float64).reshape(3, 3)
    D1 = np.asarray(D1, dtype=np.float64).reshape(1, -1)
    K2 = np.asarray(K2, dtype=np.float64).reshape(3, 3)
    D2 = np.asarray(D2, dtype=np.float64).reshape(1, -1)
    R  = np.asarray(R,  dtype=np.float64).reshape(3, 3)
    T  = np.asarray(T,  dtype=np.float64).reshape(3, 1)

    # optional rectification matrices
    R1 = None if R1 is None or np.size(R1) == 0 else np.asarray(R1, dtype=np.float64).reshape(3, 3)
    R2 = None if R2 is None or np.size(R2) == 0 else np.asarray(R2, dtype=np.float64).reshape(3, 3)
    P1 = None if P1 is None or np.size(P1) == 0 else np.asarray(P1, dtype=np.float64)
    P2 = None if P2 is None or np.size(P2) == 0 else np.asarray(P2, dtype=np.float64)
    Q  = None if Q  is None or np.size(Q)  == 0 else np.asarray(Q,  dtype=np.float64)

    baseline = float(np.linalg.norm(T))
    return K1, D1, K2, D2, R, T, baseline, R1, R2, P1, P2, Q


def make_sgbm():
    # Work in grayscale => cn=1
    cn = 1
    bs = int(BLOCK_SIZE)
    p1 = SGBM_P1 if SGBM_P1 is not None else 8 * cn * bs * bs
    p2 = SGBM_P2 if SGBM_P2 is not None else 32 * cn * bs * bs

    sgbm = cv2.StereoSGBM_create(
        minDisparity=int(MIN_DISP),
        numDisparities=int(NUM_DISP),
        blockSize=bs,
        P1=int(p1),
        P2=int(p2),
        disp12MaxDiff=int(DISP12_MAX_DIFF),
        uniquenessRatio=int(UNIQUENESS_RATIO),
        speckleWindowSize=int(SPECKLE_WINDOW_SIZE),
        speckleRange=int(SPECKLE_RANGE),
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return sgbm


def try_make_wls_filter(sgbm_left):
    if not APPLY_WLS_FILTER:
        return None, None
    try:
        import cv2.ximgproc as xip
        wls = xip.createDisparityWLSFilter(matcher_left=sgbm_left)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.5)
        # need right matcher too
        right_matcher = xip.createRightMatcher(sgbm_left)
        return wls, right_matcher
    except Exception:
        print("[WARN] WLS requested but opencv-contrib ximgproc not available. Skipping WLS.")
        return None, None


def normalize_preview(img, clip_percent=1.0):
    """Robust normalize to 8-bit for preview."""
    x = img.copy()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros_like(img, dtype=np.uint8)
    lo = np.percentile(x, clip_percent)
    hi = np.percentile(x, 100.0 - clip_percent)
    if hi <= lo:
        hi = lo + 1e-6
    y = (img - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    return (y * 255).astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    K1, D1, K2, D2, R, T, baseline, R1, R2, P1, P2, Q = read_stereo_yaml(STEREO_YAML)
    print(f"[INFO] baseline (m): {baseline}")

    left_files = list_images(LEFT_DIR)
    right_files = list_images(RIGHT_DIR)
    if not left_files or not right_files:
        raise RuntimeError("No images found in LEFT_DIR/RIGHT_DIR (expect jpg/png).")

    L = {str(numeric_key(p)): p for p in left_files}
    Rm = {str(numeric_key(p)): p for p in right_files}
    keys = sorted(set(L.keys()) & set(Rm.keys()), key=lambda s: int(s) if s.isdigit() else 10**9)
    if not keys:
        raise RuntimeError("No matching numbered pairs.")

    # Prepare SGBM
    sgbm = make_sgbm()
    wls, sgbm_right = try_make_wls_filter(sgbm)

    # If we need rectification maps, build them once using the first image size (or TARGET if resizing)
    mapLx = mapLy = mapRx = mapRy = None
    fx_for_depth = None

    # read first pair to set size
    imgL0 = cv2.imread(L[keys[0]], cv2.IMREAD_COLOR)
    imgR0 = cv2.imread(Rm[keys[0]], cv2.IMREAD_COLOR)
    if imgL0 is None or imgR0 is None:
        raise RuntimeError("Cannot read first pair (use jpg/png, HEIC not supported by cv2).")

    if RESIZE_ANYWAY or (not INPUTS_ALREADY_RECTIFIED):
        size = (TARGET_W, TARGET_H)
    else:
        size = (imgL0.shape[1], imgL0.shape[0])  # (W,H)

    if not INPUTS_ALREADY_RECTIFIED:
        # Use precomputed R1/R2/P1/P2 if present, else compute stereoRectify
        if R1 is None or R2 is None or P1 is None or P2 is None:
            flags = cv2.CALIB_ZERO_DISPARITY
            R1c, R2c, P1c, P2c, Qc, roi1c, roi2c = cv2.stereoRectify(
                K1, D1, K2, D2, size, R, T, flags=flags, alpha=0.0
            )
            R1, R2, P1, P2, Q = R1c, R2c, P1c, P2c, Qc

        mapLx, mapLy = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_16SC2)
        mapRx, mapRy = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_16SC2)

        # fx for depth should correspond to rectified projection matrix P1 (better), else K1
        fx_for_depth = float(P1[0, 0]) if P1 is not None else float(K1[0, 0])
    else:
        # Inputs already rectified: use K1 fx (or P1 if you saved it)
        fx_for_depth = float(P1[0, 0]) if P1 is not None else float(K1[0, 0])

    print(f"[INFO] Processing size: {size[0]}x{size[1]} | fx={fx_for_depth}")

    for k in keys:
        out_k = os.path.join(OUT_DIR, f"{int(k):02d}")
        os.makedirs(out_k, exist_ok=True)

        imgL = cv2.imread(L[k], cv2.IMREAD_COLOR)
        imgR = cv2.imread(Rm[k], cv2.IMREAD_COLOR)
        if imgL is None or imgR is None:
            print(f"[SKIP] cannot read {k}")
            continue

        # resize if needed
        if RESIZE_ANYWAY or (not INPUTS_ALREADY_RECTIFIED):
            imgL = cv2.resize(imgL, size, interpolation=cv2.INTER_AREA)
            imgR = cv2.resize(imgR, size, interpolation=cv2.INTER_AREA)

        # rectify if needed
        if not INPUTS_ALREADY_RECTIFIED:
            imgLr = cv2.remap(imgL, mapLx, mapLy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            imgRr = cv2.remap(imgR, mapRx, mapRy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        else:
            imgLr, imgRr = imgL, imgR

        # grayscale for matching
        gL = cv2.cvtColor(imgLr, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(imgRr, cv2.COLOR_BGR2GRAY)

        # compute disparity (fixed-point *16)
        disp_left = sgbm.compute(gL, gR)  # int16
        disp = disp_left.astype(np.float32) / 16.0

        # optional WLS refinement
        if wls is not None and sgbm_right is not None:
            disp_right = sgbm_right.compute(gR, gL)
            disp_right = disp_right.astype(np.int16)
            disp = wls.filter(disp_left, imgLr, disparity_map_right=disp_right).astype(np.float32) / 16.0

        # depth (meters): Z = fx*B/disp
        depth = np.full_like(disp, np.inf, dtype=np.float32)
        valid = disp > 0.5  # small threshold to avoid crazy depth
        depth[valid] = (fx_for_depth * baseline) / disp[valid]

        # save arrays
        if SAVE_DISP_NPY:
            np.save(os.path.join(out_k, "disp.npy"), disp)
        if SAVE_DEPTH_NPY:
            np.save(os.path.join(out_k, "depth_meter.npy"), depth)

        # previews
        if SAVE_PREVIEW_PNG:
            disp_vis8 = normalize_preview(disp)
            depth_vis8 = normalize_preview(np.clip(depth, 0, np.percentile(depth[np.isfinite(depth)], 99)) if np.isfinite(depth).any() else depth)

            cv2.imwrite(os.path.join(out_k, "disp_vis.png"), disp_vis8)
            cv2.imwrite(os.path.join(out_k, "depth_vis.png"), depth_vis8)

            # also save rectified pair for sanity
            cv2.imwrite(os.path.join(out_k, "left_rect.png"), imgLr)
            cv2.imwrite(os.path.join(out_k, "right_rect.png"), imgRr)

        print(f"[OK] {k} -> {out_k}")

    print(f"[DONE] outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
