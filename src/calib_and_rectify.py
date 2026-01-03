import os
import re
import glob
import math
import cv2
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

LEFT_DIR  = os.path.join(PROJECT_ROOT, "left")
RIGHT_DIR = os.path.join(PROJECT_ROOT, "right")

# Output directories (rectified images)
OUT_DIR = os.path.join(PROJECT_ROOT, "rectified_photos")
OUT_LEFT_DIR  = os.path.join(OUT_DIR, "left")
OUT_RIGHT_DIR = os.path.join(OUT_DIR, "right")

# Output calibration files (OpenCV YAML)
OUT_LEFT_CALIB_YAML   = os.path.join(OUT_DIR, "left_calibration_opencv.yml")
OUT_RIGHT_CALIB_YAML  = os.path.join(OUT_DIR, "right_calibration_opencv.yml")
OUT_STEREO_CALIB_YAML = os.path.join(OUT_DIR, "stereo_calibration_opencv.yml")

# Chessboard settings (INNER corners count)
# Your previous YAML had pattern_size: [10, 7] -> inner corners = 10x7
CHESSBOARD_SIZE = (10, 7)        # (cols, rows) of INNER corners
SQUARE_SIZE_M   = 0.024          # square size in meters (change if needed)

# Target calibration/rectification resolution (portrait 3:4, multiples of 32)
TARGET_W = 1440 // 2
TARGET_H = 1920 // 2

# Detection settings
USE_SB_DETECTOR = True           # try findChessboardCornersSB first (often more robust)
SUBPIX_WIN = (11, 11)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

# Calibration flags
CALIB_FLAGS_SINGLE = 0           # keep simple; you can add flags if you want
STEREO_FLAGS = cv2.CALIB_FIX_INTRINSIC

# Rectification flags
RECTIFY_ZERO_DISPARITY = True    # good for stereo networks
RECTIFY_ALPHA = 1.0              # 0=crop black, 1=keep all FOV

# Output image format
OUT_EXT = ".png"
OUT_JPEG_QUALITY = 100

# If your right/left mapping is inverted after rectify, set this True:
INVERT_RT = False
# =========================


def ensure_dirs():
    os.makedirs(OUT_LEFT_DIR, exist_ok=True)
    os.makedirs(OUT_RIGHT_DIR, exist_ok=True)


def numeric_key(path: str):
    # expects filenames like "1.jpg", "13.HEIC"
    base = os.path.basename(path)
    m = re.match(r"(\d+)", base)
    return int(m.group(1)) if m else 10**9


def list_images(folder: str):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.heic", "*.HEIC"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    files = sorted(files, key=numeric_key)
    return files


def read_image_any(path: str):
    """
    Reads JPG/PNG via cv2, and tries HEIC via Pillow if installed.
    Returns BGR uint8 image.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    # Try Pillow for HEIC (requires: pip install pillow pillow-heif)
    try:
        from PIL import Image, ImageOps
        try:
            import pillow_heif  # noqa: F401
            pillow_heif.register_heif_opener()
        except Exception:
            # if pillow-heif not available, Image.open(HEIC) likely fails
            pass

        pil = Image.open(path)
        pil = ImageOps.exif_transpose(pil)  # fix orientation
        rgb = np.array(pil.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr
    except Exception as e:
        raise RuntimeError(
            f"Cannot read image: {path}\n"
            f"OpenCV failed and Pillow HEIC reader failed.\n"
            f"Install: pip install pillow pillow-heif\n"
            f"Original error: {e}"
        )


def center_crop_to_aspect(img, target_w, target_h):
    """Center-crop image to match target aspect ratio without stretching."""
    h, w = img.shape[:2]
    target_ar = target_w / target_h
    ar = w / h

    if abs(ar - target_ar) < 1e-6:
        return img

    if ar > target_ar:
        # too wide -> crop width
        new_w = int(h * target_ar)
        x0 = (w - new_w) // 2
        return img[:, x0:x0 + new_w]
    else:
        # too tall -> crop height
        new_h = int(w / target_ar)
        y0 = (h - new_h) // 2
        return img[y0:y0 + new_h, :]


def preprocess_to_target(img_bgr):
    """
    Resize ONLY to TARGET_W x TARGET_H.
    Assumes aspect ratio already matches (e.g. 3:4 -> 3:4).
    """
    return cv2.resize(img_bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


def find_corners(gray):
    """
    Returns (found, corners) where corners is Nx1x2 float32.
    """
    pattern = CHESSBOARD_SIZE
    if USE_SB_DETECTOR:
        # SB is more robust, but can be slower
        found, corners = cv2.findChessboardCornersSB(gray, pattern, None)
        if found:
            # corners already subpixel-ish, but we can still refine a bit
            corners = corners.astype(np.float32).reshape(-1, 1, 2)
            cv2.cornerSubPix(gray, corners, SUBPIX_WIN, (-1, -1), SUBPIX_CRITERIA)
            return True, corners

    # classic fallback
    found, corners = cv2.findChessboardCorners(
        gray, pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if found:
        corners = corners.astype(np.float32)
        cv2.cornerSubPix(gray, corners, SUBPIX_WIN, (-1, -1), SUBPIX_CRITERIA)
        return True, corners

    return False, None


def make_object_points():
    """
    Creates object points for a chessboard with known square size.
    """
    cols, rows = CHESSBOARD_SIZE
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(SQUARE_SIZE_M)
    return objp


def write_opencv_yaml(path: str, data: dict):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot write: {path}")
    for k, v in data.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            fs.write(k, float(v))
        else:
            fs.write(k, v)
    fs.release()


def reproj_error(objpoints, imgpoints, rvecs, tvecs, K, D):
    total_err = 0.0
    total_pts = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2)
        n = len(objpoints[i])
        total_err += err * err
        total_pts += n
    rmse = math.sqrt(total_err / max(1, total_pts))
    return rmse


def main():
    ensure_dirs()

    left_files = list_images(LEFT_DIR)
    right_files = list_images(RIGHT_DIR)

    if not left_files:
        raise RuntimeError(f"No images found in LEFT_DIR: {LEFT_DIR}")
    if not right_files:
        raise RuntimeError(f"No images found in RIGHT_DIR: {RIGHT_DIR}")

    # Build map by index number (1..13)
    def build_map(files):
        m = {}
        for f in files:
            k = numeric_key(f)
            m[k] = f
        return m

    L = build_map(left_files)
    R = build_map(right_files)
    keys = sorted(set(L.keys()) & set(R.keys()))
    if not keys:
        raise RuntimeError("No matching numbered pairs between left/ and right/.")

    print(f"[INFO] Found {len(keys)} candidate pairs: {keys}")

    objp = make_object_points()

    objpoints = []
    imgpoints_l = []
    imgpoints_r = []
    used_keys = []

    # Detect chessboard on resized images
    for k in keys:
        lp = L[k]
        rp = R[k]

        imgL = preprocess_to_target(read_image_any(lp))
        imgR = preprocess_to_target(read_image_any(rp))

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        foundL, cornersL = find_corners(grayL)
        foundR, cornersR = find_corners(grayR)

        if foundL and foundR:
            objpoints.append(objp.copy())
            imgpoints_l.append(cornersL)
            imgpoints_r.append(cornersR)
            used_keys.append(k)
            print(f"[OK] corners found on pair {k}")
        else:
            print(f"[SKIP] pair {k} (left_found={foundL}, right_found={foundR})")

    if len(used_keys) < 5:
        raise RuntimeError(
            f"Too few valid pairs with detected chessboard: {len(used_keys)}.\n"
            f"Need more/better frames (lighting, focus, board fully visible)."
        )

    image_size = (TARGET_W, TARGET_H)
    print(f"[INFO] Calibrating at size {image_size} using pairs: {used_keys}")

    # --------- Single camera calibration ---------
    retL, KL, DL, rvecsL, tvecsL = cv2.calibrateCamera(
        objpoints, imgpoints_l, image_size, None, None, flags=CALIB_FLAGS_SINGLE
    )
    retR, KR, DR, rvecsR, tvecsR = cv2.calibrateCamera(
        objpoints, imgpoints_r, image_size, None, None, flags=CALIB_FLAGS_SINGLE
    )

    rmseL = reproj_error(objpoints, imgpoints_l, rvecsL, tvecsL, KL, DL)
    rmseR = reproj_error(objpoints, imgpoints_r, rvecsR, tvecsR, KR, DR)

    print(f"[INFO] Left  reproj RMSE: {rmseL:.6f}")
    print(f"[INFO] Right reproj RMSE: {rmseR:.6f}")

    # --------- Stereo calibration (fix intrinsics) ---------
    stereo_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    stereo_ret, KL2, DL2, KR2, DR2, Rmat, Tvec, Emat, Fmat = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        KL, DL, KR, DR,
        image_size,
        criteria=stereo_criteria,
        flags=STEREO_FLAGS
    )

    if INVERT_RT:
        # invert transform (swap direction)
        Rmat = Rmat.T
        Tvec = -Rmat @ Tvec
        print("[INFO] INVERT_RT=True -> inverted stereo R/T")

    print(f"[INFO] Stereo RMS: {stereo_ret:.6f}")
    print(f"[INFO] T (meters): {Tvec.ravel()}  | baseline ~ {np.linalg.norm(Tvec):.6f}")

    # --------- Rectification ---------
    flags = cv2.CALIB_ZERO_DISPARITY if RECTIFY_ZERO_DISPARITY else 0

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        KL, DL, KR, DR,
        image_size,
        Rmat, Tvec,
        flags=flags,
        alpha=RECTIFY_ALPHA
    )

    mapLx, mapLy = cv2.initUndistortRectifyMap(KL, DL, R1, P1, image_size, cv2.CV_16SC2)
    mapRx, mapRy = cv2.initUndistortRectifyMap(KR, DR, R2, P2, image_size, cv2.CV_16SC2)

    # --------- Save calibration YAMLs (OpenCV style) ---------
    write_opencv_yaml(OUT_LEFT_CALIB_YAML, {
        "rms": float(retL),
        "camera_matrix": KL,
        "dist_coefs": DL,
        "pattern_size": np.array([[CHESSBOARD_SIZE[0]], [CHESSBOARD_SIZE[1]]], dtype=np.int32),
        "square_size": float(SQUARE_SIZE_M),
        "image_size": np.array([[TARGET_H], [TARGET_W]], dtype=np.int32),
        "used_samples": int(len(used_keys)),
        "reproj_rmse": float(rmseL),
    })

    write_opencv_yaml(OUT_RIGHT_CALIB_YAML, {
        "rms": float(retR),
        "camera_matrix": KR,
        "dist_coefs": DR,
        "pattern_size": np.array([[CHESSBOARD_SIZE[0]], [CHESSBOARD_SIZE[1]]], dtype=np.int32),
        "square_size": float(SQUARE_SIZE_M),
        "image_size": np.array([[TARGET_H], [TARGET_W]], dtype=np.int32),
        "used_samples": int(len(used_keys)),
        "reproj_rmse": float(rmseR),
    })

    write_opencv_yaml(OUT_STEREO_CALIB_YAML, {
        "stereo_rms": float(stereo_ret),
        "left_reproj_rmse": float(rmseL),
        "right_reproj_rmse": float(rmseR),
        "K1": KL, "D1": DL,
        "K2": KR, "D2": DR,
        "R": Rmat, "T": Tvec,
        "E": Emat, "F": Fmat,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "pattern_size": np.array([[CHESSBOARD_SIZE[0]], [CHESSBOARD_SIZE[1]]], dtype=np.int32),
        "square_size": float(SQUARE_SIZE_M),
        "image_size": np.array([[TARGET_H], [TARGET_W]], dtype=np.int32),
        "used_samples": int(len(used_keys)),
    })

    print(f"[INFO] Saved YAML:\n  {OUT_LEFT_CALIB_YAML}\n  {OUT_RIGHT_CALIB_YAML}\n  {OUT_STEREO_CALIB_YAML}")

    # --------- Rectify and save ALL pairs (even if chessboard not found) ---------
    print("[INFO] Rectifying & saving all matched pairs...")
    for k in keys:
        lp = L[k]
        rp = R[k]

        imgL = preprocess_to_target(read_image_any(lp))
        imgR = preprocess_to_target(read_image_any(rp))

        rectL = cv2.remap(imgL, mapLx, mapLy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rectR = cv2.remap(imgR, mapRx, mapRy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        outL = os.path.join(OUT_LEFT_DIR, f"{k}{OUT_EXT}")
        outR = os.path.join(OUT_RIGHT_DIR, f"{k}{OUT_EXT}")

        if OUT_EXT.lower() in [".jpg", ".jpeg"]:
            cv2.imwrite(outL, rectL, [int(cv2.IMWRITE_JPEG_QUALITY), int(OUT_JPEG_QUALITY)])
            cv2.imwrite(outR, rectR, [int(cv2.IMWRITE_JPEG_QUALITY), int(OUT_JPEG_QUALITY)])
        else:
            cv2.imwrite(outL, rectL)
            cv2.imwrite(outR, rectR)

    print("[DONE] Rectified photos saved:")
    print(f"  Left : {OUT_LEFT_DIR}")
    print(f"  Right: {OUT_RIGHT_DIR}")
    print("[TIP] Проверь любую пару: точки/края в левом и правом должны быть примерно на одной горизонтали (одинаковая y).")


if __name__ == "__main__":
    main()
