# stereo_rectify_module.py
from __future__ import annotations

import os
import re
import glob
import math
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np


def _numeric_key(path: str) -> int:
    base = os.path.basename(path)
    m = re.match(r"(\d+)", base)
    return int(m.group(1)) if m else 10**9


def _list_images(folder: str) -> List[str]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.heic", "*.HEIC"]
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files, key=_numeric_key)


def _read_image_any(path: str) -> np.ndarray:
    """
    Reads JPG/PNG via OpenCV, and tries HEIC via Pillow if installed.
    Returns BGR uint8 image.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img

    try:
        from PIL import Image, ImageOps
        try:
            import pillow_heif  # noqa: F401
            pillow_heif.register_heif_opener()
        except Exception:
            pass

        pil = Image.open(path)
        pil = ImageOps.exif_transpose(pil)  # fix orientation
        rgb = np.array(pil.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr
    except Exception as e:
        raise RuntimeError(
            f"Cannot read image: {path}\n"
            f"Original error: {e}"
        )


def _resize_to_target(img_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    return cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _aspect_rel_error(w: int, h: int, target_w: int, target_h: int) -> float:
    ar_img = w / h
    ar_tgt = target_w / target_h
    return abs(ar_img - ar_tgt) / max(1e-12, ar_tgt)


def _assert_aspect_ratio(
    img_bgr: np.ndarray,
    target_w: int,
    target_h: int,
    *,
    tol: float,
    path_hint: str = ""
) -> None:
    h, w = img_bgr.shape[:2]
    rel = _aspect_rel_error(w, h, target_w, target_h)
    if rel > tol:
        ar_img = w / h
        ar_tgt = target_w / target_h

        # если фиксируем target_w, то правильная высота:
        suggested_h = int(round(target_w / ar_img))
        # если фиксируем target_h, то правильная ширина:
        suggested_w = int(round(target_h * ar_img))

        raise ValueError(
            "Aspect ratio mismatch (crop is disabled; must be strictly proportional).\n"
            f"File: {path_hint}\n"
            f"  source (w,h)=({w},{h}) ar={ar_img:.6f}\n"
            f"  target (w,h)=({target_w},{target_h}) ar={ar_tgt:.6f}\n"
            f"  relative_error={rel:.6f} > tol={tol}\n"
            "Fix by choosing proportional target_size, e.g.:\n"
            f"  - keep target_w={target_w} -> set target_h≈{suggested_h}\n"
            f"  - keep target_h={target_h} -> set target_w≈{suggested_w}\n"
        )


def _make_object_points(chessboard_size: Tuple[int, int], square_size_m: float) -> np.ndarray:
    cols, rows = chessboard_size  # inner corners
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_m)
    return objp


def _find_corners(
    gray: np.ndarray,
    chessboard_size: Tuple[int, int],
    *,
    use_sb_detector: bool,
    subpix_win: Tuple[int, int],
    subpix_criteria: Tuple[int, int, float],
) -> Tuple[bool, Optional[np.ndarray]]:
    pattern = chessboard_size

    if use_sb_detector:
        found, corners = cv2.findChessboardCornersSB(gray, pattern, None)
        if found:
            corners = corners.astype(np.float32).reshape(-1, 1, 2)
            cv2.cornerSubPix(gray, corners, subpix_win, (-1, -1), subpix_criteria)
            return True, corners

    found, corners = cv2.findChessboardCorners(
        gray, pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if found:
        corners = corners.astype(np.float32)
        cv2.cornerSubPix(gray, corners, subpix_win, (-1, -1), subpix_criteria)
        return True, corners

    return False, None


def _write_opencv_yaml(path: str, data: Dict[str, Any]) -> None:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot write: {path}")
    for k, v in data.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            fs.write(k, float(v))
        else:
            fs.write(k, v)
    fs.release()


def _reproj_rmse(objpoints, imgpoints, rvecs, tvecs, K, D) -> float:
    total_err = 0.0
    total_pts = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2)
        n = len(objpoints[i])
        total_err += err * err
        total_pts += n
    return math.sqrt(total_err / max(1, total_pts))


def stereo_calibrate_and_rectify(
    *,
    left_dir: str,
    right_dir: str,
    out_dir: str,
    chessboard_size: Tuple[int, int] = (10, 7),     # (cols, rows) inner corners
    square_size_m: float = 0.024,
    target_size: Tuple[int, int] = (1440, 1920),    # (W, H) MUST be proportional
    aspect_tol: float = 1e-3,

    use_sb_detector: bool = True,
    subpix_win: Tuple[int, int] = (11, 11),
    subpix_criteria: Tuple[int, int, float] = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4),

    calib_flags_single: int = 0,
    stereo_flags: int = cv2.CALIB_FIX_INTRINSIC,
    invert_rt: bool = False,

    rectify_zero_disparity: bool = True,
    rectify_alpha: float = 1.0,

    out_ext: str = ".png",
    out_jpeg_quality: int = 100,

    save_yaml: bool = True,
    left_yaml_name: str = "left_calibration_opencv.yml",
    right_yaml_name: str = "right_calibration_opencv.yml",
    stereo_yaml_name: str = "stereo_calibration_opencv.yml",

    min_valid_pairs: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Калибрует стерео по шахматке и сохраняет rectified пары.
    """
    target_w, target_h = int(target_size[0]), int(target_size[1])
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"Invalid target_size={target_size}")

    os.makedirs(out_dir, exist_ok=True)
    out_left_dir = os.path.join(out_dir, "left")
    out_right_dir = os.path.join(out_dir, "right")
    os.makedirs(out_left_dir, exist_ok=True)
    os.makedirs(out_right_dir, exist_ok=True)

    left_files = _list_images(left_dir)
    right_files = _list_images(right_dir)

    if not left_files:
        raise RuntimeError(f"No images found in left_dir: {left_dir}")
    if not right_files:
        raise RuntimeError(f"No images found in right_dir: {right_dir}")

    def build_map(files: List[str]) -> Dict[int, str]:
        m: Dict[int, str] = {}
        for f in files:
            m[_numeric_key(f)] = f
        return m

    L = build_map(left_files)
    R = build_map(right_files)
    keys = sorted(set(L.keys()) & set(R.keys()))
    if not keys:
        raise RuntimeError("No matching numbered pairs between left_dir and right_dir.")

    if verbose:
        print(f"[INFO] Found {len(keys)} candidate pairs: {keys}")

    # preprocess with strict AR check
    def preprocess(path: str) -> np.ndarray:
        img = _read_image_any(path)
        _assert_aspect_ratio(img, target_w, target_h, tol=aspect_tol, path_hint=path)
        return _resize_to_target(img, target_w, target_h)

    objp = _make_object_points(chessboard_size, square_size_m)
    objpoints = []
    imgpoints_l = []
    imgpoints_r = []
    used_keys = []

    # Detect corners
    for k in keys:
        imgL = preprocess(L[k])
        imgR = preprocess(R[k])

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        foundL, cornersL = _find_corners(
            grayL, chessboard_size,
            use_sb_detector=use_sb_detector,
            subpix_win=subpix_win,
            subpix_criteria=subpix_criteria,
        )
        foundR, cornersR = _find_corners(
            grayR, chessboard_size,
            use_sb_detector=use_sb_detector,
            subpix_win=subpix_win,
            subpix_criteria=subpix_criteria,
        )

        if foundL and foundR:
            objpoints.append(objp.copy())
            imgpoints_l.append(cornersL)
            imgpoints_r.append(cornersR)
            used_keys.append(k)
            if verbose:
                print(f"[OK] corners found on pair {k}")
        else:
            if verbose:
                print(f"[SKIP] pair {k} (left_found={foundL}, right_found={foundR})")

    if len(used_keys) < int(min_valid_pairs):
        raise RuntimeError(
            f"Too few valid pairs with detected chessboard: {len(used_keys)}.\n"
            f"Need at least {min_valid_pairs}.\n"
            f"TIP: improve lighting/focus, ensure board fully visible, add more frames."
        )

    image_size = (target_w, target_h)  # OpenCV expects (W,H)
    if verbose:
        print(f"[INFO] Calibrating at size {image_size} using pairs: {used_keys}")

    # Single camera calibration
    retL, KL, DL, rvecsL, tvecsL = cv2.calibrateCamera(
        objpoints, imgpoints_l, image_size, None, None, flags=calib_flags_single
    )
    retR, KR, DR, rvecsR, tvecsR = cv2.calibrateCamera(
        objpoints, imgpoints_r, image_size, None, None, flags=calib_flags_single
    )

    rmseL = _reproj_rmse(objpoints, imgpoints_l, rvecsL, tvecsL, KL, DL)
    rmseR = _reproj_rmse(objpoints, imgpoints_r, rvecsR, tvecsR, KR, DR)

    if verbose:
        print(f"[INFO] Left  reproj RMSE: {rmseL:.6f}")
        print(f"[INFO] Right reproj RMSE: {rmseR:.6f}")

    # Stereo calibration (fix intrinsics)
    stereo_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    stereo_ret, _, _, _, _, Rmat, Tvec, Emat, Fmat = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        KL, DL, KR, DR,
        image_size,
        criteria=stereo_criteria,
        flags=stereo_flags
    )

    if invert_rt:
        Rmat = Rmat.T
        Tvec = -Rmat @ Tvec
        if verbose:
            print("[INFO] invert_rt=True -> inverted stereo R/T")

    if verbose:
        print(f"[INFO] Stereo RMS: {stereo_ret:.6f}")
        print(f"[INFO] T (meters): {Tvec.ravel()}  | baseline ~ {np.linalg.norm(Tvec):.6f}")

    flags = cv2.CALIB_ZERO_DISPARITY if rectify_zero_disparity else 0
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        KL, DL, KR, DR,
        image_size,
        Rmat, Tvec,
        flags=flags,
        alpha=float(rectify_alpha)
    )

    mapLx, mapLy = cv2.initUndistortRectifyMap(KL, DL, R1, P1, image_size, cv2.CV_16SC2)
    mapRx, mapRy = cv2.initUndistortRectifyMap(KR, DR, R2, P2, image_size, cv2.CV_16SC2)

    left_yaml = os.path.join(out_dir, left_yaml_name)
    right_yaml = os.path.join(out_dir, right_yaml_name)
    stereo_yaml = os.path.join(out_dir, stereo_yaml_name)

    if save_yaml:
        _write_opencv_yaml(left_yaml, {
            "rms": float(retL),
            "camera_matrix": KL,
            "dist_coefs": DL,
            "pattern_size": np.array([[chessboard_size[0]], [chessboard_size[1]]], dtype=np.int32),
            "square_size": float(square_size_m),
            "image_width": int(target_w),
            "image_height": int(target_h),
            "used_samples": int(len(used_keys)),
            "reproj_rmse": float(rmseL),
        })

        _write_opencv_yaml(right_yaml, {
            "rms": float(retR),
            "camera_matrix": KR,
            "dist_coefs": DR,
            "pattern_size": np.array([[chessboard_size[0]], [chessboard_size[1]]], dtype=np.int32),
            "square_size": float(square_size_m),
            "image_width": int(target_w),
            "image_height": int(target_h),
            "used_samples": int(len(used_keys)),
            "reproj_rmse": float(rmseR),
        })

        _write_opencv_yaml(stereo_yaml, {
            "stereo_rms": float(stereo_ret),
            "left_reproj_rmse": float(rmseL),
            "right_reproj_rmse": float(rmseR),
            "K1": KL, "D1": DL,
            "K2": KR, "D2": DR,
            "R": Rmat, "T": Tvec,
            "E": Emat, "F": Fmat,
            "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
            "pattern_size": np.array([[chessboard_size[0]], [chessboard_size[1]]], dtype=np.int32),
            "square_size": float(square_size_m),
            "image_width": int(target_w),
            "image_height": int(target_h),
            "used_samples": int(len(used_keys)),
        })

        if verbose:
            print(f"[INFO] Saved YAML:\n  {left_yaml}\n  {right_yaml}\n  {stereo_yaml}")

    # Rectify & save all matched pairs (AR already checked inside preprocess)
    if verbose:
        print("[INFO] Rectifying & saving all matched pairs...")

    out_ext_l = out_ext.lower()
    for k in keys:
        imgL = preprocess(L[k])
        imgR = preprocess(R[k])

        rectL = cv2.remap(imgL, mapLx, mapLy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        rectR = cv2.remap(imgR, mapRx, mapRy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        outL = os.path.join(out_left_dir, f"{k}{out_ext}")
        outR = os.path.join(out_right_dir, f"{k}{out_ext}")

        if out_ext_l in [".jpg", ".jpeg"]:
            cv2.imwrite(outL, rectL, [int(cv2.IMWRITE_JPEG_QUALITY), int(out_jpeg_quality)])
            cv2.imwrite(outR, rectR, [int(cv2.IMWRITE_JPEG_QUALITY), int(out_jpeg_quality)])
        else:
            cv2.imwrite(outL, rectL)
            cv2.imwrite(outR, rectR)

    if verbose:
        print("[DONE] Rectified photos saved:")
        print(f"  Left : {out_left_dir}")
        print(f"  Right: {out_right_dir}")

    return {
        "used_keys": used_keys,
        "all_keys": keys,
        "target_size": (target_w, target_h),
        "chessboard_size": chessboard_size,
        "square_size_m": float(square_size_m),

        "retL": float(retL),
        "retR": float(retR),
        "rmseL": float(rmseL),
        "rmseR": float(rmseR),
        "stereo_rms": float(stereo_ret),

        "K1": KL, "D1": DL,
        "K2": KR, "D2": DR,
        "R": Rmat, "T": Tvec,
        "E": Emat, "F": Fmat,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,

        "out_dir": out_dir,
        "out_left_dir": out_left_dir,
        "out_right_dir": out_right_dir,
        "left_yaml": left_yaml if save_yaml else None,
        "right_yaml": right_yaml if save_yaml else None,
        "stereo_yaml": stereo_yaml if save_yaml else None,
    }
