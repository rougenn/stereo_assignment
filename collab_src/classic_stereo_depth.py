# classic_stereo_depth.py
from __future__ import annotations

import os
import re
import glob
from typing import Dict, Any, Tuple, List, Optional

import cv2
import numpy as np


def _numeric_key(path: str) -> int:
    base = os.path.basename(path)
    m = re.match(r"(\d+)", base)
    return int(m.group(1)) if m else 10**9


def _list_images(folder: str) -> List[str]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files, key=_numeric_key)


def _aspect_rel_error(w: int, h: int, target_w: int, target_h: int) -> float:
    ar_img = w / h
    ar_tgt = target_w / target_h
    return abs(ar_img - ar_tgt) / max(1e-12, ar_tgt)


def _assert_aspect_ratio(img_bgr: np.ndarray, calib_size: Tuple[int, int], *, tol: float, path_hint: str) -> None:
    target_w, target_h = int(calib_size[0]), int(calib_size[1])
    h, w = img_bgr.shape[:2]
    rel = _aspect_rel_error(w, h, target_w, target_h)
    if rel > tol:
        ar_img = w / h
        ar_tgt = target_w / target_h
        suggested_h = int(round(target_w / ar_img))
        suggested_w = int(round(target_h * ar_img))
        raise ValueError(
            "Aspect ratio mismatch (crop is disabled; images must be proportional to calib_size).\n"
            f"File: {path_hint}\n"
            f"  image (w,h)=({w},{h}) ar={ar_img:.6f}\n"
            f"  calib (w,h)=({target_w},{target_h}) ar={ar_tgt:.6f}\n"
            f"  relative_error={rel:.6f} > tol={tol}\n"
            "Fix by choosing proportional calib_size or using matching input images.\n"
            f"Hint (if you want to keep calib_w={target_w}): calib_h≈{suggested_h}\n"
            f"Hint (if you want to keep calib_h={target_h}): calib_w≈{suggested_w}\n"
        )


def _read_stereo_yaml(path: str):
    """
    OpenCV FileStorage YAML must contain: K1,D1,K2,D2,R,T.
    Optionally contains: R1,R2,P1,P2,Q (precomputed rectification).
    """
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open stereo yaml: {path}")

    K1 = fs.getNode("K1").mat()
    D1 = fs.getNode("D1").mat()
    K2 = fs.getNode("K2").mat()
    D2 = fs.getNode("D2").mat()
    R  = fs.getNode("R").mat()
    T  = fs.getNode("T").mat()

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

    R1 = None if (R1 is None or np.size(R1) == 0) else np.asarray(R1, dtype=np.float64).reshape(3, 3)
    R2 = None if (R2 is None or np.size(R2) == 0) else np.asarray(R2, dtype=np.float64).reshape(3, 3)
    P1 = None if (P1 is None or np.size(P1) == 0) else np.asarray(P1, dtype=np.float64)
    P2 = None if (P2 is None or np.size(P2) == 0) else np.asarray(P2, dtype=np.float64)
    Q  = None if (Q  is None or np.size(Q)  == 0) else np.asarray(Q,  dtype=np.float64)

    baseline = float(np.linalg.norm(T))
    return K1, D1, K2, D2, R, T, baseline, R1, R2, P1, P2, Q


def _normalize_preview(img: np.ndarray, clip_percent: float = 1.0) -> np.ndarray:
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


def _make_sgbm(
    *,
    min_disp: int,
    num_disp: int,
    block_size: int,
    p1: Optional[int],
    p2: Optional[int],
    uniqueness_ratio: int,
    speckle_window_size: int,
    speckle_range: int,
    disp12_max_diff: int,
):
    if num_disp % 16 != 0:
        raise ValueError(f"num_disp must be divisible by 16, got {num_disp}")
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError(f"block_size must be odd and >=3, got {block_size}")

    cn = 1  # grayscale
    bs = int(block_size)
    P1 = int(p1) if p1 is not None else int(8 * cn * bs * bs)
    P2 = int(p2) if p2 is not None else int(32 * cn * bs * bs)

    return cv2.StereoSGBM_create(
        minDisparity=int(min_disp),
        numDisparities=int(num_disp),
        blockSize=bs,
        P1=P1,
        P2=P2,
        disp12MaxDiff=int(disp12_max_diff),
        uniquenessRatio=int(uniqueness_ratio),
        speckleWindowSize=int(speckle_window_size),
        speckleRange=int(speckle_range),
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def _try_make_wls_filter(sgbm_left):
    """
    Requires opencv-contrib-python / opencv-contrib-python-headless.
    If unavailable -> returns (None, None) with warning.
    """
    try:
        import cv2.ximgproc as xip
        wls = xip.createDisparityWLSFilter(matcher_left=sgbm_left)
        wls.setLambda(8000.0)
        wls.setSigmaColor(1.5)
        right_matcher = xip.createRightMatcher(sgbm_left)
        return wls, right_matcher
    except Exception:
        return None, None


def classic_stereo_depth(
    *,
    left_dir: str,
    right_dir: str,
    stereo_yaml_path: str,
    calib_size: Tuple[int, int],          # (W,H) where you calibrated / rectified
    out_dir: str = "classic_depth_outputs",

    # If your inputs already rectified (e.g. rectified_photos), keep True (faster)
    inputs_already_rectified: bool = True,

    # If True: always resize input images to calib_size before SGBM (recommended for reproducibility)
    resize_to_calib: bool = True,

    # strict aspect ratio check (no crop)
    aspect_tol: float = 1e-3,

    # SGBM params (defaults = ваши)
    min_disp: int = 0,
    num_disp: int = 320,
    block_size: int = 11,
    sgbm_p1: Optional[int] = None,
    sgbm_p2: Optional[int] = None,
    uniqueness_ratio: int = 7,
    speckle_window_size: int = 200,
    speckle_range: int = 1,
    disp12_max_diff: int = 1,

    # Postprocessing
    apply_wls_filter: bool = True,

    # Saving
    save_disp_npy: bool = True,
    save_depth_npy: bool = True,
    save_preview_png: bool = True,

    # Depth validity threshold
    min_valid_disp: float = 0.5,

    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Classic stereo depth from rectified pairs (or raw pairs + rectify from YAML).
    Expects numbered pairs in left_dir/right_dir: 1.png, 2.png, ...
    Saves per-pair folder: out_dir/01, out_dir/02, ...

    You provide ONLY:
      - left_dir, right_dir
      - calib_size=(W,H) used in calibration
      - stereo_yaml_path
      - (optionally) out_dir

    No crop. If aspect != calib aspect => ValueError.
    """
    calib_w, calib_h = int(calib_size[0]), int(calib_size[1])
    if calib_w <= 0 or calib_h <= 0:
        raise ValueError(f"Invalid calib_size={calib_size}")

    os.makedirs(out_dir, exist_ok=True)

    # Read stereo calibration
    K1, D1, K2, D2, R, T, baseline, R1, R2, P1, P2, Q = _read_stereo_yaml(stereo_yaml_path)
    if verbose:
        print(f"[INFO] baseline (m): {baseline:.6f}")

    # List inputs
    left_files = _list_images(left_dir)
    right_files = _list_images(right_dir)
    if not left_files or not right_files:
        raise RuntimeError("No images found in left_dir/right_dir (expect jpg/png).")

    L = {str(_numeric_key(p)): p for p in left_files}
    Rm = {str(_numeric_key(p)): p for p in right_files}
    keys = sorted(set(L.keys()) & set(Rm.keys()), key=lambda s: int(s) if s.isdigit() else 10**9)
    if not keys:
        raise RuntimeError("No matching numbered pairs between left_dir and right_dir.")
    if verbose:
        print(f"[INFO] Found {len(keys)} pairs: {keys}")

    # Read first pair to validate aspect ratio & to know native size
    imgL0 = cv2.imread(L[keys[0]], cv2.IMREAD_COLOR)
    imgR0 = cv2.imread(Rm[keys[0]], cv2.IMREAD_COLOR)
    if imgL0 is None or imgR0 is None:
        raise RuntimeError("Cannot read first pair (only jpg/png supported here).")

    _assert_aspect_ratio(imgL0, (calib_w, calib_h), tol=aspect_tol, path_hint=L[keys[0]])
    _assert_aspect_ratio(imgR0, (calib_w, calib_h), tol=aspect_tol, path_hint=Rm[keys[0]])

    # Processing size for SGBM
    if resize_to_calib:
        size = (calib_w, calib_h)
    else:
        # use input size, but still require same aspect
        size = (imgL0.shape[1], imgL0.shape[0])

    # Prepare rectification maps if needed
    mapLx = mapLy = mapRx = mapRy = None
    if not inputs_already_rectified:
        # If YAML has R1/R2/P1/P2 use them; else compute for calib_size
        if R1 is None or R2 is None or P1 is None or P2 is None:
            flags = cv2.CALIB_ZERO_DISPARITY
            R1c, R2c, P1c, P2c, Qc, _, _ = cv2.stereoRectify(
                K1, D1, K2, D2,
                (calib_w, calib_h),
                R, T,
                flags=flags,
                alpha=0.0
            )
            R1, R2, P1, P2, Q = R1c, R2c, P1c, P2c, Qc

        mapLx, mapLy = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (calib_w, calib_h), cv2.CV_16SC2)
        mapRx, mapRy = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (calib_w, calib_h), cv2.CV_16SC2)

    # fx for depth
    fx_for_depth = float(P1[0, 0]) if (P1 is not None and np.size(P1) > 0) else float(K1[0, 0])
    if verbose:
        print(f"[INFO] Processing size: {size[0]}x{size[1]} | fx={fx_for_depth:.3f} | already_rectified={inputs_already_rectified}")

    # Prepare SGBM + optional WLS
    sgbm = _make_sgbm(
        min_disp=min_disp,
        num_disp=num_disp,
        block_size=block_size,
        p1=sgbm_p1,
        p2=sgbm_p2,
        uniqueness_ratio=uniqueness_ratio,
        speckle_window_size=speckle_window_size,
        speckle_range=speckle_range,
        disp12_max_diff=disp12_max_diff,
    )

    wls = sgbm_right = None
    if apply_wls_filter:
        wls, sgbm_right = _try_make_wls_filter(sgbm)
        if (wls is None or sgbm_right is None) and verbose:
            print("[WARN] WLS requested but ximgproc not available. Install opencv-contrib to enable WLS.")

    processed = []
    skipped = []

    for k in keys:
        out_k = os.path.join(out_dir, f"{int(k):02d}")
        os.makedirs(out_k, exist_ok=True)

        imgL = cv2.imread(L[k], cv2.IMREAD_COLOR)
        imgR = cv2.imread(Rm[k], cv2.IMREAD_COLOR)
        if imgL is None or imgR is None:
            if verbose:
                print(f"[SKIP] cannot read pair {k}")
            skipped.append(k)
            continue

        # strict aspect check (no crop)
        _assert_aspect_ratio(imgL, (calib_w, calib_h), tol=aspect_tol, path_hint=L[k])
        _assert_aspect_ratio(imgR, (calib_w, calib_h), tol=aspect_tol, path_hint=Rm[k])

        # resize to chosen processing size
        if resize_to_calib or (not inputs_already_rectified):
            imgL = cv2.resize(imgL, (calib_w, calib_h), interpolation=cv2.INTER_AREA)
            imgR = cv2.resize(imgR, (calib_w, calib_h), interpolation=cv2.INTER_AREA)

        # rectify if needed
        if not inputs_already_rectified:
            imgLr = cv2.remap(imgL, mapLx, mapLy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            imgRr = cv2.remap(imgR, mapRx, mapRy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        else:
            imgLr, imgRr = imgL, imgR

        # grayscale
        gL = cv2.cvtColor(imgLr, cv2.COLOR_BGR2GRAY)
        gR = cv2.cvtColor(imgRr, cv2.COLOR_BGR2GRAY)

        # disparity (fixed-point *16)
        disp_left_fp16 = sgbm.compute(gL, gR)  # int16
        disp = disp_left_fp16.astype(np.float32) / 16.0

        # optional WLS
        if wls is not None and sgbm_right is not None:
            disp_right_fp16 = sgbm_right.compute(gR, gL).astype(np.int16)
            disp = wls.filter(disp_left_fp16, imgLr, disparity_map_right=disp_right_fp16).astype(np.float32) / 16.0

        # depth: Z = fx*B/disp
        depth = np.full_like(disp, np.inf, dtype=np.float32)
        valid = disp > float(min_valid_disp)
        depth[valid] = (fx_for_depth * baseline) / disp[valid]

        # save arrays
        if save_disp_npy:
            np.save(os.path.join(out_k, "disp.npy"), disp)
        if save_depth_npy:
            np.save(os.path.join(out_k, "depth_meter.npy"), depth)

        # previews + rectified images for sanity
        if save_preview_png:
            disp_vis8 = _normalize_preview(disp)
            if np.isfinite(depth).any():
                finite_depth = depth[np.isfinite(depth)]
                clip_max = float(np.percentile(finite_depth, 99))
                depth_vis8 = _normalize_preview(np.clip(depth, 0, clip_max))
            else:
                depth_vis8 = _normalize_preview(depth)

            cv2.imwrite(os.path.join(out_k, "disp_vis.png"), disp_vis8)
            cv2.imwrite(os.path.join(out_k, "depth_vis.png"), depth_vis8)
            cv2.imwrite(os.path.join(out_k, "left_rect.png"), imgLr)
            cv2.imwrite(os.path.join(out_k, "right_rect.png"), imgRr)

        processed.append(k)
        if verbose:
            print(f"[OK] {k} -> {out_k}")

    if verbose:
        print(f"[DONE] outputs: {out_dir}")
        print(f"[INFO] processed={len(processed)} skipped={len(skipped)}")

    return {
        "out_dir": out_dir,
        "processed_keys": processed,
        "skipped_keys": skipped,
        "baseline_m": float(baseline),
        "fx": float(fx_for_depth),
        "calib_size": (calib_w, calib_h),
        "inputs_already_rectified": bool(inputs_already_rectified),
        "resize_to_calib": bool(resize_to_calib),
    }
