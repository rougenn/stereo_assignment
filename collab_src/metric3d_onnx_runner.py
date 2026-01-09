# metric3d_onnx_runner.py
from __future__ import annotations

import os
import re
import glob
import json
import logging
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np


def _numeric_key(path: str) -> int:
    m = re.match(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def _list_images(folder: str) -> List[str]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files: List[str] = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files, key=_numeric_key)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _robust_range(x: np.ndarray, p_low: float, p_high: float) -> Tuple[float, float]:
    mask = np.isfinite(x)
    if not np.any(mask):
        return 0.0, 1.0
    vals = x[mask].astype(np.float32)
    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _colorize_depth(
    depth_m: np.ndarray,
    *,
    colormap: int,
    auto_range: bool,
    p_low: float,
    p_high: float,
    clamp_min_m: float,
    clamp_max_m: float,
) -> np.ndarray:
    d = depth_m.copy().astype(np.float32)
    d[~np.isfinite(d)] = np.nan
    d = np.clip(d, clamp_min_m, clamp_max_m)

    if auto_range:
        vmin, vmax = _robust_range(d, p_low, p_high)
    else:
        vmin, vmax = clamp_min_m, clamp_max_m

    dd = d.copy()
    dd[~np.isfinite(dd)] = vmax
    dd = np.clip(dd, vmin, vmax)
    norm = (dd - vmin) / (vmax - vmin + 1e-8)
    norm = 1.0 - norm  # ближе = теплее
    img8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(img8, colormap)


def _read_fx_from_opencv_yamls(
    *,
    stereo_yaml_path: Optional[str],
    right_calib_yaml_path: Optional[str],
) -> Optional[float]:
    """
    Пытаемся вытащить fx (px) для ПРАВОЙ rectified камеры.
    Приоритет:
      1) stereo_yaml: P2[0,0] или K2[0,0]
      2) right_calib_yaml: camera_matrix[0,0] (или K/K2)
    """
    def _get_mat(fs, key: str):
        node = fs.getNode(key)
        if node is None:
            return None
        mat = node.mat()
        if mat is None or np.size(mat) == 0:
            return None
        return mat

    if stereo_yaml_path and os.path.exists(stereo_yaml_path):
        fs = cv2.FileStorage(stereo_yaml_path, cv2.FILE_STORAGE_READ)
        if fs.isOpened():
            P2 = _get_mat(fs, "P2")
            if P2 is not None:
                P2 = np.asarray(P2, dtype=np.float32)
                fs.release()
                return float(P2[0, 0])

            K2 = _get_mat(fs, "K2")
            if K2 is not None:
                K2 = np.asarray(K2, dtype=np.float32).reshape(3, 3)
                fs.release()
                return float(K2[0, 0])
            fs.release()

    if right_calib_yaml_path and os.path.exists(right_calib_yaml_path):
        fs = cv2.FileStorage(right_calib_yaml_path, cv2.FILE_STORAGE_READ)
        if fs.isOpened():
            K = _get_mat(fs, "camera_matrix")
            if K is None:
                K = _get_mat(fs, "K")
            if K is None:
                K = _get_mat(fs, "K2")
            fs.release()
            if K is not None:
                K = np.asarray(K, dtype=np.float32).reshape(3, 3)
                return float(K[0, 0])

    return None


def _resize_keep_aspect_multiple(
    img_rgb: np.ndarray,
    *,
    target_size_hw: Tuple[int, int],
    ensure_multiple_of: int,
    keep_aspect_ratio: bool,
    interp: int,
) -> Tuple[np.ndarray, float]:
    """
    Resize по правилам HF-конфига:
      - keep_aspect_ratio (обычно True)
      - ensure_multiple_of (обычно 14)
      - target_size (обычно 518x518)
    Возвращает (img_resized, scale_eff), где scale_eff = new_w / old_w
    """
    H, W = img_rgb.shape[:2]
    th, tw = int(target_size_hw[0]), int(target_size_hw[1])

    if keep_aspect_ratio:
        scale = min(th / H, tw / W)
        nh = max(ensure_multiple_of, int((H * scale) // ensure_multiple_of) * ensure_multiple_of)
        nw = max(ensure_multiple_of, int((W * scale) // ensure_multiple_of) * ensure_multiple_of)
    else:
        nh = (th // ensure_multiple_of) * ensure_multiple_of
        nw = (tw // ensure_multiple_of) * ensure_multiple_of

    resized = cv2.resize(img_rgb, (nw, nh), interpolation=interp)
    scale_eff = nw / float(W)
    return resized, float(scale_eff)


def _load_onnx_session(
    model_path: str,
    preferred_providers: List[str],
):
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available = ort.get_available_providers()
    providers = [p for p in preferred_providers if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
    logging.info(f"ONNX providers: {session.get_providers()}")
    return session


def _pick_io_names(session) -> Tuple[str, str]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    in_name = None
    for i in inputs:
        if ("pixel" in i.name.lower()) or ("input" in i.name.lower()):
            in_name = i.name
            break
    if in_name is None:
        in_name = inputs[0].name

    out_name = None
    for o in outputs:
        n = o.name.lower()
        if "depth" in n:
            out_name = o.name
            break
    if out_name is None:
        out_name = outputs[0].name

    return in_name, out_name


def run_metric3d_onnx(
    *,
    right_dir: str,
    out_dir: str,

    # calibration for metric scaling (optional but recommended)
    stereo_yaml_path: Optional[str] = None,
    right_calib_yaml_path: Optional[str] = None,

    # HF model info
    hf_model_id: str = "onnx-community/metric3d-vit-large",
    hf_onnx_path_fp32: str = "onnx/model.onnx",
    hf_onnx_path_fp16: str = "onnx/model_fp16.onnx",
    use_fp16_model: bool = False,

    # OR: provide local onnx path directly (if not None, skip HF download)
    local_onnx_path: Optional[str] = None,

    # onnxruntime providers
    preferred_providers: Optional[List[str]] = None,

    # preprocess
    target_size_hw: Tuple[int, int] = (518, 518),   # (H,W)
    ensure_multiple_of: int = 14,
    keep_aspect_ratio: bool = True,
    interp: int = cv2.INTER_CUBIC,
    rescale_to_0_1: bool = False,

    # canonical focal
    canonical_focal_px: float = 1000.0,

    # save options
    save_depth_npy: bool = True,
    save_depth_vis: bool = True,
    save_depth_raw_canonical: bool = False,

    # vis options
    colormap: int = cv2.COLORMAP_TURBO,
    auto_range: bool = True,
    p_low: float = 2.0,
    p_high: float = 98.0,

    # clamp
    clamp_min_m: float = 0.0,
    clamp_max_m: float = 300.0,

    # logging
    log_every: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Runs Metric3D ONNX on all images in right_dir (assumed already rectified).
    Saves results into out_dir/01, out_dir/02, ...

    Metric scaling (meters):
      depth_m = depth_canonical_up * (fx_input / canonical_focal_px)
      where fx_input = fx_orig * scale_eff (if fx known)
    """
    if preferred_providers is None:
        preferred_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    _ensure_dir(out_dir)

    # --- model path ---
    if local_onnx_path is None:
        from huggingface_hub import hf_hub_download

        onnx_path_in_repo = hf_onnx_path_fp16 if use_fp16_model else hf_onnx_path_fp32
        logging.info(f"Downloading {hf_model_id}/{onnx_path_in_repo} ...")
        local_onnx_path = hf_hub_download(repo_id=hf_model_id, filename=onnx_path_in_repo)
        logging.info(f"Model downloaded to: {local_onnx_path}")
    else:
        if not os.path.exists(local_onnx_path):
            raise FileNotFoundError(f"local_onnx_path not found: {local_onnx_path}")
        onnx_path_in_repo = os.path.basename(local_onnx_path)

    # --- session ---
    session = _load_onnx_session(local_onnx_path, preferred_providers)
    in_name, out_name = _pick_io_names(session)
    logging.info(f"Using input='{in_name}', output='{out_name}'")

    # --- fx ---
    fx_orig = _read_fx_from_opencv_yamls(
        stereo_yaml_path=stereo_yaml_path,
        right_calib_yaml_path=right_calib_yaml_path,
    )
    if fx_orig is None:
        logging.warning("Cannot read fx from YAMLs. Depth will be saved, but metric scaling (meters) may be wrong.")
    else:
        logging.info(f"fx (orig/rectified) = {fx_orig:.3f} px")

    # meta
    meta = {
        "hf_model_id": hf_model_id,
        "onnx_path": onnx_path_in_repo,
        "local_onnx_path": local_onnx_path,
        "providers": session.get_providers(),
        "target_size_hw": list(target_size_hw),
        "ensure_multiple_of": int(ensure_multiple_of),
        "keep_aspect_ratio": bool(keep_aspect_ratio),
        "rescale_to_0_1": bool(rescale_to_0_1),
        "canonical_focal_px": float(canonical_focal_px),
        "fx_orig_px": fx_orig,
        "note": "depth_metric = depth_canonical * (fx_input / canonical_focal_px); fx_input = fx_orig * scale_eff",
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # images
    files = _list_images(right_dir)
    if not files:
        raise RuntimeError(f"No images in right_dir: {right_dir}")

    logging.info(f"Found {len(files)} images in {right_dir}")

    processed = []
    skipped = []

    for idx, path in enumerate(files, 1):
        stem = re.match(r"(\d+)", os.path.basename(path))
        key = int(stem.group(1)) if stem else idx
        out_k = os.path.join(out_dir, f"{key:02d}")
        _ensure_dir(out_k)

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            skipped.append(key)
            logging.warning(f"Skip unreadable: {path}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = rgb.shape[:2]

        # preprocess
        rgb_in, scale_eff = _resize_keep_aspect_multiple(
            rgb,
            target_size_hw=target_size_hw,
            ensure_multiple_of=ensure_multiple_of,
            keep_aspect_ratio=keep_aspect_ratio,
            interp=interp,
        )

        # to NCHW float32
        x = rgb_in.astype(np.float32)
        if rescale_to_0_1:
            x = x / 255.0
        x = np.transpose(x, (2, 0, 1))[None, :, :, :]

        # inference
        y = session.run([out_name], {in_name: x})[0]
        y = np.asarray(y)
        if y.ndim == 4:
            y = y[0, 0]
        elif y.ndim == 3:
            y = y[0]
        depth_canon = y.astype(np.float32)

        # upsample back to original size
        depth_canon_up = cv2.resize(depth_canon, (W0, H0), interpolation=cv2.INTER_LINEAR)

        # metric scaling
        if fx_orig is not None:
            fx_input = float(fx_orig) * float(scale_eff)
            depth_m = depth_canon_up * (fx_input / float(canonical_focal_px))
        else:
            depth_m = depth_canon_up

        depth_m = np.clip(depth_m, clamp_min_m, clamp_max_m).astype(np.float32)

        if save_depth_raw_canonical:
            np.save(os.path.join(out_k, "depth_canonical.npy"), depth_canon_up)

        if save_depth_npy:
            np.save(os.path.join(out_k, "depth_meter.npy"), depth_m)

        if save_depth_vis:
            vis = _colorize_depth(
                depth_m,
                colormap=colormap,
                auto_range=auto_range,
                p_low=p_low,
                p_high=p_high,
                clamp_min_m=clamp_min_m,
                clamp_max_m=clamp_max_m,
            )
            cv2.imwrite(os.path.join(out_k, "depth_vis.png"), vis)

        processed.append(key)
        if (log_every > 0) and (idx % int(log_every) == 0):
            logging.info(f"[{idx}/{len(files)}] {os.path.basename(path)} -> {out_k}")

    logging.info(f"DONE. Outputs in: {out_dir}")

    return {
        "out_dir": out_dir,
        "processed_keys": processed,
        "skipped_keys": skipped,
        "num_images_total": len(files),
        "fx_orig_px": float(fx_orig) if fx_orig is not None else None,
        "providers": session.get_providers(),
        "onnx_path": local_onnx_path,
    }
