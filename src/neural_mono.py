import os
import re
import glob
import json
import logging
from typing import Optional, Tuple, Dict

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Input: уже rectified
RIGHT_DIR = os.path.join(PROJECT_ROOT, "rectified_photos", "right")

# Калибровки (берём fx из P2 если он есть, иначе из camera_matrix/K2)
STEREO_YAML = os.path.join(PROJECT_ROOT, "rectified_photos", "stereo_calibration_opencv.yml")
RIGHT_CALIB_YAML = os.path.join(PROJECT_ROOT, "rectified_photos", "right_calibration_opencv.yml")

# Output
OUT_DIR = os.path.join(PROJECT_ROOT, "metric3d_outputs", "right")

# Hugging Face ONNX model
HF_MODEL_ID = "onnx-community/metric3d-vit-large"
HF_ONNX_PATH_FP32 = "onnx/model.onnx"
HF_ONNX_PATH_FP16 = "onnx/model_fp16.onnx"

# Важно: fp16 на CPU часто НЕ поддерживается -> оставь False, если не уверен
USE_FP16_MODEL = False

# ONNXRuntime providers (на macOS чаще всего реально работает CPU; CoreML может быть недоступен)
PREFERRED_PROVIDERS = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

# Preprocess (из preprocessor_config.json HF)
TARGET_SIZE = (518, 518)          # (H, W)
ENSURE_MULTIPLE_OF = 14
KEEP_ASPECT_RATIO = True
INTERP = cv2.INTER_CUBIC          # resample=3 ~ bicubic :contentReference[oaicite:2]{index=2}

# В конфиге HF: do_rescale=false, do_normalize=false :contentReference[oaicite:3]{index=3}
# Но на практике иногда модель ожидает 0..1. Оставляю тумблер:
RESCALE_TO_0_1 = False            # если глубина выглядит странно — попробуй True

# Canonical camera focal length (как в Metric3D): 1000px :contentReference[oaicite:4]{index=4}
CANONICAL_FOCAL_PX = 1000.0

# Сохранения
SAVE_DEPTH_NPY = True
SAVE_DEPTH_VIS = True
SAVE_DEPTH_RAW_CANONICAL = False  # если хочешь сохранить до умножения на fx/1000

# Визуализация глубины
COLORMAP = cv2.COLORMAP_TURBO
AUTO_RANGE = True
P_LOW = 2.0
P_HIGH = 98.0
CLAMP_MIN_M = 0.0
CLAMP_MAX_M = 300.0               # как в issue: clamp 0..300 :contentReference[oaicite:5]{index=5}

# =========================


def numeric_key(path: str) -> int:
    m = re.match(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def list_images(folder: str):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files, key=numeric_key)


def read_fx_from_opencv_yaml() -> Optional[float]:
    """
    Пытаемся вытащить fx (в пикселях) для ПРАВОЙ rectified камеры.
    Приоритет:
      1) STEREO_YAML: P2[0,0] или K2[0,0]
      2) RIGHT_CALIB_YAML: camera_matrix[0,0]
    """
    def _get_mat(fs, key: str):
        node = fs.getNode(key)
        if node is None:
            return None
        mat = node.mat()
        if mat is None or np.size(mat) == 0:
            return None
        return mat

    # 1) stereo yaml
    if os.path.exists(STEREO_YAML):
        fs = cv2.FileStorage(STEREO_YAML, cv2.FILE_STORAGE_READ)
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

    # 2) right calib yaml
    if os.path.exists(RIGHT_CALIB_YAML):
        fs = cv2.FileStorage(RIGHT_CALIB_YAML, cv2.FILE_STORAGE_READ)
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


def resize_keep_aspect_multiple(img_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Resize по правилам HF-конфига:
      - do_resize=true
      - keep_aspect_ratio=true
      - ensure_multiple_of=14
      - size 518x518 :contentReference[oaicite:6]{index=6}
    Возвращаем (img_resized, scale), где scale применён к обоим осям.
    """
    H, W = img_rgb.shape[:2]
    th, tw = TARGET_SIZE

    if KEEP_ASPECT_RATIO:
        scale = min(th / H, tw / W)
        nh = max(ENSURE_MULTIPLE_OF, int((H * scale) // ENSURE_MULTIPLE_OF) * ENSURE_MULTIPLE_OF)
        nw = max(ENSURE_MULTIPLE_OF, int((W * scale) // ENSURE_MULTIPLE_OF) * ENSURE_MULTIPLE_OF)
    else:
        nh = (th // ENSURE_MULTIPLE_OF) * ENSURE_MULTIPLE_OF
        nw = (tw // ENSURE_MULTIPLE_OF) * ENSURE_MULTIPLE_OF
        scale = nh / H

    resized = cv2.resize(img_rgb, (nw, nh), interpolation=INTERP)
    # фактический scale (после округления до multiple-of)
    scale_eff = nw / float(W)
    return resized, float(scale_eff)


def robust_range(x: np.ndarray, p_low: float, p_high: float) -> Tuple[float, float]:
    mask = np.isfinite(x)
    if not np.any(mask):
        return 0.0, 1.0
    vals = x[mask].astype(np.float32)
    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    d = depth_m.copy().astype(np.float32)
    d[~np.isfinite(d)] = np.nan
    d = np.clip(d, CLAMP_MIN_M, CLAMP_MAX_M)

    if AUTO_RANGE:
        vmin, vmax = robust_range(d, P_LOW, P_HIGH)
    else:
        vmin, vmax = CLAMP_MIN_M, CLAMP_MAX_M

    dd = d.copy()
    dd[~np.isfinite(dd)] = vmax
    dd = np.clip(dd, vmin, vmax)
    norm = (dd - vmin) / (vmax - vmin + 1e-8)
    norm = 1.0 - norm  # ближе = теплее
    img8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(img8, COLORMAP)


def load_onnx_session(model_path: str):
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # провайдеры: попробуем CoreML, если не получится — CPU
    available = ort.get_available_providers()
    providers = [p for p in PREFERRED_PROVIDERS if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)

    # Для отладки (имена входов/выходов)
    inps = [(i.name, i.shape, i.type) for i in session.get_inputs()]
    outs = [(o.name, o.shape, o.type) for o in session.get_outputs()]
    logging.info(f"ONNX providers: {session.get_providers()}")
    logging.info(f"ONNX inputs: {inps}")
    logging.info(f"ONNX outputs: {outs}")
    return session


def pick_io_names(session) -> Tuple[str, str]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    in_name = None
    for i in inputs:
        if "pixel" in i.name or "input" in i.name:
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


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ensure_dir(OUT_DIR)

    # 1) download model from HF
    from huggingface_hub import hf_hub_download

    onnx_path_in_repo = HF_ONNX_PATH_FP16 if USE_FP16_MODEL else HF_ONNX_PATH_FP32
    logging.info(f"Downloading {HF_MODEL_ID}/{onnx_path_in_repo} ... (big file)")
    local_onnx = hf_hub_download(repo_id=HF_MODEL_ID, filename=onnx_path_in_repo)
    logging.info(f"Model downloaded to: {local_onnx}")

    # 2) onnxruntime session
    session = load_onnx_session(local_onnx)
    in_name, out_name = pick_io_names(session)
    logging.info(f"Using input='{in_name}', output='{out_name}'")

    # 3) read fx
    fx_orig = read_fx_from_opencv_yaml()
    if fx_orig is None:
        logging.warning("Cannot read fx from YAML. Will still run, but metric scaling (meters) may be wrong.")
    else:
        logging.info(f"fx (orig/rectified) = {fx_orig:.3f} px")

    # save meta
    meta = {
        "hf_model_id": HF_MODEL_ID,
        "onnx_path": onnx_path_in_repo,
        "providers": session.get_providers(),
        "target_size": TARGET_SIZE,
        "ensure_multiple_of": ENSURE_MULTIPLE_OF,
        "keep_aspect_ratio": KEEP_ASPECT_RATIO,
        "rescale_to_0_1": RESCALE_TO_0_1,
        "canonical_focal_px": CANONICAL_FOCAL_PX,
        "fx_orig_px": fx_orig,
        "note": "depth_metric = depth_canonical * (fx_input / 1000.0) (canonical focal=1000px)",
    }
    with open(os.path.join(OUT_DIR, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 4) run all right images
    files = list_images(RIGHT_DIR)
    if not files:
        raise RuntimeError(f"No images in {RIGHT_DIR}")

    logging.info(f"Found {len(files)} images in {RIGHT_DIR}")

    for idx, path in enumerate(files, 1):
        stem = re.match(r"(\d+)", os.path.basename(path))
        key = int(stem.group(1)) if stem else idx
        out_dir = os.path.join(OUT_DIR, f"{key:02d}")
        ensure_dir(out_dir)

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            logging.warning(f"Skip unreadable: {path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = rgb.shape[:2]

        # preprocess (HF)
        rgb_in, scale_eff = resize_keep_aspect_multiple(rgb)
        H1, W1 = rgb_in.shape[:2]

        # to tensor-like (1,3,H,W) float32
        x = rgb_in.astype(np.float32)
        if RESCALE_TO_0_1:
            x = x / 255.0
        x = np.transpose(x, (2, 0, 1))[None, :, :, :]  # NCHW

        # inference
        y = session.run([out_name], {in_name: x})[0]

        # привести к (H,W)
        y = np.asarray(y)
        if y.ndim == 4:
            # (N,1,H,W) or (N,C,H,W)
            y = y[0, 0]
        elif y.ndim == 3:
            y = y[0]
        y = y.astype(np.float32)

        depth_canon = y  # canonical camera space depth

        # upsample depth to original resolution
        depth_canon_up = cv2.resize(depth_canon, (W0, H0), interpolation=cv2.INTER_LINEAR)

        # de-canonical to metric meters: * (fx_input / 1000) :contentReference[oaicite:7]{index=7}
        if fx_orig is not None:
            fx_input = float(fx_orig) * float(scale_eff)  # как в issue: intrinsic scaled вместе с resize :contentReference[oaicite:8]{index=8}
            depth_m = depth_canon_up * (fx_input / CANONICAL_FOCAL_PX)
        else:
            depth_m = depth_canon_up  # без гарантии метрики

        depth_m = np.clip(depth_m, CLAMP_MIN_M, CLAMP_MAX_M).astype(np.float32)

        if SAVE_DEPTH_RAW_CANONICAL:
            np.save(os.path.join(out_dir, "depth_canonical.npy"), depth_canon_up)

        if SAVE_DEPTH_NPY:
            np.save(os.path.join(out_dir, "depth_meter.npy"), depth_m)

        if SAVE_DEPTH_VIS:
            vis = colorize_depth(depth_m)
            cv2.imwrite(os.path.join(out_dir, "depth_vis.png"), vis)

        if idx % 1 == 0:
            logging.info(f"[{idx}/{len(files)}] {os.path.basename(path)} -> {out_dir}")

    logging.info(f"DONE. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
