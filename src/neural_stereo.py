import os
import sys
import re
import glob
import logging
import cv2
import numpy as np
import torch
import imageio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
FOUNDATION_DIR = os.path.join(PROJECT_ROOT, "FoundationStereo")

# Папки с ГОТОВЫМИ rectified парами
LEFT_DIR  = os.path.join(PROJECT_ROOT, "rectified_photos", "left")
RIGHT_DIR = os.path.join(PROJECT_ROOT, "rectified_photos", "right")

# !!!! модель, можно выбрать побольше и поточнее !!!!
CKPT_PATH = os.path.join(FOUNDATION_DIR, "pretrained_models", "11-33-40", "model_best_bp2.pth")

STEREO_YAML = os.path.join(PROJECT_ROOT, "rectified_photos", "stereo_calibration_opencv.yml")

# Куда сохранять результаты
OUT_DIR = os.path.join(FOUNDATION_DIR, "test_outputs", "all_pairs_rectified")

# Инференс настройки
VALID_ITERS = 128
HIERA = 0 # лучше не включать
MIXED_PRECISION = False  # на macOS лучше False

# Что сохранять
SAVE_DISP_NPY = True
SAVE_VIS = True
SAVE_DEPTH_METER = True       # работает если STEREO_YAML не None и прочитался
REMOVE_INVISIBLE = True       # как в demo (правые u<0 -> disp=inf)

# Визуализация
VIS_EXT = ".jpg"
VIS_JPEG_QUALITY = 95


# Make FoundationStereo importable
sys.path.insert(0, FOUNDATION_DIR)

from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import set_logging_format, set_seed, vis_disparity
from core.foundation_stereo import FoundationStereo


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def read_stereo_fx_and_baseline(stereo_yaml_path: str):
    """
    Reads OpenCV YAML (FileStorage) with keys: T and (P1 or K1).
    baseline = ||T|| (meters)
    fx = P1[0,0] if available else K1[0,0]
    """
    if stereo_yaml_path is None or not os.path.exists(stereo_yaml_path):
        return None, None

    fs = cv2.FileStorage(stereo_yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return None, None

    T = fs.getNode("T").mat()
    P1 = fs.getNode("P1").mat()
    K1 = fs.getNode("K1").mat()
    fs.release()

    if T is None or np.size(T) == 0:
        return None, None

    T = np.asarray(T, dtype=np.float32).reshape(3, 1)
    baseline = float(np.linalg.norm(T))

    fx = None
    if P1 is not None and np.size(P1) != 0:
        P1 = np.asarray(P1, dtype=np.float32)
        fx = float(P1[0, 0])
    elif K1 is not None and np.size(K1) != 0:
        K1 = np.asarray(K1, dtype=np.float32).reshape(3, 3)
        fx = float(K1[0, 0])

    if fx is None:
        return None, None

    return fx, baseline


@torch.no_grad()
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    device = get_device()
    logging.info(f"Using device: {device}")

    # Load cfg.yaml рядом с ckpt и переопределим нужные параметры
    cfg_path = os.path.join(os.path.dirname(CKPT_PATH), "cfg.yaml")
    cfg = OmegaConf.load(cfg_path)
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"

    cfg["ckpt_dir"] = CKPT_PATH
    cfg["valid_iters"] = VALID_ITERS
    cfg["hiera"] = HIERA
    cfg["mixed_precision"] = bool(MIXED_PRECISION) and (device.type == "cuda")

    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")

    # Model
    model = FoundationStereo(args)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device).eval()
    logging.info("Model loaded.")

    # Optional: K + baseline for depth
    fx0, baseline = read_stereo_fx_and_baseline(STEREO_YAML)
    if SAVE_DEPTH_METER and (fx0 is None or baseline is None):
        logging.warning("SAVE_DEPTH_METER=True but cannot read K/baseline from STEREO_YAML. Depth will be skipped.")
        save_depth = False
    else:
        save_depth = bool(SAVE_DEPTH_METER)

    # Pairs
    left_files = list_images(LEFT_DIR)
    right_files = list_images(RIGHT_DIR)

    if not left_files:
        raise RuntimeError(f"No images in LEFT_DIR: {LEFT_DIR}")
    if not right_files:
        raise RuntimeError(f"No images in RIGHT_DIR: {RIGHT_DIR}")

    L = {numeric_key(p): p for p in left_files}
    R = {numeric_key(p): p for p in right_files}
    keys = sorted(set(L.keys()) & set(R.keys()))
    if not keys:
        raise RuntimeError("No matching numbered pairs between LEFT_DIR and RIGHT_DIR")

    logging.info(f"Found {len(keys)} pairs: {keys}")

    for idx, k in enumerate(keys):
        left_path = L[k]
        right_path = R[k]

        out_pair_dir = os.path.join(OUT_DIR, f"{k:02d}")
        os.makedirs(out_pair_dir, exist_ok=True)

        # Read images (RGB expected by original demo; imageio gives RGB for jpg)
        img0 = imageio.imread(left_path)
        img1 = imageio.imread(right_path)

        # Ensure 3-channel
        if img0.ndim == 2:
            img0 = np.stack([img0]*3, axis=-1)
        if img1.ndim == 2:
            img1 = np.stack([img1]*3, axis=-1)

        H, W = img0.shape[:2]
        img0_ori = img0.copy()

        # To tensor (B,3,H,W)
        t0 = torch.as_tensor(img0).to(device).float()[None].permute(0, 3, 1, 2)
        t1 = torch.as_tensor(img1).to(device).float()[None].permute(0, 3, 1, 2)

        padder = InputPadder(t0.shape, divis_by=32, force_square=False)
        t0, t1 = padder.pad(t0, t1)

        # Inference (AMP only on CUDA if enabled)
        if bool(args.mixed_precision) and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if not HIERA:
                    disp = model.forward(t0, t1, iters=VALID_ITERS, test_mode=True)
                else:
                    disp = model.run_hierachical(t0, t1, iters=VALID_ITERS, test_mode=True, small_ratio=0.5)
        else:
            if not HIERA:
                disp = model.forward(t0, t1, iters=VALID_ITERS, test_mode=True)
            else:
                disp = model.run_hierachical(t0, t1, iters=VALID_ITERS, test_mode=True, small_ratio=0.5)

        disp = padder.unpad(disp.float())
        disp = disp.detach().cpu().numpy().reshape(H, W)

        if REMOVE_INVISIBLE:
            yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing="ij")
            us_right = xx - disp
            invalid = us_right < 0
            disp[invalid] = np.inf

        if SAVE_DISP_NPY:
            np.save(os.path.join(out_pair_dir, "disp.npy"), disp)

        if SAVE_VIS:
            vis = vis_disparity(disp)
            vis = np.concatenate([img0_ori, vis], axis=1)
            out_vis = os.path.join(out_pair_dir, f"vis{VIS_EXT}")
            if VIS_EXT.lower() in [".jpg", ".jpeg"]:
                cv2.imwrite(out_vis, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(VIS_JPEG_QUALITY)])
            else:
                imageio.imwrite(out_vis, vis)

        if save_depth:
            fx = float(fx0)
            depth = np.full_like(disp, np.inf, dtype=np.float32)
            valid = np.isfinite(disp) & (disp > 0.1)
            depth[valid] = (fx * float(baseline)) / disp[valid]
            np.save(os.path.join(out_pair_dir, "depth_meter.npy"), depth)


        if idx % 1 == 0:
            logging.info(f"Processed pair {k} ({idx+1}/{len(keys)}) -> {out_pair_dir}")

    logging.info(f"DONE. Saved all results to: {OUT_DIR}")


if __name__ == "__main__":
    main()
