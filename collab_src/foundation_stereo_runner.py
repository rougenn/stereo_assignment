# foundation_stereo_runner.py
from __future__ import annotations

import os
import re
import glob
import sys
import logging
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import cv2
import torch

try:
    import imageio.v2 as imageio  # стабильнее в новых версиях
except Exception:
    import imageio


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


def _ensure_3ch_rgb(img: np.ndarray) -> np.ndarray:
    # imageio обычно даёт RGB uint8
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def _get_device(device: str = "auto") -> torch.device:
    device = (device or "auto").lower()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _read_stereo_fx_and_baseline(stereo_yaml_path: str) -> tuple[Optional[float], Optional[float]]:
    """
    Reads OpenCV YAML (FileStorage) with keys: T and (P1 or K1).
    baseline = ||T|| (meters)
    fx = P1[0,0] if available else K1[0,0]
    """
    if stereo_yaml_path is None or (not os.path.exists(stereo_yaml_path)):
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
def run_foundation_stereo(
    *,
    left_dir: str,
    right_dir: str,
    foundation_dir: str,
    ckpt_path: str,
    out_dir: str,

    stereo_yaml_path: Optional[str] = None,   # нужно только если save_depth_meter=True

    # inference params
    valid_iters: int = 128,
    hiera: int = 0,
    mixed_precision: bool = False,
    seed: int = 0,
    device: str = "auto",                     # "auto" | "cuda" | "mps" | "cpu"

    # behavior / outputs
    remove_invisible: bool = True,
    save_disp_npy: bool = True,
    save_vis: bool = True,
    save_depth_meter: bool = True,

    # vis settings
    vis_ext: str = ".jpg",
    vis_jpeg_quality: int = 95,

    # padding
    pad_divis_by: int = 32,

    # logging
    log_every: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Runs FoundationStereo on all numbered pairs in left_dir/right_dir.
    Expects filenames like 1.png, 2.jpg, ... in both folders.

    Saves per-pair folder: out_dir/01, out_dir/02, ...
      - disp.npy (if save_disp_npy)
      - vis.jpg/png (if save_vis) -> [left | disp_vis]
      - depth_meter.npy (if save_depth_meter and stereo_yaml readable)

    Returns a dict with summary info.
    """
    # --- sanity ---
    if not os.path.isdir(left_dir):
        raise FileNotFoundError(f"left_dir not found: {left_dir}")
    if not os.path.isdir(right_dir):
        raise FileNotFoundError(f"right_dir not found: {right_dir}")
    if not os.path.isdir(foundation_dir):
        raise FileNotFoundError(f"foundation_dir not found: {foundation_dir}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt_path not found: {ckpt_path}")

    os.makedirs(out_dir, exist_ok=True)

    # --- make FoundationStereo importable ---
    # аккуратно добавим в sys.path (без удаления, это обычно норм для ноутбука)
    if foundation_dir not in sys.path:
        sys.path.insert(0, foundation_dir)

    # imports from repo
    from omegaconf import OmegaConf
    from core.utils.utils import InputPadder
    from Utils import set_logging_format, set_seed, vis_disparity
    from core.foundation_stereo import FoundationStereo

    # logging / seed
    if verbose:
        try:
            set_logging_format()
        except Exception:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    set_seed(int(seed))
    torch.autograd.set_grad_enabled(False)

    dev = _get_device(device)
    if verbose:
        logging.info(f"Using device: {dev}")

    # --- load cfg.yaml next to ckpt and override runtime params ---
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "cfg.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"cfg.yaml not found next to ckpt: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"

    cfg["ckpt_dir"] = ckpt_path
    cfg["valid_iters"] = int(valid_iters)
    cfg["hiera"] = int(hiera)
    cfg["mixed_precision"] = bool(mixed_precision) and (dev.type == "cuda")

    args = OmegaConf.create(cfg)
    if verbose:
        logging.info(f"args:\n{args}")

    # --- model ---
    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" not in ckpt:
        raise RuntimeError("Checkpoint does not contain key 'model' (unexpected format).")

    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(dev).eval()
    if verbose:
        logging.info("Model loaded.")

    # --- optional depth scale: fx + baseline ---
    fx0, baseline = _read_stereo_fx_and_baseline(stereo_yaml_path) if stereo_yaml_path else (None, None)
    save_depth = bool(save_depth_meter) and (fx0 is not None) and (baseline is not None)
    if save_depth_meter and not save_depth and verbose:
        logging.warning("save_depth_meter=True, but cannot read fx/baseline from stereo_yaml_path -> depth will NOT be saved.")

    # --- pairs ---
    left_files = _list_images(left_dir)
    right_files = _list_images(right_dir)
    if not left_files:
        raise RuntimeError(f"No images in left_dir: {left_dir}")
    if not right_files:
        raise RuntimeError(f"No images in right_dir: {right_dir}")

    L = { _numeric_key(p): p for p in left_files }
    R = { _numeric_key(p): p for p in right_files }
    keys = sorted(set(L.keys()) & set(R.keys()))
    if not keys:
        raise RuntimeError("No matching numbered pairs between left_dir and right_dir.")

    if verbose:
        logging.info(f"Found {len(keys)} pairs: {keys}")

    processed: List[int] = []
    skipped: List[int] = []

    for idx, k in enumerate(keys):
        left_path = L[k]
        right_path = R[k]

        out_pair_dir = os.path.join(out_dir, f"{k:02d}")
        os.makedirs(out_pair_dir, exist_ok=True)

        try:
            img0 = _ensure_3ch_rgb(imageio.imread(left_path))
            img1 = _ensure_3ch_rgb(imageio.imread(right_path))
        except Exception as e:
            skipped.append(k)
            if verbose:
                logging.warning(f"[SKIP] cannot read pair {k}: {e}")
            continue

        if img0.shape[:2] != img1.shape[:2]:
            skipped.append(k)
            if verbose:
                logging.warning(f"[SKIP] size mismatch pair {k}: left={img0.shape} right={img1.shape}")
            continue

        H, W = img0.shape[:2]
        img0_ori = img0.copy()

        # to tensor (B,3,H,W), float32
        t0 = torch.as_tensor(img0).to(dev).float()[None].permute(0, 3, 1, 2)
        t1 = torch.as_tensor(img1).to(dev).float()[None].permute(0, 3, 1, 2)

        padder = InputPadder(t0.shape, divis_by=int(pad_divis_by), force_square=False)
        t0p, t1p = padder.pad(t0, t1)

        # inference
        if bool(args.mixed_precision) and dev.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                if int(hiera) == 0:
                    disp = model.forward(t0p, t1p, iters=int(valid_iters), test_mode=True)
                else:
                    disp = model.run_hierachical(t0p, t1p, iters=int(valid_iters), test_mode=True, small_ratio=0.5)
        else:
            if int(hiera) == 0:
                disp = model.forward(t0p, t1p, iters=int(valid_iters), test_mode=True)
            else:
                disp = model.run_hierachical(t0p, t1p, iters=int(valid_iters), test_mode=True, small_ratio=0.5)

        disp = padder.unpad(disp.float()).detach().cpu().numpy().reshape(H, W)

        # remove invisible (as in demo)
        if remove_invisible:
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            us_right = xx - disp
            disp[us_right < 0] = np.inf

        # save disp
        if save_disp_npy:
            np.save(os.path.join(out_pair_dir, "disp.npy"), disp)

        # save vis
        if save_vis:
            disp_vis = vis_disparity(disp)        # RGB
            vis = np.concatenate([img0_ori, disp_vis], axis=1)
            out_vis = os.path.join(out_pair_dir, f"vis{vis_ext}")

            if vis_ext.lower() in [".jpg", ".jpeg"]:
                cv2.imwrite(
                    out_vis,
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(vis_jpeg_quality)],
                )
            else:
                imageio.imwrite(out_vis, vis)

        # save depth (meters) if possible
        if save_depth:
            fx = float(fx0)
            B = float(baseline)
            depth = np.full_like(disp, np.inf, dtype=np.float32)
            valid = np.isfinite(disp) & (disp > 0.1)
            depth[valid] = (fx * B) / disp[valid]
            np.save(os.path.join(out_pair_dir, "depth_meter.npy"), depth)

        processed.append(k)

        if verbose and (log_every > 0) and ((idx % int(log_every)) == 0):
            logging.info(f"Processed pair {k} ({idx+1}/{len(keys)}) -> {out_pair_dir}")

    if verbose:
        logging.info(f"DONE. Saved all results to: {out_dir}")

    return {
        "out_dir": out_dir,
        "processed_keys": processed,
        "skipped_keys": skipped,
        "device": str(dev),
        "valid_iters": int(valid_iters),
        "hiera": int(hiera),
        "mixed_precision_used": bool(args.mixed_precision),
        "save_depth_meter": bool(save_depth),
        "fx": float(fx0) if fx0 is not None else None,
        "baseline_m": float(baseline) if baseline is not None else None,
        "num_pairs_total": len(keys),
    }
