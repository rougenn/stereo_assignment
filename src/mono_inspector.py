import os
import re
import glob
import cv2
import numpy as np

# =========================
# CONSTANTS (EDIT HERE)
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Где лежат исходные кадры (правая камера, rectified)
RIGHT_IMG_DIR = os.path.join(PROJECT_ROOT, "rectified_photos", "right")

# Где лежат результаты Metric3D ONNX
M3D_OUT_DIR = os.path.join(PROJECT_ROOT, "metric3d_outputs", "right")

# Выбор стартовой пары
START_IDX = 1

# Файлы внутри папки результата
DEPTH_NPY_NAME = "depth_meter.npy"

# Визуализация
WINDOW_NAME = "Metric3D Inspector"
COLORMAP = cv2.COLORMAP_TURBO

# Автодиапазон глубины (для colormap)
AUTO_DEPTH_RANGE = True
DEPTH_P_LOW = 2.0
DEPTH_P_HIGH = 98.0

# Жёсткие границы, если AUTO_DEPTH_RANGE=False
DEPTH_MIN_M = 0.3
DEPTH_MAX_M = 50.0

# Для отображения NaN/inf
FILL_INVALID_WITH = "max"  # "max" or "min"

# Клавиши:
# q/ESC: выход
# n/p: next/prev
# s: сохранить скриншот
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


def robust_range(arr: np.ndarray, p_low: float, p_high: float):
    mask = np.isfinite(arr)
    if not np.any(mask):
        return 0.0, 1.0
    vals = arr[mask].astype(np.float32)
    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def colorize_depth(depth_m: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    d = depth_m.copy().astype(np.float32)

    # заполним невалидные
    if FILL_INVALID_WITH == "min":
        fill = vmin
    else:
        fill = vmax
    d[~np.isfinite(d)] = fill

    d = np.clip(d, vmin, vmax)
    norm = (d - vmin) / (vmax - vmin + 1e-8)
    norm = 1.0 - norm  # ближе = теплее
    img8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(img8, COLORMAP)


def find_image_for_idx(idx: int) -> str:
    # Ищем 1.jpg / 1.png и т.п.
    candidates = []
    for ext in ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]:
        candidates.append(os.path.join(RIGHT_IMG_DIR, f"{idx}.{ext}"))
    for p in candidates:
        if os.path.exists(p):
            return p

    # fallback: по списку
    files = list_images(RIGHT_IMG_DIR)
    for p in files:
        if numeric_key(p) == idx:
            return p
    return ""


def load_case(idx: int):
    # result dir: metric3d_outputs/right/01
    rdir = os.path.join(M3D_OUT_DIR, f"{idx:02d}")
    depth_path = os.path.join(rdir, DEPTH_NPY_NAME)

    if not os.path.exists(depth_path):
        return None

    depth = np.load(depth_path).astype(np.float32)

    img_path = find_image_for_idx(idx)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR) if img_path else None
    if img is None:
        img = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
        img_path = "(not found)"

    # подгоняем размеры (на всякий случай)
    if img.shape[:2] != depth.shape[:2]:
        img = cv2.resize(img, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)

    return {
        "idx": idx,
        "rdir": rdir,
        "depth": depth,
        "img": img,
        "img_path": img_path,
        "depth_path": depth_path,
    }


class State:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.locked = False
        self.lx = 0
        self.ly = 0


def main():
    state = State()
    idx = int(START_IDX)

    data = load_case(idx)
    if data is None:
        raise SystemExit(f"Cannot load idx={idx}. Check M3D_OUT_DIR={M3D_OUT_DIR}")

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_MOUSEMOVE:
            state.x, state.y = x, y
        elif event == cv2.EVENT_LBUTTONDOWN:
            state.locked = True
            state.lx, state.ly = x, y
        elif event == cv2.EVENT_RBUTTONDOWN:
            state.locked = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        img = data["img"].copy()
        depth = data["depth"]

        H, W = depth.shape[:2]

        cx = int(np.clip(state.x, 0, W - 1))
        cy = int(np.clip(state.y, 0, H - 1))
        if state.locked:
            px, py = int(np.clip(state.lx, 0, W - 1)), int(np.clip(state.ly, 0, H - 1))
        else:
            px, py = cx, cy

        z = float(depth[py, px])
        z_valid = np.isfinite(z) and z > 0.0

        if AUTO_DEPTH_RANGE:
            vmin, vmax = robust_range(depth, DEPTH_P_LOW, DEPTH_P_HIGH)
        else:
            vmin, vmax = float(DEPTH_MIN_M), float(DEPTH_MAX_M)

        depth_vis = colorize_depth(depth, vmin, vmax)

        # markers
        cv2.circle(img, (px, py), 6, (0, 255, 0), 2)
        cv2.circle(depth_vis, (px, py), 6, (0, 255, 0), 2)

        # concat
        vis = np.concatenate([img, depth_vis], axis=1)

        # header
        if z_valid:
            msg = f"idx={data['idx']:02d}  x={px} y={py}  depth={z:.3f} m   (range {vmin:.2f}..{vmax:.2f})"
        else:
            msg = f"idx={data['idx']:02d}  x={px} y={py}  depth=INVALID   (range {vmin:.2f}..{vmax:.2f})"

        cv2.rectangle(vis, (0, 0), (vis.shape[1], 45), (0, 0, 0), -1)
        cv2.putText(vis, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        hint = "mouse move=read | LMB=lock | RMB=unlock | n/p next/prev | s screenshot | q quit"
        cv2.rectangle(vis, (0, vis.shape[0] - 35), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
        cv2.putText(vis, hint, (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, vis)
        key = cv2.waitKey(15) & 0xFF

        if key in [27, ord("q")]:
            break
        elif key == ord("n"):
            idx += 1
            nd = load_case(idx)
            if nd is not None:
                data = nd
                state.locked = False
        elif key == ord("p"):
            idx = max(1, idx - 1)
            nd = load_case(idx)
            if nd is not None:
                data = nd
                state.locked = False
        elif key == ord("s"):
            out_path = os.path.join(data["rdir"], "inspect_depth.png")
            cv2.imwrite(out_path, vis)
            print(f"[SAVED] {out_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
