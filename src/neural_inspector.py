import os
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Где лежит левый rectified кадр (чтобы видеть реальную сцену)
RECT_LEFT_DIR = os.path.join(PROJECT_ROOT, "rectified_photos", "left")

# Где лежат результаты FoundationStereo
FS_OUT_DIR = os.path.join(
    PROJECT_ROOT,
    "FoundationStereo_outputs",
)

# Stereo YAML чтобы считать метры (baseline + fx)
STEREO_YAML = os.path.join(PROJECT_ROOT, "rectified_photos", "stereo_calibration_opencv.yml")

# Выбор стартовой пары
START_PAIR = 1

# Файлы внутри папки FS пары
FS_DISP_NPY = "disp.npy"
FS_DEPTH_NPY = "depth_meter.npy"   # если вдруг есть, можем читать напрямую

# Настройки визуализации
WINDOW_NAME = "FoundationStereo Inspector"
COLORMAP = cv2.COLORMAP_TURBO

# Автодиапазон глубины (если строим depth из disp)
AUTO_DEPTH_RANGE = True
DEPTH_P_LOW = 2.0
DEPTH_P_HIGH = 98.0
DEPTH_MIN_M = 0.3
DEPTH_MAX_M = 20.0

# Диспарантность (для колормэпа)
AUTO_DISP_RANGE = True
DISP_P_LOW = 2.0
DISP_P_HIGH = 98.0

# Клавиши:
# q/ESC: выход
# n/p: next/prev
# c: переключить режим (disp <-> depth)
# s: сохранить скриншот


def read_stereo_K_and_baseline(stereo_yaml_path: str):
    if stereo_yaml_path is None or not os.path.exists(stereo_yaml_path):
        return None, None
    fs = cv2.FileStorage(stereo_yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return None, None
    K1 = fs.getNode("K1").mat()
    T = fs.getNode("T").mat()
    P1 = fs.getNode("P1").mat()  # если есть, лучше брать fx из P1
    fs.release()

    if K1 is None or T is None or np.size(K1) == 0 or np.size(T) == 0:
        return None, None

    K1 = np.asarray(K1, dtype=np.float32).reshape(3, 3)
    T = np.asarray(T, dtype=np.float32).reshape(3, 1)
    baseline = float(np.linalg.norm(T))

    # fx: лучше из P1 если есть
    if P1 is not None and np.size(P1) != 0:
        P1 = np.asarray(P1, dtype=np.float32)
        fx = float(P1[0, 0])
    else:
        fx = float(K1[0, 0])

    return fx, baseline


def robust_range(arr, p_low, p_high):
    mask = np.isfinite(arr)
    if not np.any(mask):
        return 0.0, 1.0
    vals = arr[mask].astype(np.float32)
    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def colorize_scalar_map(x: np.ndarray, vmin: float, vmax: float, invert=False):
    xx = x.copy().astype(np.float32)
    xx[~np.isfinite(xx)] = vmax

    xx = np.clip(xx, vmin, vmax)
    norm = (xx - vmin) / (vmax - vmin + 1e-8)
    if invert:
        norm = 1.0 - norm
    img8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(img8, COLORMAP)


def load_pair(pair_idx: int):
    # FS pair dir: .../all_pairs_rectified/01
    pdir = os.path.join(FS_OUT_DIR, f"{pair_idx:02d}")
    if not os.path.isdir(pdir):
        return None

    disp_path = os.path.join(pdir, FS_DISP_NPY)
    if not os.path.exists(disp_path):
        return None

    disp = np.load(disp_path).astype(np.float32)

    # left rect image from rectified_photos/left/<k>.jpg (note: there filenames are 1.jpg..)
    left_img_path = os.path.join(RECT_LEFT_DIR, f"{pair_idx}.jpg")
    left = cv2.imread(left_img_path, cv2.IMREAD_COLOR)
    if left is None:
        # fallback: try png
        left_img_path = os.path.join(RECT_LEFT_DIR, f"{pair_idx}.png")
        left = cv2.imread(left_img_path, cv2.IMREAD_COLOR)

    # If left not found, just show blank
    if left is None:
        left = np.zeros((disp.shape[0], disp.shape[1], 3), dtype=np.uint8)
        left_img_path = "(not found)"

    # Ensure same size
    if left.shape[:2] != disp.shape[:2]:
        left = cv2.resize(left, (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_AREA)

    return {
        "pdir": pdir,
        "disp": disp,
        "left": left,
        "left_path": left_img_path,
        "disp_path": disp_path,
    }


class State:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.locked = False
        self.lx = 0
        self.ly = 0
        self.mode = "disp"  # or "depth"


def main():
    fx, baseline = read_stereo_K_and_baseline(STEREO_YAML)
    has_metric = (fx is not None and baseline is not None)
    if has_metric:
        print(f"[INFO] metric enabled: fx={fx:.3f}, baseline={baseline:.6f} m")
    else:
        print("[WARN] Cannot read fx/baseline from stereo yaml -> will show disparity only.")

    state = State()
    pair = int(START_PAIR)
    data = load_pair(pair)
    if data is None:
        raise SystemExit(f"Cannot load FS pair {pair}. Check FS_OUT_DIR={FS_OUT_DIR}")

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
        left = data["left"].copy()
        disp = data["disp"]

        H, W = disp.shape[:2]
        cx = int(np.clip(state.x, 0, W - 1))
        cy = int(np.clip(state.y, 0, H - 1))
        if state.locked:
            px, py = int(np.clip(state.lx, 0, W - 1)), int(np.clip(state.ly, 0, H - 1))
        else:
            px, py = cx, cy

        d = float(disp[py, px])
        d_valid = np.isfinite(d) and d > 0.1

        # choose render map
        if state.mode == "depth" and has_metric:
            depth = np.full_like(disp, np.inf, dtype=np.float32)
            valid = np.isfinite(disp) & (disp > 0.1)
            depth[valid] = (fx * baseline) / disp[valid]

            if AUTO_DEPTH_RANGE:
                dmin, dmax = robust_range(depth, DEPTH_P_LOW, DEPTH_P_HIGH)
            else:
                dmin, dmax = float(DEPTH_MIN_M), float(DEPTH_MAX_M)

            right = colorize_scalar_map(depth, dmin, dmax, invert=True)

            z = float((fx * baseline) / d) if d_valid else float("nan")
            msg = f"pair={pair:02d} x={px} y={py}  disp={d:.2f}px  depth={z:.3f}m  (depth range {dmin:.2f}..{dmax:.2f})"
        else:
            if AUTO_DISP_RANGE:
                vmin, vmax = robust_range(disp, DISP_P_LOW, DISP_P_HIGH)
            else:
                vmin, vmax = 0.0, float(np.nanmax(disp)) if np.isfinite(disp).any() else 1.0

            right = colorize_scalar_map(disp, vmin, vmax, invert=False)
            msg = f"pair={pair:02d} x={px} y={py}  disp={d:.2f}px  (disp range {vmin:.2f}..{vmax:.2f})"
            if has_metric and d_valid:
                z = float((fx * baseline) / d)
                msg += f"  depth={z:.3f}m"

        # draw markers
        cv2.circle(left, (px, py), 6, (0, 255, 0), 2)
        cv2.circle(right, (px, py), 6, (0, 255, 0), 2)

        vis = np.concatenate([left, right], axis=1)

        # header
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 45), (0, 0, 0), -1)
        cv2.putText(vis, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        hint = "mouse move=read | LMB=lock | RMB=unlock | n/p pair | c disp/depth | s screenshot | q quit"
        cv2.rectangle(vis, (0, vis.shape[0]-35), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
        cv2.putText(vis, hint, (10, vis.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, vis)
        key = cv2.waitKey(15) & 0xFF

        if key in [27, ord("q")]:
            break
        elif key == ord("n"):
            pair += 1
            new_data = load_pair(pair)
            if new_data is None:
                pair -= 1
            else:
                data = new_data
                state.locked = False
        elif key == ord("p"):
            pair = max(1, pair - 1)
            new_data = load_pair(pair)
            if new_data is not None:
                data = new_data
                state.locked = False
        elif key == ord("c"):
            if state.mode == "disp":
                state.mode = "depth" if has_metric else "disp"
            else:
                state.mode = "disp"
        elif key == ord("s"):
            out_path = os.path.join(data["pdir"], f"inspect_{state.mode}.png")
            cv2.imwrite(out_path, vis)
            print(f"[SAVED] {out_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
