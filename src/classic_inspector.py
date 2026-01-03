import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Где лежат результаты classic_depth_from_stereo.py
BASE_OUT_DIR = os.path.join(PROJECT_ROOT, "classic_depth_outputs")

# Пара для старта (1..13)
START_PAIR = 1

# Имена файлов внутри папки пары (01/02/...)
LEFT_IMAGE_CANDIDATES = ["left_rect.png", "left.png", "left.jpg", "left.jpeg"]
DEPTH_NPY_NAME = "depth_meter.npy"   # то, что сохранял classic скрипт
DISP_NPY_NAME = "disp.npy"           # опционально, если хочешь отображать disparity

# Визуализация глубины
AUTO_RANGE = True          # True = диапазон по процентилям, False = по MIN/MAX ниже
RANGE_P_LOW = 2.0          # процентиль ближних (например 2%)
RANGE_P_HIGH = 98.0        # процентиль дальних (например 98%)
MIN_DEPTH_M = 0.3          # используется если AUTO_RANGE=False
MAX_DEPTH_M = 20.0         # используется если AUTO_RANGE=False

# Колормэп (TURBO красивый; JET тоже ок)
COLORMAP = cv2.COLORMAP_TURBO

# Управление
WINDOW_NAME = "Depth Inspector"
FONT_SCALE = 0.7
TEXT_THICKNESS = 2

# Клавиши:
#   q / ESC : выход
#   n       : следующая пара
#   p       : предыдущая пара
#   s       : сохранить скриншот в папку пары


def pair_dir(pair_idx: int) -> str:
    return os.path.join(BASE_OUT_DIR, f"{pair_idx:02d}")


def load_left_image(pdir: str):
    for name in LEFT_IMAGE_CANDIDATES:
        path = os.path.join(pdir, name)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                return img, path
    return None, None


def load_depth(pdir: str):
    path = os.path.join(pdir, DEPTH_NPY_NAME)
    if not os.path.exists(path):
        return None, None
    depth = np.load(path)
    return depth, path


def compute_depth_range(depth: np.ndarray):
    finite = np.isfinite(depth) & (depth > 0)
    if not np.any(finite):
        return (0.0, 1.0)

    vals = depth[finite].astype(np.float32)

    if AUTO_RANGE:
        lo = float(np.percentile(vals, RANGE_P_LOW))
        hi = float(np.percentile(vals, RANGE_P_HIGH))
    else:
        lo, hi = float(MIN_DEPTH_M), float(MAX_DEPTH_M)

    if hi <= lo:
        hi = lo + 1e-3
    return lo, hi


def colorize_depth(depth: np.ndarray, dmin: float, dmax: float):
    """
    Делает цветную карту глубины:
    ближе (меньше Z) -> более "горячие" цвета.
    """
    depth_vis = depth.copy().astype(np.float32)
    depth_vis[~np.isfinite(depth_vis)] = dmax

    depth_vis = np.clip(depth_vis, dmin, dmax)

    # normalize 0..1
    norm = (depth_vis - dmin) / (dmax - dmin)
    # invert so near -> high intensity
    norm = 1.0 - norm

    img8 = (norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(img8, COLORMAP)
    return colored


class UIState:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.locked = False
        self.lock_x = 0
        self.lock_y = 0


def main():
    state = UIState()
    pair = int(START_PAIR)

    def load_pair(pair_idx: int):
        pdir = pair_dir(pair_idx)
        if not os.path.isdir(pdir):
            return None

        left_img, left_path = load_left_image(pdir)
        depth, depth_path = load_depth(pdir)

        if left_img is None:
            print(f"[ERR] cannot find left image in {pdir}. Tried: {LEFT_IMAGE_CANDIDATES}")
            return None
        if depth is None:
            print(f"[ERR] cannot find depth in {pdir} ({DEPTH_NPY_NAME})")
            return None

        dmin, dmax = compute_depth_range(depth)
        depth_col = colorize_depth(depth, dmin, dmax)

        return {
            "pdir": pdir,
            "left_img": left_img,
            "depth": depth,
            "depth_col": depth_col,
            "left_path": left_path,
            "depth_path": depth_path,
            "dmin": dmin,
            "dmax": dmax,
        }

    data = load_pair(pair)
    if data is None:
        raise SystemExit(f"Cannot load pair {pair}. Check BASE_OUT_DIR={BASE_OUT_DIR} and that outputs exist.")

    def on_mouse(event, x, y, flags, userdata):
        nonlocal state
        if event == cv2.EVENT_MOUSEMOVE:
            state.x, state.y = x, y
        elif event == cv2.EVENT_LBUTTONDOWN:
            state.locked = True
            state.lock_x, state.lock_y = x, y
        elif event == cv2.EVENT_RBUTTONDOWN:
            state.locked = False

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        left = data["left_img"].copy()
        depth_col = data["depth_col"].copy()
        depth = data["depth"]

        H, W = left.shape[:2]
        # координаты курсора в границах
        cx = int(np.clip(state.x, 0, W - 1))
        cy = int(np.clip(state.y, 0, H - 1))

        # если кликнут — показываем lock точку, иначе курсор
        if state.locked:
            px, py = int(np.clip(state.lock_x, 0, W - 1)), int(np.clip(state.lock_y, 0, H - 1))
        else:
            px, py = cx, cy

        z = float(depth[py, px])
        valid = np.isfinite(z) and z > 0

        # рисуем маркер
        cv2.circle(left, (px, py), 6, (0, 255, 0), 2)
        cv2.circle(depth_col, (px, py), 6, (0, 255, 0), 2)

        # текст
        if valid:
            msg = f"pair={pair:02d}  x={px} y={py}  depth={z:.3f} m   (range {data['dmin']:.2f}..{data['dmax']:.2f} m)"
        else:
            msg = f"pair={pair:02d}  x={px} y={py}  depth=INVALID   (range {data['dmin']:.2f}..{data['dmax']:.2f} m)"

        # объединяем в один экран: [left | depth]
        vis = np.concatenate([left, depth_col], axis=1)

        # фон под текст
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(
            vis, msg, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
            (255, 255, 255), TEXT_THICKNESS, cv2.LINE_AA
        )

        hint = "Mouse: move=read  LMB=lock  RMB=unlock | keys: n/p pair, s screenshot, q quit"
        cv2.rectangle(vis, (0, vis.shape[0]-35), (vis.shape[1], vis.shape[0]), (0, 0, 0), -1)
        cv2.putText(
            vis, hint, (10, vis.shape[0]-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA
        )

        cv2.imshow(WINDOW_NAME, vis)
        key = cv2.waitKey(15) & 0xFF

        if key in [27, ord('q')]:
            break

        elif key == ord('n'):
            pair += 1
            new_data = load_pair(pair)
            if new_data is None:
                pair -= 1
            else:
                data = new_data
                state.locked = False

        elif key == ord('p'):
            pair = max(1, pair - 1)
            new_data = load_pair(pair)
            if new_data is not None:
                data = new_data
                state.locked = False

        elif key == ord('s'):
            out_path = os.path.join(data["pdir"], "inspect_screenshot.png")
            cv2.imwrite(out_path, vis)
            print(f"[SAVED] {out_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
