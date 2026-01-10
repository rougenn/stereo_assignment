# mono_depth_demo.py
from __future__ import annotations

import os
import numpy as np
import cv2

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


def _robust_range(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> tuple[float, float]:
    mask = np.isfinite(arr)
    if not np.any(mask):
        return 0.0, 1.0
    vals = arr[mask].astype(np.float32)
    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def _load_image_rgb(image_path: str) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image (cv2.imread returned None): {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def _load_depth(depth_path: str) -> np.ndarray:
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth not found: {depth_path}")

    ext = os.path.splitext(depth_path)[1].lower()
    if ext == ".npy":
        depth = np.load(depth_path).astype(np.float32)
    else:
        # fallback: попробуем прочитать как картинку (на всякий случай)
        d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise ValueError(f"Failed to read depth: {depth_path}")
        depth = d.astype(np.float32)

    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D array, got shape={depth.shape} from {depth_path}")

    return depth


def mono_neural_demonstrate(
    rectified_image_path: str,
    depth_path: str,
    *,
    title: str | None = None,
    auto_depth_range: bool = True,
    p_low: float = 2.0,
    p_high: float = 98.0,
    depth_min_m: float = 0.3,
    depth_max_m: float = 50.0,
    colorscale: str = "Turbo",
    renderer: str | None = "colab",
    height: int = 520,
    width: int = 1050,
    show: bool = True,
):
    """
    Показывает интерактивную пару:
      - слева RGB, при наведении показывается depth (tooltip)
      - справа depth heatmap
    Принимает ровно 2 пути: rectified image и depth (обычно depth_meter.npy).

    Возвращает plotly Figure.
    """

    if renderer:
        # "colab" хорошо для Google Colab, в обычном Jupyter можно поставить "notebook_connected" или "browser"
        pio.renderers.default = renderer

    img_rgb = _load_image_rgb(rectified_image_path)
    depth = _load_depth(depth_path)

    # подгоним размеры (как у вас было)
    if img_rgb.shape[:2] != depth.shape[:2]:
        img_rgb = cv2.resize(img_rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)

    # диапазон для отображения depth-heatmap
    if auto_depth_range:
        vmin, vmax = _robust_range(depth, p_low, p_high)
    else:
        vmin, vmax = float(depth_min_m), float(depth_max_m)

    # hover шаблон
    hover_tmpl = "x=%{x} y=%{y}<br>depth=%{z:.3f} m<extra></extra>"

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("RGB (hover → depth)", "Depth (m)"),
        horizontal_spacing=0.02,
    )

    # 1) RGB
    fig.add_trace(go.Image(z=img_rgb), 1, 1)

    # Невидимый слой поверх RGB, чтобы hover показывал глубину на том же пикселе
    fig.add_trace(
        go.Heatmap(
            z=depth,
            opacity=0.0,
            showscale=False,
            hovertemplate=hover_tmpl,
        ),
        1, 1
    )

    # 2) Depth
    fig.add_trace(
        go.Heatmap(
            z=depth,
            colorscale=colorscale,
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title="m"),
            hovertemplate=hover_tmpl,
        ),
        1, 2
    )

    # чтобы y шёл как в изображениях (сверху вниз)
    fig.update_yaxes(autorange="reversed", scaleanchor="x", row=1, col=1)
    fig.update_yaxes(autorange="reversed", scaleanchor="x", row=1, col=2)

    fig.update_layout(
        title=title,
        height=height,
        width=width,
        hovermode="closest",
        margin=dict(l=10, r=10, t=60 if title else 40, b=10),
    )

    if show:
        fig.show()

    return fig
