from __future__ import annotations

import os, io, base64
import numpy as np
import cv2

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from PIL import Image


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
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _load_depth(depth_path: str) -> np.ndarray:
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth not found: {depth_path}")
    ext = os.path.splitext(depth_path)[1].lower()
    if ext == ".npy":
        depth = np.load(depth_path).astype(np.float32)
    else:
        d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise ValueError(f"Failed to read depth: {depth_path}")
        depth = d.astype(np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D array, got shape={depth.shape} from {depth_path}")
    return depth


def _rgb_to_data_uri(img_rgb: np.ndarray) -> str:
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + b64


def show_photo_and_depth_map(
    image_path: str,
    depth_path: str,
    *,
    title: str | None = None,
    renderer: str | None = "colab",

    auto_depth_range: bool = True,
    p_low: float = 2.0,
    p_high: float = 98.0,
    depth_min_m: float = 0.3,
    depth_max_m: float = 50.0,

    colorscale: str = "Turbo",
    reverse_depth_colors: bool = False,  # True => ближе "горячее" (как в твоём cv2-инспекторе)

    align: str = "resize_depth_to_image",  # "resize_depth_to_image" | "resize_image_to_depth" | "strict"
    depth_resize_interp: int = cv2.INTER_NEAREST,

    height: int = 700,
    width: int = 1200,
    show: bool = True,
):
    """
    2 панели:
      слева: фото (ОДНО)
      справа: depth heatmap
    Hover на фото показывает x,y и depth ровно по пикселю.
    """
    if renderer:
        pio.renderers.default = renderer

    img_rgb = _load_image_rgb(image_path)
    depth = _load_depth(depth_path)

    ih, iw = img_rgb.shape[:2]
    dh, dw = depth.shape[:2]

    # выравнивание размеров
    if (ih, iw) != (dh, dw):
        if align == "strict":
            raise ValueError(f"Image size {iw}x{ih} != depth size {dw}x{dh}. Set align=... to auto-resize.")
        elif align == "resize_depth_to_image":
            depth = cv2.resize(depth, (iw, ih), interpolation=depth_resize_interp)
        elif align == "resize_image_to_depth":
            img_rgb = cv2.resize(img_rgb, (dw, dh), interpolation=cv2.INTER_AREA)
            ih, iw = img_rgb.shape[:2]
        else:
            raise ValueError(f"Unknown align={align}")

    # диапазон отображения depth
    if auto_depth_range:
        vmin, vmax = _robust_range(depth, p_low, p_high)
    else:
        vmin, vmax = float(depth_min_m), float(depth_max_m)

    # для hover: nan вместо inf
    depth_hover = depth.astype(np.float32)
    depth_hover[~np.isfinite(depth_hover)] = np.nan

    # пиксельные координаты центров
    x = np.arange(iw)
    y = np.arange(ih)

    hover_tmpl = "x=%{x:.0f} y=%{y:.0f}<br>depth=%{z:.3f} m<extra></extra>"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Photo (hover → depth)", "Depth (m)"),
        horizontal_spacing=0.02,
    )

    # --- ЛЕВАЯ ПАНЕЛЬ: фото как фон осей ---
    fig.add_layout_image(
        dict(
            source=_rgb_to_data_uri(img_rgb),
            xref="x1", yref="y1",
            x=-0.5, y=-0.5,
            sizex=iw, sizey=ih,
            xanchor="left", yanchor="top",
            sizing="stretch",
            layer="below",
        )
    )

    # невидимая heatmap поверх фото для правильного hover по depth
    fig.add_trace(
        go.Heatmap(
            z=depth_hover,
            x=x, y=y,
            opacity=0.0,
            showscale=False,
            hovertemplate=hover_tmpl,
        ),
        row=1, col=1
    )

    # --- ПРАВАЯ ПАНЕЛЬ: видимая depth heatmap ---
    fig.add_trace(
        go.Heatmap(
            z=depth_hover,
            x=x, y=y,
            colorscale=colorscale,
            reversescale=bool(reverse_depth_colors),
            zmin=vmin, zmax=vmax,
            colorbar=dict(title="m"),
            hovertemplate=hover_tmpl,
        ),
        row=1, col=2
    )

    # Настройка осей (важно для 1:1 пикселей)
    # левая
    fig.update_xaxes(range=[-0.5, iw - 0.5], visible=False, showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(range=[ih - 0.5, -0.5], visible=False, showgrid=False, zeroline=False,
                     scaleanchor="x1", row=1, col=1)

    # правая
    fig.update_xaxes(range=[-0.5, iw - 0.5], visible=False, showgrid=False, zeroline=False, row=1, col=2)
    fig.update_yaxes(range=[ih - 0.5, -0.5], visible=False, showgrid=False, zeroline=False,
                     scaleanchor="x2", row=1, col=2)

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
