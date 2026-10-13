"""Fake phone photos of hand-drawn crease patterns, with a known answer.

To measure the photo reader honestly it needs photos where the true
pattern is known. This draws a pattern the way a person would (red marker
for mountains, blue for valleys, wobbly lines that overshoot or stop
short), puts the paper on a desk at an angle, and adds uneven light,
blur, sensor noise and JPEG compression.

The difficulty settings only scale those effects; nothing about the
pattern is passed to the reader.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from .io import normalize
from .model import Pattern

LEVELS = {
    #          tilt  corner  wobble  blur  noise  light  jpeg_lo
    "easy":   (5.0,  0.02,   0.8,    0.3,  2.0,   0.10,  85),
    "normal": (15.0, 0.05,   1.8,    0.8,  4.0,   0.22,  70),
    "hard":   (25.0, 0.08,   2.8,    1.3,  7.0,   0.35,  55),
}


def _draw_line(img, p, q, color, thick, wobble, rng):
    p, q = np.array(p, float), np.array(q, float)
    d = q - p
    length = float(np.hypot(*d))
    if length < 1:
        return
    u = d / length
    nrm = np.array([-u[1], u[0]])
    # People overshoot or stop short of where lines meet.
    p = p - u * rng.uniform(-5, 6)
    q = q + u * rng.uniform(-5, 6)
    length = float(np.hypot(*(q - p)))
    k = max(8, int(length / 15))
    t = np.linspace(0, 1, k)
    phase, freq = rng.uniform(0, 6.28), rng.uniform(0.5, 2.0)
    off = wobble * np.sin(2 * np.pi * freq * t + phase) * np.sin(np.pi * t)
    pts = p[None, :] + (q - p)[None, :] * t[:, None] + nrm[None, :] * off[:, None]
    cv2.polylines(img, [np.round(pts).astype(np.int32)], False, color, thick, cv2.LINE_AA)


def draw_sheet(pattern: Pattern, rng: random.Random, size: int = 800, wobble: float = 1.8) -> np.ndarray:
    p = normalize(pattern)
    img = np.empty((size, size, 3), np.uint8)
    img[:] = (226 + rng.randint(-6, 6), 236 + rng.randint(-5, 5), 243 + rng.randint(-4, 4))  # BGR cream
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    grain = nrng.normal(0, 2.5, (size, size, 1))
    img = np.clip(img + grain, 0, 255).astype(np.uint8)
    creases = [(a, b, k) for a, b, k in p.segments() if k in ("M", "V")]
    rng.shuffle(creases)
    for (x1, y1), (x2, y2), kind in creases:
        if kind == "M":
            color = (rng.randint(30, 70), rng.randint(25, 60), rng.randint(175, 225))
        else:
            color = (rng.randint(165, 210), rng.randint(70, 110), rng.randint(25, 60))
        thick = rng.randint(4, 8)
        _draw_line(img, (x1 * size, (1 - y1) * size), (x2 * size, (1 - y2) * size),
                   color, thick, wobble, rng)
    return img


def render_photo(pattern: Pattern, seed: int, level: str = "normal") -> bytes:
    tilt, corner, wobble, blur, noise, light, jpeg_lo = LEVELS[level]
    rng = random.Random(seed)
    S = 800
    sheet = draw_sheet(pattern, rng, S, wobble)

    W, H = int(S * 1.45), int(S * 1.45)
    desk = np.empty((H, W, 3), np.uint8)
    base = rng.choice([(60, 80, 110), (90, 100, 105), (45, 55, 70), (120, 130, 140)])
    desk[:] = base
    nrng = np.random.default_rng(seed)
    desk = np.clip(desk + nrng.normal(0, 8, (H, W, 1)), 0, 255).astype(np.uint8)

    # Where the sheet's corners land: a rotated square, each corner nudged.
    ang = np.radians(rng.uniform(-tilt, tilt))
    side = S * rng.uniform(0.78, 0.92)
    cx, cy = W / 2 + rng.uniform(-30, 30), H / 2 + rng.uniform(-30, 30)
    base_pts = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * side / 2
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = base_pts @ rot.T + [cx, cy]
    dst += np.array([[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(4)]) * corner * S
    src = np.array([[0, 0], [S, 0], [S, S], [0, S]], float)
    M = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    warped = cv2.warpPerspective(sheet, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((S, S), 255, np.uint8), M, (W, H))
    photo = np.where(mask[..., None] > 127, warped, desk)

    # Uneven light: a gradient in a random direction plus a soft shadow.
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    th = rng.uniform(0, 2 * np.pi)
    g = (np.cos(th) * (xx / W - 0.5) + np.sin(th) * (yy / H - 0.5))
    shade = 1.0 - light * (g + 0.5)
    sx, sy = rng.uniform(0, W), rng.uniform(0, H)
    shade -= light * 0.6 * np.exp(-(((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * (0.25 * W) ** 2)))
    photo = np.clip(photo * shade[..., None], 0, 255)

    if blur > 0:
        photo = cv2.GaussianBlur(photo, (0, 0), rng.uniform(0.2, blur))
    photo = np.clip(photo + nrng.normal(0, noise, photo.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(jpeg_lo, 92)])
    return buf.tobytes()


def render_digital(pattern: Pattern, seed: int = 0, size: int = 900, grid: int | None = 16) -> bytes:
    """A screenshot-style image of a pattern, as design software draws it:
    thin anti-aliased red and blue lines on white, a faint grey grid, and
    a dark frame a few pixels inside the image."""
    rng = random.Random(seed)
    p = normalize(pattern)
    pad = 6
    img = np.full((size, size, 3), 255, np.uint8)
    inner = size - 2 * pad
    if grid:
        for k in range(grid + 1):
            v = pad + round(k * inner / grid)
            cv2.line(img, (v, pad), (v, size - pad), (232, 232, 232), 1)
            cv2.line(img, (pad, v), (size - pad, v), (232, 232, 232), 1)
    cv2.rectangle(img, (pad, pad), (size - pad, size - pad), (40, 40, 40), 2)
    thick = rng.choice((2, 2, 3))
    for (x1, y1), (x2, y2), kind in p.segments():
        if kind not in ("M", "V"):
            continue
        color = (90, 90, 255) if kind == "M" else (255, 100, 100)    # BGR: pink-red, light blue
        a = (pad + x1 * inner, pad + (1 - y1) * inner)
        b = (pad + x2 * inner, pad + (1 - y2) * inner)
        cv2.line(img, (round(a[0] * 16), round(a[1] * 16)), (round(b[0] * 16), round(b[1] * 16)),
                 color, thick, cv2.LINE_AA, shift=4)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()
