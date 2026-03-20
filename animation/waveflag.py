#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "Pillow>=9.0.0",
#   "numpy>=1.20.0",
# ]
# ///
"""
waveflag.py — Convert a still image into an animated waving-flag GIF or APNG.

Requirements:
    pip install Pillow numpy

Usage:
    python waveflag.py FLAG.png output.gif
    python waveflag.py FLAG.png output.png --style ripple --fireworks
    python waveflag.py logo.png output.gif --border-color black --amplitude 20
    python waveflag.py photo.jpg output.png --sparkles twinkle --frames 30 --fps 24

Output format is inferred from the file extension of OUTPUT:
    .gif          Animated GIF  (256 colours, binary transparency)
    .png / .apng  Animated PNG  (full 32-bit RGBA, better quality)

Wave styles (default: diagonal):
    diagonal  Diagonally spreading ripple — wave crests run diagonally so the
              right-hand free edge flutters realistically.  Default.
    wave      Classic horizontal sinusoidal wave, pole fixed at left edge.
    ripple    2-D ripple: primary vertical wave + subtle horizontal wriggle.
    flutter   Cloth flutter — two superimposed high-frequency waves.
    fold      Crumpled-fold look: horizontal pinch plus vertical wave.

Sparkle styles (--sparkles):
    twinkle   Star-shaped sparkles that appear, brighten, and fade in-place.
    burst     8-point starburst sparks with a sharp flash and slow fade.
    drift     Stars that drift upward while twinkling.

Border:
    A thin outline is drawn around the warped silhouette.  Default "auto"
    picks white on dark images, black on light images.

Fireworks:
    --fireworks composites a particle-burst animation behind the flag.
    Works in both GIF (binary transparency) and APNG (smooth alpha fades).

Clipping:
    The output canvas is taller than the source by 2 x pad so displacement
    never clips.  For wide flags (wider than tall), this gives the flag room
    to move up and down within its bounding box without any cropping.
"""

from __future__ import annotations

import argparse
import colorsys
import math
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WAVE_STYLES    = ("diagonal", "wave", "ripple", "flutter", "fold")
SPARKLE_STYLES = ("twinkle", "burst", "drift")

_WAVE_DOCS = {
    "diagonal": "diagonally spreading ripple, right edge flutters (default)",
    "wave":     "smooth sinusoidal wave, pole fixed at left edge",
    "ripple":   "2-D ripple with both vertical and horizontal displacement",
    "flutter":  "cloth flutter — two superimposed high-frequency waves",
    "fold":     "crumpled fold — horizontal pinch + vertical wave",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input",  metavar="INPUT",  help="Source image (JPEG/PNG/…)")
    p.add_argument("output", metavar="OUTPUT", help="Output animation (.gif / .png / .apng)")

    g = p.add_argument_group("wave")
    g.add_argument(
        "--style", choices=WAVE_STYLES, default="diagonal", metavar="STYLE",
        help=("Wave style: "
              + "  ".join(f"{k} ({v})" for k, v in _WAVE_DOCS.items())
              + "  (default: diagonal)"),
    )
    g.add_argument("--amplitude", type=float, default=12.0, metavar="PX",
                   help="Peak displacement in pixels (default: 12)")
    g.add_argument("--frequency", type=float, default=1.5,  metavar="N",
                   help="Spatial cycles across the flag width (default: 1.5)")
    g.add_argument("--speed",     type=float, default=0.6,  metavar="N",
                   help=("Wave cycles per second.  The frame count is auto-adjusted "
                         "to the nearest multiple of round(fps/speed) so the "
                         "animation loops without a jerk regardless of this value. "
                         "(default: 0.6)"))

    g = p.add_argument_group("animation")
    g.add_argument("--frames", type=int, default=24, metavar="N",
                   help="Number of frames (default: 24)")
    g.add_argument("--fps",    type=int, default=20, metavar="N",
                   help="Frames per second (default: 20)")
    g.add_argument("--size",   metavar="WxH",
                   help="Resize the input image before processing, e.g. 320x200")

    g = p.add_argument_group("border")
    g.add_argument(
        "--border-color", default="auto", metavar="COLOR",
        help="Outline colour: auto (default), white, black, none, or #rrggbb",
    )
    g.add_argument("--border-width", type=int, default=2, metavar="PX",
                   help="Outline thickness in pixels (default: 2)")

    g = p.add_argument_group("extras")
    g.add_argument("--fireworks", action="store_true",
                   help="Composite animated fireworks behind the flag")
    g.add_argument(
        "--sparkles", choices=SPARKLE_STYLES, default=None, metavar="STYLE",
        help="Star sparkles over the front: " + ", ".join(SPARKLE_STYLES),
    )

    return p


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------

def mean_luminance(img: Image.Image) -> float:
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    return float((0.299 * a[:, :, 0]
                + 0.587 * a[:, :, 1]
                + 0.114 * a[:, :, 2]).mean())


def resolve_border_color(
    spec: str, src: Image.Image
) -> Optional[tuple[int, int, int]]:
    s = spec.strip().lower()
    if s == "none":
        return None
    if s == "auto":
        return (255, 255, 255) if mean_luminance(src) < 128 else (0, 0, 0)
    if s == "white":
        return (255, 255, 255)
    if s == "black":
        return (0, 0, 0)
    if len(s) == 7 and s[0] == "#":
        try:
            return (int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))
        except ValueError:
            pass
    sys.exit(f"Unknown colour {spec!r}.  Use: auto / white / black / none / #rrggbb")


# ---------------------------------------------------------------------------
# Warp engine
# ---------------------------------------------------------------------------
#
# Inverse-warp formula derivation
# ================================
# Forward model: source pixel (sx, sy) moves to canvas output position
#
#     ox = sx + dx(sx, sy)
#     oy = sy + pad + dy(sx, sy)        ← '+pad' centres flag vertically
#
# Inverse (approximate, treating displacement as function of output coords):
#
#     map_x[oy, ox] = ox - dx(ox, oy)
#     map_y[oy, ox] = oy - pad - dy(ox, oy)
#
# Canvas height = h + 2*pad.  Pixels whose source coords fall outside
# [0,w) x [0,h) are returned as transparent.  Because pad >= amplitude,
# even maximum-displacement edge pixels always land within the canvas.

def make_warp_maps(
    w: int, h: int, pad: int,
    frame: int, frames_per_cycle: int,
    style: str, amplitude: float, frequency: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return inverse-warp maps ``(map_x, map_y)`` with shape ``(h + 2*pad, w)``.

    Loop invariant: phase advances by exactly 2pi every ``frames_per_cycle``
    frames, so any frame count that is a multiple of ``frames_per_cycle``
    produces a perfectly seamless loop — no jerk at the wrap point.

    ``frames_per_cycle`` is computed in ``build_frames`` as
    ``round(fps / speed)`` (minimum 2), and the total frame count is rounded
    up to the nearest multiple of ``frames_per_cycle`` before rendering begins.
    """
    OH    = h + 2 * pad
    # phase in [0, 2pi) per wave cycle; resets exactly every frames_per_cycle frames
    phase = (frame / frames_per_cycle) * 2.0 * math.pi

    # Output-pixel coordinate grids, shape (OH, w)
    gx, gy = np.meshgrid(
        np.arange(w,  dtype=np.float32),
        np.arange(OH, dtype=np.float32),
    )

    # x-envelope: 0 at pole (left edge) → 1 at free edge (right edge)
    env = gx / max(w - 1, 1)

    # Normalised y within the flag area, clamped to [0, 1]
    yn = np.clip((gy - pad) / max(h - 1, 1), 0.0, 1.0)

    TAU = 2.0 * math.pi

    if style == "diagonal":
        # Diagonal ripple: adding a yn term to the spatial phase tilts the
        # wave crests diagonally, matching the look of a real flag in wind.
        # coefficient 0.30 → ~108° phase shift top-to-bottom.
        dy = amplitude * env * np.sin(TAU * frequency * (env + 0.30 * yn) - phase)
        dx = np.zeros_like(dy)

    elif style == "wave":
        # Pure vertical wave; crests are vertical lines.
        dy = amplitude * env * np.sin(TAU * frequency * env - phase)
        dx = np.zeros_like(dy)

    elif style == "ripple":
        # Primary vertical wave + smaller horizontal wriggle at a different
        # spatial frequency so the two axes feel decoupled.
        dy = amplitude       * env * np.sin(TAU * frequency       * env - phase)
        dx = amplitude * 0.2 * env * np.sin(TAU * frequency * 0.7 * env - phase + 1.1)

    elif style == "flutter":
        # Two incommensurable frequencies → irregular, cloth-like motion.
        dy = (amplitude * 0.55 * env * np.sin(TAU * frequency * 1.8 * env - phase)
            + amplitude * 0.45 * env * np.sin(TAU * frequency * 3.5 * env - phase * 2.2))
        dx = np.zeros_like(dy)

    elif style == "fold":
        # Horizontal pinch + stretched vertical wave → stiff fabric folding.
        dx = amplitude * 0.35 * env * np.sin(TAU * frequency       * env - phase + 0.7)
        dy = amplitude * 0.75 * env * np.sin(TAU * frequency * 1.1 * env - phase)

    else:
        sys.exit(f"Unknown style {style!r}.  Choices: {WAVE_STYLES}")

    return gx - dx, gy - pad - dy


def warp_rgba(
    src: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """
    Bilinear inverse-warp of RGBA *src* (H x W x 4, uint8).

    *map_x* / *map_y* may have a larger shape than *src*; pixels whose source
    coordinates fall outside [0,W) x [0,H) become fully-transparent black.
    """
    h, w   = src.shape[:2]
    src_f  = src.astype(np.float32)

    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = (map_x - x0)[..., np.newaxis]   # fractional parts (OH, OW, 1)
    fy = (map_y - y0)[..., np.newaxis]

    oob = (map_x < 0) | (map_x > w - 1) | (map_y < 0) | (map_y > h - 1)

    x0c = np.clip(x0, 0, w - 1);  x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1);  y1c = np.clip(y1, 0, h - 1)

    c00 = src_f[y0c, x0c];  c10 = src_f[y0c, x1c]
    c01 = src_f[y1c, x0c];  c11 = src_f[y1c, x1c]

    blended = (c00 * (1 - fx) * (1 - fy)
             + c10 *      fx  * (1 - fy)
             + c01 * (1 - fx) *      fy
             + c11 *      fx  *      fy)

    out      = np.rint(blended).astype(np.uint8)
    out[oob] = 0
    return out


# ---------------------------------------------------------------------------
# Border / outline
# ---------------------------------------------------------------------------

def apply_border(
    warped: np.ndarray,
    color: tuple[int, int, int],
    width: int,
) -> np.ndarray:
    """Morphologically dilate the alpha silhouette to draw a *width*-px outline."""
    alpha_img = Image.fromarray(warped[:, :, 3], "L")
    dilated   = alpha_img.filter(ImageFilter.MaxFilter(size=2 * width + 1))

    d  = np.asarray(dilated)
    a  = warped[:, :, 3]
    bm = (d > 0) & (a == 0)   # ring between dilation and original alpha

    out = warped.copy()
    out[bm, 0] = color[0]
    out[bm, 1] = color[1]
    out[bm, 2] = color[2]
    out[bm, 3] = 255
    return out


# ---------------------------------------------------------------------------
# Fireworks particle system
# ---------------------------------------------------------------------------

class _Firework:
    GRAVITY = 0.18
    DRAG    = 0.97
    N_PARTS = 50

    def __init__(self, w: int, h: int) -> None:
        self.cx  = random.uniform(0.10 * w, 0.90 * w)
        self.cy  = random.uniform(0.04 * h, 0.50 * h)
        hue      = random.random()
        self.rgb = tuple(int(v * 255) for v in colorsys.hsv_to_rgb(hue, 1.0, 1.0))
        self.pts: list[list[float]] = []
        for _ in range(self.N_PARTS):
            angle = random.uniform(0, 2 * math.pi)
            spd   = max(0.4, random.gauss(3.2, 1.0))
            self.pts.append([self.cx, self.cy,
                              spd * math.cos(angle), spd * math.sin(angle), 1.0])
        self.age     = 0
        self.max_age = random.randint(16, 30)

    @property
    def alive(self) -> bool:
        return self.age < self.max_age

    def tick(self) -> None:
        self.age += 1
        t = self.age / self.max_age
        for p in self.pts:
            p[0] += p[2];  p[1] += p[3]
            p[3] += self.GRAVITY
            p[2] *= self.DRAG;  p[3] *= self.DRAG
            p[4]  = max(0.0, 1.0 - t ** 1.3)

    def draw(self, draw: ImageDraw.ImageDraw, gif_mode: bool) -> None:
        r, g, b = self.rgb  # type: ignore[misc]
        for p in self.pts:
            life = p[4]
            if life <= 0 or (gif_mode and life < 0.30):
                continue
            alpha = 255 if gif_mode else int(life * 220)
            x, y  = int(p[0]), int(p[1])
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(r, g, b, alpha))


class _FireworkSystem:
    def __init__(self, w: int, h: int, n_frames: int) -> None:
        self.w, self.h = w, h
        self.spawn_p   = min(0.5, 8.0 / max(n_frames, 1))
        self.active: list[_Firework] = []
        for _ in range(random.randint(1, 3)):
            fw = _Firework(w, h)
            for _ in range(random.randint(0, max(0, fw.max_age - 1))):
                fw.tick()
            if fw.alive:
                self.active.append(fw)

    def tick(self) -> None:
        for fw in self.active:
            fw.tick()
        self.active = [fw for fw in self.active if fw.alive]
        if random.random() < self.spawn_p:
            self.active.append(_Firework(self.w, self.h))

    def render(self, size: tuple[int, int], gif_mode: bool) -> Image.Image:
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        draw  = ImageDraw.Draw(layer, "RGBA")
        for fw in self.active:
            fw.draw(draw, gif_mode)
        return layer


# ---------------------------------------------------------------------------
# Sparkle system
# ---------------------------------------------------------------------------

def _draw_4star(
    draw: ImageDraw.ImageDraw,
    cx: float, cy: float,
    r_outer: float, r_inner: float,
    angle_offset: float,
    color: tuple,
) -> None:
    """Draw a 4-point star polygon centred at (cx, cy)."""
    pts = []
    for i in range(8):
        r = r_outer if i % 2 == 0 else r_inner
        a = math.pi * i / 4 + angle_offset
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    draw.polygon(pts, fill=color)


_SPARKLE_COLORS: list[tuple[int, int, int]] = [
    (255, 255, 220),   # warm white
    (220, 240, 255),   # cool blue-white
    (255, 255, 160),   # bright yellow
    (255, 210, 255),   # soft pink
    (180, 255, 255),   # cyan
    (255, 255, 255),   # pure white
]


class _SparkleSystem:
    """
    Pre-seeded star sparkles composited over the flag on every frame.

    All sparkles loop seamlessly: their brightness envelopes are based on
    ``frame / n_frames`` so frame 0 == frame n_frames.

    Sparkle positions are generated within the flag area of the padded canvas
    (y ∈ [pad, pad+flag_h)).
    """

    def __init__(
        self,
        flag_w: int, flag_h: int,
        canvas_h: int, pad: int,
        n_frames: int, style: str,
    ) -> None:
        self.flag_w   = flag_w
        self.canvas_h = canvas_h
        self.pad      = pad
        self.flag_h   = flag_h
        self.n_frames = n_frames
        self.style    = style

        n = max(8, min(40, (flag_w * flag_h) // 500))

        self.sparkles: list[dict] = []
        for _ in range(n):
            x = random.uniform(0.05 * flag_w, 0.95 * flag_w)
            y = random.uniform(pad + 0.05 * flag_h, pad + 0.95 * flag_h)
            sp: dict = {
                "x0":    x,
                "y0":    y,
                "color": random.choice(_SPARKLE_COLORS),
                "max_r": random.uniform(3.5, 9.0),
                "phase": random.uniform(0.0, 2.0 * math.pi),
                "period": random.uniform(0.4, 2.2),
            }
            if style == "drift":
                sp["vx"] = random.uniform(-0.25, 0.25)
                sp["vy"] = random.uniform(-1.0,  -0.3)   # upward
            self.sparkles.append(sp)

    def get_layer(self, frame: int, size: tuple[int, int], gif_mode: bool) -> Image.Image:
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        draw  = ImageDraw.Draw(layer, "RGBA")
        t     = frame / max(self.n_frames, 1)   # ∈ [0, 1)

        for sp in self.sparkles:
            color  = sp["color"]
            phase  = sp["phase"]
            period = sp["period"]
            max_r  = sp["max_r"]

            if self.style == "twinkle":
                raw = math.sin(2 * math.pi * period * t + phase)
                b   = max(0.0, raw) ** 1.5        # positive half only, nonlinear
                if b < 0.04:
                    continue
                r     = max_r * b
                alpha = 255 if gif_mode else int(b * 255)
                col   = (*color, alpha)
                _draw_4star(draw, sp["x0"], sp["y0"], r, r * 0.28, -math.pi / 4, col)
                cr = r * 0.28
                draw.ellipse(
                    [sp["x0"] - cr, sp["y0"] - cr, sp["x0"] + cr, sp["y0"] + cr],
                    fill=col,
                )

            elif self.style == "burst":
                raw = math.sin(2 * math.pi * period * t + phase)
                b   = max(0.0, raw) ** 0.6        # sharper attack than twinkle
                if b < 0.04:
                    continue
                r     = max_r * b
                alpha = 255 if gif_mode else int(b * 240)
                col   = (*color, alpha)
                # Two 4-point stars at 45° offset → 8-point starburst
                _draw_4star(draw, sp["x0"], sp["y0"], r,        r * 0.18, -math.pi / 4, col)
                _draw_4star(draw, sp["x0"], sp["y0"], r * 0.75, r * 0.18, 0.0,          col)
                cr = r * 0.25
                draw.ellipse(
                    [sp["x0"] - cr, sp["y0"] - cr, sp["x0"] + cr, sp["y0"] + cr],
                    fill=(255, 255, 255, alpha),
                )

            elif self.style == "drift":
                steps = t * self.n_frames
                x     = (sp["x0"] + sp["vx"] * steps) % self.flag_w
                y_raw = sp["y0"] + sp["vy"] * steps
                # Wrap y within the flag area of the canvas
                top = self.pad
                bot = self.pad + self.flag_h
                fh  = bot - top
                y   = top + (y_raw - top) % fh if fh > 0 else y_raw
                raw = (math.sin(2 * math.pi * period * t + phase) + 1) / 2
                b   = raw ** 1.3
                r   = max_r * max(0.25, b)
                alpha = 255 if gif_mode else int(b * 220)
                col   = (*color, alpha)
                _draw_4star(draw, x, y, r, r * 0.28, -math.pi / 4, col)

        return layer


# ---------------------------------------------------------------------------
# GIF palette conversion
# ---------------------------------------------------------------------------

_GIF_TRANSP = 255   # palette slot reserved for transparent pixels


def _rgba_to_gif_palette(rgba: Image.Image) -> Image.Image:
    alpha = np.asarray(rgba.split()[3])
    q     = rgba.convert("RGB").quantize(colors=_GIF_TRANSP, dither=0)
    pal   = list(q.getpalette())[: _GIF_TRANSP * 3] + [0, 0, 0]
    arr   = np.asarray(q, dtype=np.uint8).copy()
    arr[alpha < 128] = _GIF_TRANSP
    out   = Image.fromarray(arr, "P")
    out.putpalette(pal)
    return out


# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------

def build_frames(src: Image.Image, args: argparse.Namespace) -> list[Image.Image]:
    """
    Build all animation frames.

    Canvas height  =  flag_h + 2 * pad.
    The warp maps are generated in output space, so the '-pad' in the
    map_y formula already centres the flag.  There is no separate paste step:
    ``warp_rgba`` produces an array exactly (canvas_h, flag_w) that can be
    alpha-composited directly onto the canvas.
    """
    w, h   = src.size
    pad    = int(args.amplitude) + args.border_width + 4
    CW, CH = w, h + 2 * pad

    src_arr    = np.asarray(src.convert("RGBA"), dtype=np.uint8).copy()
    border_col = resolve_border_color(args.border_color, src)
    is_gif     = Path(args.output).suffix.lower() == ".gif"

    # ── Guarantee a seamless loop ─────────────────────────────────────────────
    # One wave cycle occupies exactly frames_per_cycle frames.  The total frame
    # count is rounded up to the nearest multiple so the animation always ends
    # one frame's motion before the start — no jerk at the wrap point.
    # Derivation: phase(frame) = frame/frames_per_cycle * 2π
    #   → phase(frames_per_cycle) = 2π ≡ phase(0)  ✓
    #   → any integer multiple of frames_per_cycle also loops cleanly  ✓
    frames_per_cycle = max(2, round(args.fps / args.speed))
    n_cycles         = max(1, math.ceil(args.frames / frames_per_cycle))
    n_frames         = n_cycles * frames_per_cycle
    if n_frames != args.frames:
        print(f"  (frames {args.frames} → {n_frames}: "
              f"{n_cycles} cycle(s) × {frames_per_cycle} frames/cycle "
              f"for seamless loop)")

    fw_sys = (_FireworkSystem(CW, CH, n_frames) if args.fireworks  else None)
    sp_sys = (
        _SparkleSystem(w, h, CH, pad, n_frames, args.sparkles)
        if args.sparkles else None
    )

    frames: list[Image.Image] = []

    for i in range(n_frames):
        print(f"\r  Building frame {i + 1}/{n_frames} …", end="", flush=True)

        # Warp maps have shape (CH, CW) — the full padded canvas height
        mx, my = make_warp_maps(
            w, h, pad, i, frames_per_cycle,
            args.style, args.amplitude, args.frequency,
        )

        # warped is already (CH, CW, 4): no paste-with-offset needed
        warped = warp_rgba(src_arr, mx, my)

        if border_col is not None:
            warped = apply_border(warped, border_col, args.border_width)

        canvas = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))

        # Fireworks behind the flag
        if fw_sys is not None:
            fw_sys.tick()
            canvas = Image.alpha_composite(canvas, fw_sys.render((CW, CH), is_gif))

        # Flag (already correct size — alpha_composite, not paste)
        canvas = Image.alpha_composite(canvas, Image.fromarray(warped, "RGBA"))

        # Sparkles in front of the flag
        if sp_sys is not None:
            canvas = Image.alpha_composite(
                canvas, sp_sys.get_layer(i, (CW, CH), is_gif))

        frames.append(canvas)

    print()
    return frames


# ---------------------------------------------------------------------------
# Output savers
# ---------------------------------------------------------------------------

def _save_apng(frames: list[Image.Image], path: str, fps: int) -> None:
    ms = int(1000 / fps)
    frames[0].save(
        path, format="PNG", save_all=True, append_images=frames[1:],
        loop=0, duration=ms,
    )
    print(f"✓  APNG saved:  {path}  ({len(frames)} frames @ {fps} fps, {ms} ms/frame)")


def _save_gif(frames: list[Image.Image], path: str, fps: int) -> None:
    ms = int(1000 / fps)
    print("  Converting to palette …", end="", flush=True)
    pal = [_rgba_to_gif_palette(f) for f in frames]
    print("\r  Writing GIF …          ", end="", flush=True)
    pal[0].save(
        path, format="GIF", save_all=True, append_images=pal[1:],
        loop=0, duration=ms,
        transparency=_GIF_TRANSP, disposal=2, optimize=False,
    )
    print(f"\r✓  GIF saved:   {path}  ({len(frames)} frames @ {fps} fps, {ms} ms/frame)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    if Path(args.input).resolve() == Path(args.output).resolve():
        sys.exit("Error: input and output paths are the same file")

    try:
        src = Image.open(args.input)
    except FileNotFoundError:
        sys.exit(f"Error: cannot open {args.input!r}")

    # If the input is animated, use only the first frame
    try:
        src.seek(0)
    except (AttributeError, EOFError):
        pass
    src = src.copy()

    if args.size:
        try:
            rw, rh = (int(d) for d in args.size.lower().split("x", 1))
        except ValueError:
            sys.exit("Error: --size must be WxH, e.g. 320x200")
        src = src.resize((rw, rh), Image.LANCZOS)

    print(f"Input : {args.input}  ({src.width}×{src.height})")
    print(f"Output: {args.output}")
    print(f"  style={args.style}  amplitude={args.amplitude}  "
          f"frequency={args.frequency}  speed={args.speed}")
    print(f"  frames={args.frames}  fps={args.fps}  "
          f"border={args.border_color}/{args.border_width}px"
          + ("  fireworks=yes"          if args.fireworks else "")
          + (f"  sparkles={args.sparkles}" if args.sparkles   else ""))

    frames = build_frames(src, args)

    ext = Path(args.output).suffix.lower()
    if ext == ".gif":
        _save_gif(frames, args.output, args.fps)
    elif ext in (".png", ".apng"):
        _save_apng(frames, args.output, args.fps)
    else:
        sys.exit(
            f"Error: unknown output extension {ext!r}.  Use .gif, .png, or .apng"
        )


if __name__ == "__main__":
    main()
