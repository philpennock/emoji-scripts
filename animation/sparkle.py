#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "Pillow>=9.0.0",
#   "numpy>=1.20.0",
# ]
# ///
"""
sparkle.py — Add animated sparkle effects to a still or animated image.

Works on both static images (producing an animated output) and already-animated
GIFs/APNGs (compositing sparkles on top of existing frames).

Requirements:
    pip install Pillow numpy

Usage:
    python sparkle.py bread.png sparkly_bread.gif
    python sparkle.py bread.png sparkly_bread.png --style shimmer
    python sparkle.py animated.gif with_sparkles.gif --style glitter --density 1.5
    python sparkle.py logo.png sparkling.png --style twinkle --frames 30 --fps 24

Output format is inferred from the file extension of OUTPUT:
    .gif          Animated GIF  (256 colours, binary transparency)
    .png / .apng  Animated PNG  (full 32-bit RGBA, better quality)

Sparkle styles:
    twinkle   Classic 4-point star sparkles that appear, brighten, and fade
              in-place.  The quintessential "✨ freshly baked" gleam.  Default.
    burst     8-point starburst sparks with a sharp flash and slow fade.
              More dramatic, like a camera flash catching facets.
    drift     Stars that drift upward while twinkling, like rising motes.
    shimmer   A bright diagonal band that sweeps across the image, with
              small sparkles popping along the wavefront.  Gives a "sheen"
              or "polish" look.
    glitter   Dense small sparkles that randomly flash like fine glitter
              catching light.  Quick, staccato twinkles.

Library usage:
    The sparkle system is designed to be importable by other animation scripts.
    Two approaches are supported — see "Library usage" below.

    Approach 1 — High-level helper (simplest):

        from animation.sparkle import add_sparkles

        # frames is a list of PIL RGBA Images, all the same size
        sparkled = add_sparkles(frames, style="twinkle", density=1.0,
                                fps=20, gif_mode=False)

    Approach 2 — Direct SparkleSystem (full control):

        from animation.sparkle import SparkleSystem

        sys = SparkleSystem(width=128, height=128, n_frames=24,
                            style="shimmer", density=1.0)
        for i in range(24):
            layer = sys.get_layer(i, (128, 128), gif_mode=False)
            frame = Image.alpha_composite(base_frames[i], layer)

    For standalone use without installing the package, run via:
        uv run --script animation/sparkle.py INPUT OUTPUT [options]
"""

from __future__ import annotations

import argparse
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

SPARKLE_STYLES = ("twinkle", "burst", "drift", "shimmer", "glitter")
SIZE_VARY_MODES = ("random", "fixed", "crescendo")

_STYLE_DOCS = {
    "twinkle": "classic 4-point star sparkles, brightening and fading (default)",
    "burst":   "8-point starburst with sharp flash and slow fade",
    "drift":   "stars drift upward while twinkling",
    "shimmer": "sweeping diagonal highlight band with sparkles along the wavefront",
    "glitter": "dense small sparkles that randomly flash like fine glitter",
}

_SIZE_VARY_DOCS = {
    "random":    "each sparkle gets a random size between 40%% and 100%% of max (default)",
    "fixed":     "all sparkles use the same max size",
    "crescendo": "sparkles grow larger over the animation cycle, then reset",
}

_SPARKLE_COLORS: list[tuple[int, int, int]] = [
    (255, 255, 220),   # warm white
    (220, 240, 255),   # cool blue-white
    (255, 255, 160),   # bright yellow
    (255, 210, 255),   # soft pink
    (180, 255, 255),   # cyan
    (255, 255, 255),   # pure white
]


# ---------------------------------------------------------------------------
# Size specification parsing
# ---------------------------------------------------------------------------

def parse_size_spec(spec: str, image_dim: int) -> float:
    """
    Parse a sparkle size specification into a pixel value.

    Accepts:
        "20px"  → 20.0  (absolute pixels)
        "15%"   → 0.15 * image_dim  (percentage of min(width, height))
        "8"     → 8.0  (bare number treated as pixels)
    """
    spec = spec.strip()
    if spec.endswith("px"):
        return float(spec[:-2])
    if spec.endswith("%"):
        return float(spec[:-1]) / 100.0 * image_dim
    return float(spec)


# ---------------------------------------------------------------------------
# Drawing primitives
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


def _draw_cross(
    draw: ImageDraw.ImageDraw,
    cx: float, cy: float,
    length: float,
    thickness: float,
    color: tuple,
) -> None:
    """Draw a thin cross (+ shape) centred at (cx, cy)."""
    ht = thickness / 2
    hl = length / 2
    # horizontal bar
    draw.rectangle([cx - hl, cy - ht, cx + hl, cy + ht], fill=color)
    # vertical bar
    draw.rectangle([cx - ht, cy - hl, cx + ht, cy + hl], fill=color)


# ---------------------------------------------------------------------------
# Sparkle system — the public API class
# ---------------------------------------------------------------------------

class SparkleSystem:
    """
    Pre-seeded star sparkles composited over an image on every frame.

    All sparkles loop seamlessly: their brightness envelopes are based on
    ``frame / n_frames`` so frame 0 == frame n_frames.

    Parameters
    ----------
    width, height : int
        Dimensions of the target canvas.
    n_frames : int
        Total number of animation frames.
    style : str
        One of ``SPARKLE_STYLES``.
    density : float
        Multiplier for sparkle count.  1.0 = default density.
        Ignored if *count* is set.
    count : int or None
        Exact number of sparkles.  Overrides *density* when set.
    max_size : float or None
        Maximum sparkle radius in pixels.  None = style-dependent default.
    size_vary : str
        Size distribution mode: ``"random"`` (default) assigns each sparkle
        a random size between 40–100% of *max_size*.  ``"fixed"`` makes all
        sparkles the same size.  ``"crescendo"`` ramps sizes up over the
        animation cycle (smallest at frame 0, largest at the end).
    seed : int or None
        Random seed for reproducibility.  None = random.
    """

    def __init__(
        self,
        width: int, height: int,
        n_frames: int, style: str = "twinkle",
        density: float = 1.0,
        count: int | None = None,
        max_size: float | None = None,
        size_vary: str = "random",
        seed: int | None = None,
    ) -> None:
        self.width     = width
        self.height    = height
        self.n_frames  = n_frames
        self.style     = style
        self.size_vary = size_vary

        rng = random.Random(seed)

        if count is not None:
            n = max(1, count)
        else:
            base_n = max(8, min(60, (width * height) // 400))
            n = max(4, int(base_n * density))

        # Default max radius depends on style; user override takes precedence
        if max_size is not None:
            default_max_r = max_size
        elif style == "glitter":
            default_max_r = 4.0
        else:
            default_max_r = 8.0

        # Minimum radius for random variation (40% of max)
        default_min_r = default_max_r * 0.4 if style != "glitter" else default_max_r * 0.38

        self.sparkles: list[dict] = []
        for idx in range(n):
            x = rng.uniform(0.02 * width,  0.98 * width)
            y = rng.uniform(0.02 * height, 0.98 * height)

            # Determine this sparkle's max_r based on size_vary mode
            if size_vary == "fixed":
                sp_max_r = default_max_r
            elif size_vary == "crescendo":
                # Assign a base fraction: sparkles seeded earlier in the list
                # correspond to earlier in the animation.  The actual crescendo
                # scaling happens at render time (see get_layer); here we store
                # a uniform max_r so the crescendo multiplier can scale it.
                sp_max_r = default_max_r
            else:
                # "random" — each sparkle gets a random size
                sp_max_r = rng.uniform(default_min_r, default_max_r)

            sp: dict = {
                "x0":     x,
                "y0":     y,
                "color":  rng.choice(_SPARKLE_COLORS),
                "max_r":  sp_max_r,
                "phase":  rng.uniform(0.0, 2.0 * math.pi),
                "period": rng.uniform(0.4, 2.2),
            }
            if style == "drift":
                # Generate random velocities, then quantize so that
                # total displacement over n_frames is an exact multiple
                # of the canvas dimension — guarantees seamless looping.
                raw_vx = rng.uniform(-0.25, 0.25)
                raw_vy = rng.uniform(-1.0, -0.3)  # upward
                if n_frames > 0 and width > 0:
                    kx = round(raw_vx * n_frames / width)
                    sp["vx"] = kx * width / n_frames if kx != 0 else raw_vx
                else:
                    sp["vx"] = raw_vx
                if n_frames > 0 and height > 0:
                    ky = round(raw_vy * n_frames / height)
                    # Ensure at least one full wrap so sparkles visibly move
                    if ky == 0:
                        ky = -1
                    sp["vy"] = ky * height / n_frames
                else:
                    sp["vy"] = raw_vy
            if style == "glitter":
                sp["period"] = rng.uniform(1.5, 5.0)
            self.sparkles.append(sp)

        # shimmer: extra parameters for the sweep band
        if style == "shimmer":
            self._shimmer_angle = math.radians(25)
            shimmer_max_r = default_max_r * 0.6
            # spawn a second population of sparkles along the band
            for _ in range(n // 2):
                x = rng.uniform(0.02 * width, 0.98 * width)
                y = rng.uniform(0.02 * height, 0.98 * height)
                self.sparkles.append({
                    "x0":     x,
                    "y0":     y,
                    "color":  (255, 255, 255),
                    "max_r":  rng.uniform(shimmer_max_r * 0.5, shimmer_max_r),
                    "phase":  rng.uniform(0.0, 2.0 * math.pi),
                    "period": rng.uniform(0.6, 1.8),
                    "shimmer_pop": True,
                })

    def get_layer(
        self, frame: int, size: tuple[int, int], gif_mode: bool = False
    ) -> Image.Image:
        """
        Render sparkle layer for the given frame.

        Parameters
        ----------
        frame : int
            Current frame index.
        size : tuple[int, int]
            (width, height) of the output layer.
        gif_mode : bool
            If True, use binary alpha (255 or 0) for GIF compatibility.

        Returns
        -------
        Image.Image
            RGBA layer with sparkles drawn on a transparent background.
        """
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        draw  = ImageDraw.Draw(layer, "RGBA")
        t     = frame / max(self.n_frames, 1)  # ∈ [0, 1)

        # Crescendo: scale all sizes by a ramp from ~0.2 at t=0 to 1.0 at t≈1
        # Uses a smooth sine quarter-wave so the loop resets gracefully
        if self.size_vary == "crescendo":
            cresc = 0.2 + 0.8 * math.sin(t * math.pi / 2)
        else:
            cresc = 1.0

        for sp in self.sparkles:
            # Apply crescendo scaling to the per-sparkle max_r
            effective_sp = sp
            if cresc != 1.0:
                effective_sp = {**sp, "max_r": sp["max_r"] * cresc}

            if self.style == "twinkle":
                self._draw_twinkle(draw, effective_sp, t, gif_mode)
            elif self.style == "burst":
                self._draw_burst(draw, effective_sp, t, gif_mode)
            elif self.style == "drift":
                self._draw_drift(draw, effective_sp, t, gif_mode)
            elif self.style == "shimmer":
                self._draw_shimmer(draw, effective_sp, t, gif_mode)
            elif self.style == "glitter":
                self._draw_glitter(draw, effective_sp, t, gif_mode)

        return layer

    # -- Individual style renderers -----------------------------------------

    def _draw_twinkle(
        self, draw: ImageDraw.ImageDraw, sp: dict, t: float, gif_mode: bool
    ) -> None:
        color  = sp["color"]
        phase  = sp["phase"]
        period = sp["period"]
        max_r  = sp["max_r"]

        raw = math.sin(2 * math.pi * period * t + phase)
        b   = max(0.0, raw) ** 1.5
        if b < 0.04:
            return
        r     = max_r * b
        alpha = 255 if gif_mode else int(b * 255)
        col   = (*color, alpha)
        _draw_4star(draw, sp["x0"], sp["y0"], r, r * 0.28, -math.pi / 4, col)
        # bright centre dot
        cr = r * 0.28
        draw.ellipse(
            [sp["x0"] - cr, sp["y0"] - cr, sp["x0"] + cr, sp["y0"] + cr],
            fill=col,
        )

    def _draw_burst(
        self, draw: ImageDraw.ImageDraw, sp: dict, t: float, gif_mode: bool
    ) -> None:
        color  = sp["color"]
        phase  = sp["phase"]
        period = sp["period"]
        max_r  = sp["max_r"]

        raw = math.sin(2 * math.pi * period * t + phase)
        b   = max(0.0, raw) ** 0.6
        if b < 0.04:
            return
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

    def _draw_drift(
        self, draw: ImageDraw.ImageDraw, sp: dict, t: float, gif_mode: bool
    ) -> None:
        color  = sp["color"]
        phase  = sp["phase"]
        period = sp["period"]
        max_r  = sp["max_r"]

        steps = t * self.n_frames
        x = (sp["x0"] + sp["vx"] * steps) % self.width
        y = (sp["y0"] + sp["vy"] * steps) % self.height if self.height > 0 else sp["y0"]

        raw = (math.sin(2 * math.pi * period * t + phase) + 1) / 2
        b   = raw ** 1.3
        r   = max_r * max(0.25, b)
        alpha = 255 if gif_mode else int(b * 220)
        col   = (*color, alpha)
        _draw_4star(draw, x, y, r, r * 0.28, -math.pi / 4, col)

    def _draw_shimmer(
        self, draw: ImageDraw.ImageDraw, sp: dict, t: float, gif_mode: bool
    ) -> None:
        color  = sp["color"]
        phase  = sp["phase"]
        period = sp["period"]
        max_r  = sp["max_r"]

        # The shimmer sweep: a diagonal band moves across the image
        # Band position sweeps from -0.3 to 1.3 across the image
        band_pos = (t * 1.6 - 0.3) % 1.6  # wraps seamlessly

        # Project sparkle position onto the sweep direction
        nx = sp["x0"] / max(self.width, 1)
        ny = sp["y0"] / max(self.height, 1)
        proj = nx * math.cos(self._shimmer_angle) + ny * math.sin(self._shimmer_angle)

        # Distance from sparkle to the band centre
        dist = abs(proj - band_pos)
        band_width = 0.18

        if sp.get("shimmer_pop"):
            # Sparkles along the wavefront: bright when the band passes over them
            if dist > band_width:
                return
            proximity = 1.0 - (dist / band_width)
            b = proximity ** 0.8
            r = max_r * b
            alpha = 255 if gif_mode else int(b * 230)
            col = (*color, alpha)
            _draw_4star(draw, sp["x0"], sp["y0"], r, r * 0.3, -math.pi / 4, col)
        else:
            # Background sparkles: also triggered by the band, with some independence
            raw = math.sin(2 * math.pi * period * t + phase)
            b_base = max(0.0, raw) ** 1.5

            # Boost brightness when the band passes nearby
            if dist < band_width * 1.5:
                proximity = 1.0 - (dist / (band_width * 1.5))
                b = max(b_base, proximity ** 0.6 * 0.9)
            else:
                b = b_base * 0.3  # dim background twinkle
            if b < 0.06:
                return
            r = max_r * b
            alpha = 255 if gif_mode else int(b * 240)
            col = (*color, alpha)
            _draw_4star(draw, sp["x0"], sp["y0"], r, r * 0.28, -math.pi / 4, col)
            cr = r * 0.25
            draw.ellipse(
                [sp["x0"] - cr, sp["y0"] - cr, sp["x0"] + cr, sp["y0"] + cr],
                fill=col,
            )

    def _draw_glitter(
        self, draw: ImageDraw.ImageDraw, sp: dict, t: float, gif_mode: bool
    ) -> None:
        color  = sp["color"]
        phase  = sp["phase"]
        period = sp["period"]
        max_r  = sp["max_r"]

        # Glitter: short, sharp flashes — use a triangle wave with a narrow peak
        raw = math.sin(2 * math.pi * period * t + phase)
        # Only flash in a narrow band around the peak
        b = max(0.0, (raw - 0.7) / 0.3) if raw > 0.7 else 0.0
        if b < 0.05:
            return
        r = max_r * b
        alpha = 255 if gif_mode else int(b * 255)
        col = (*color, alpha)
        # Simple cross shape for glitter
        _draw_cross(draw, sp["x0"], sp["y0"], r * 2, max(1.0, r * 0.3), col)
        # Tiny centre dot
        cr = max(0.5, r * 0.3)
        draw.ellipse(
            [sp["x0"] - cr, sp["y0"] - cr, sp["x0"] + cr, sp["y0"] + cr],
            fill=(255, 255, 255, alpha),
        )


# ---------------------------------------------------------------------------
# High-level helper — the easiest library API
# ---------------------------------------------------------------------------

def add_sparkles(
    frames: list[Image.Image],
    style: str = "twinkle",
    density: float = 1.0,
    count: int | None = None,
    max_size: float | None = None,
    size_vary: str = "random",
    fps: int = 20,
    gif_mode: bool = False,
    seed: int | None = None,
) -> list[Image.Image]:
    """
    Composite sparkle effects onto a list of RGBA frames.

    This is the simplest way to use the sparkle system from another script.
    It creates a :class:`SparkleSystem`, renders each frame's sparkle layer,
    and alpha-composites it onto the corresponding input frame.

    Parameters
    ----------
    frames : list[Image.Image]
        Input frames (all same size, RGBA mode).
    style : str
        Sparkle style name (see ``SPARKLE_STYLES``).
    density : float
        Sparkle count multiplier.  1.0 = default.  Ignored if *count* is set.
    count : int or None
        Exact number of sparkles.  Overrides *density*.
    max_size : float or None
        Maximum sparkle radius in pixels.  None = style-dependent default.
    size_vary : str
        Size distribution: ``"random"``, ``"fixed"``, or ``"crescendo"``.
    fps : int
        Frames per second (used for timing calculations).
    gif_mode : bool
        If True, sparkles use binary alpha for GIF compatibility.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    list[Image.Image]
        New list of RGBA frames with sparkles composited.
    """
    if not frames:
        return []
    w, h = frames[0].size
    n = len(frames)
    sys = SparkleSystem(
        w, h, n, style=style, density=density, count=count,
        max_size=max_size, size_vary=size_vary, seed=seed,
    )
    out = []
    for i, f in enumerate(frames):
        base = f.convert("RGBA")
        layer = sys.get_layer(i, (w, h), gif_mode=gif_mode)
        out.append(Image.alpha_composite(base, layer))
    return out


# ---------------------------------------------------------------------------
# GIF palette conversion
# ---------------------------------------------------------------------------

_GIF_TRANSP = 255


def _rgba_to_gif_palette(rgba: Image.Image,
                         ref_palette: Image.Image | None = None,
                         ) -> Image.Image:
    """Convert an RGBA frame to a 255-colour paletted image.

    If *ref_palette* is provided (a ``"P"`` mode image), the frame is
    quantized to that existing palette instead of generating a new one.
    This ensures identical pixels map to the same index across frames,
    which is critical for delta-based GIF compression.
    """
    alpha = np.asarray(rgba.split()[3])
    if ref_palette is not None:
        q = rgba.convert("RGB").quantize(
            colors=_GIF_TRANSP, palette=ref_palette, dither=0,
        )
    else:
        q = rgba.convert("RGB").quantize(colors=_GIF_TRANSP, dither=0)
    pal   = list(q.getpalette())[: _GIF_TRANSP * 3] + [0, 0, 0]
    arr   = np.asarray(q, dtype=np.uint8).copy()
    arr[alpha < 128] = _GIF_TRANSP
    out   = Image.fromarray(arr, "P")
    out.putpalette(pal)
    return out


# ---------------------------------------------------------------------------
# Input loading (static or animated)
# ---------------------------------------------------------------------------

def _load_frames(path: str) -> list[Image.Image]:
    """
    Load all frames from an image file.

    For animated GIF/APNG, returns every frame as RGBA.
    For static images, returns a single-element list.
    """
    img = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(img.convert("RGBA").copy())
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    if not frames:
        frames.append(img.convert("RGBA").copy())
    return frames


def _get_input_fps(path: str) -> int | None:
    """Try to read the frame duration from an animated image."""
    try:
        img = Image.open(path)
        duration = img.info.get("duration", None)
        if duration and duration > 0:
            return max(1, round(1000 / duration))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------

def build_frames(
    input_frames: list[Image.Image],
    style: str = "twinkle",
    density: float = 1.0,
    count: int | None = None,
    max_size: float | None = None,
    size_vary: str = "random",
    n_frames: int = 24,
    fps: int = 20,
    seed: int | None = None,
) -> list[Image.Image]:
    """
    Build sparkle animation frames from input frame(s).

    If input_frames has one element (static image), the sparkle animation
    creates n_frames output frames.  If input_frames has multiple elements
    (animated input), sparkles are composited onto each existing frame and
    n_frames is ignored.
    """
    w, h = input_frames[0].size
    is_animated = len(input_frames) > 1

    if is_animated:
        n = len(input_frames)
        base_frames = input_frames
    else:
        n = n_frames
        base_frames = [input_frames[0].copy() for _ in range(n)]

    sys = SparkleSystem(
        w, h, n, style=style, density=density, count=count,
        max_size=max_size, size_vary=size_vary, seed=seed,
    )

    frames: list[Image.Image] = []
    for i in range(n):
        print(f"\r  Building frame {i + 1}/{n} …", end="", flush=True)
        base = base_frames[i].convert("RGBA")
        layer = sys.get_layer(i, (w, h), gif_mode=False)
        frames.append(Image.alpha_composite(base, layer))

    print()
    return frames


# ---------------------------------------------------------------------------
# Output savers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Custom APNG writer — bypasses Pillow's auto-optimiser so that we
# control bbox cropping, blend mode, and within-bbox zeroing ourselves.
# ---------------------------------------------------------------------------

import io
import struct
import zlib


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Build a single PNG chunk: length + type + data + CRC."""
    out = struct.pack(">I", len(data)) + chunk_type + data
    crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    return out + crc


def _compress_rgba_subframe(img: Image.Image) -> bytes:
    """Return raw PNG IDAT payload (deflate-compressed, with filter bytes).

    Uses Pillow to encode a single RGBA frame and then extracts the
    IDAT data, which is the same byte sequence we need for fdAT.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    buf.seek(0)

    # Walk PNG chunks and collect IDAT payloads.
    buf.read(8)  # skip PNG signature
    idat_data = b""
    while True:
        header = buf.read(8)
        if len(header) < 8:
            break
        length = struct.unpack(">I", header[:4])[0]
        chunk_type = header[4:8]
        payload = buf.read(length)
        buf.read(4)  # skip CRC
        if chunk_type == b"IDAT":
            idat_data += payload
        elif chunk_type == b"IEND":
            break
    return idat_data


def _save_apng(frames: list[Image.Image], path: str, fps: int) -> None:
    """Write an APNG with manual delta encoding.

    Unlike GIF, APNG offers two blend modes per frame:

    * **OP_OVER** (1): the sub-frame is alpha-composited onto the
      canvas.  Fully-transparent pixels leave the canvas unchanged,
      allowing us to zero-out unmodified pixels *within* the
      bounding-box for dramatically better deflate compression.

    * **OP_SOURCE** (0): the sub-frame *replaces* the canvas region,
      including transparent pixels erasing opaque ones.  Used when a
      pixel transitions from visible → transparent — something
      ``OP_OVER`` cannot represent.

    We write the APNG ourselves (rather than via Pillow's
    ``save_all``) because Pillow's ``_write_multiple_frames``
    re-computes its own bounding boxes and ignores our pre-zeroed
    deltas.
    """
    ms = int(1000 / fps)
    delay_num = ms
    delay_den = 1000
    n = len(frames)
    w, h = frames[0].size
    print("  Delta-encoding …", end="", flush=True)

    # ---- Build sub-frame data for each frame ----
    # Each entry: (sub_img, x, y, dispose_op, blend_op)
    DISPOSE_NONE = 0
    BLEND_SOURCE = 0
    BLEND_OVER = 1

    sub_frames: list[tuple[Image.Image, int, int, int, int]] = []

    # Frame 0: always full, source blend.
    sub_frames.append((frames[0], 0, 0, DISPOSE_NONE, BLEND_SOURCE))

    # Track accumulated canvas state to detect stale pixels.
    canvas = np.asarray(frames[0]).copy()
    keyframe_interval = 60
    stale = np.zeros((h, w), dtype=bool)

    for i in range(1, n):
        curr = np.asarray(frames[i])
        diff = (canvas != curr).any(axis=2)

        if not diff.any():
            # Identical — emit a 1×1 transparent patch (APNG requires
            # a frame; Pillow merges durations but we write manually).
            patch = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            sub_frames.append((patch, 0, 0, DISPOSE_NONE, BLEND_OVER))
            continue

        # Track opaque→transparent transitions (can't fix with OP_OVER).
        newly_stale = diff & (canvas[:, :, 3] > 0) & (curr[:, :, 3] == 0)
        fixed = stale & (curr[:, :, 3] > 0)
        stale = (stale | newly_stale) & ~fixed
        n_stale = int(np.count_nonzero(stale))

        force_key = (
            (i % keyframe_interval == 0)
            or (n_stale > w * h * 0.05)
        )

        if force_key:
            # OP_SOURCE keyframe: full content, resets canvas perfectly.
            # Pillow auto-crops it to the content bbox via the diff
            # against what was on canvas.
            sub_frames.append((frames[i], 0, 0, DISPOSE_NONE, BLEND_SOURCE))
            canvas = curr.copy()
            stale[:] = False
            continue

        # Bounding box of changed pixels.
        rows = np.any(diff, axis=1)
        cols = np.any(diff, axis=0)
        r0, r1 = int(np.argmax(rows)), int(h - 1 - np.argmax(rows[::-1]))
        c0, c1 = int(np.argmax(cols)), int(w - 1 - np.argmax(cols[::-1]))

        # OP_OVER delta: zero unchanged pixels inside bbox so PNG's
        # deflate compresses the mostly-zero rows efficiently.
        # Alpha=0 under OP_OVER = "keep canvas" → perfect delta.
        sub = curr[r0:r1+1, c0:c1+1].copy()
        local_diff = diff[r0:r1+1, c0:c1+1]
        sub[~local_diff] = (0, 0, 0, 0)

        sub_img = Image.fromarray(sub, "RGBA")
        sub_frames.append((sub_img, c0, r0, DISPOSE_NONE, BLEND_OVER))

        # Update virtual canvas: OP_OVER only paints non-transparent.
        painted = (sub != 0).any(axis=2) if sub.ndim == 3 else (sub != 0)
        canvas[r0:r1+1, c0:c1+1][painted] = curr[r0:r1+1, c0:c1+1][painted]

    # ---- Assemble the APNG file ----
    print("\r  Writing APNG …       ", end="", flush=True)

    # PNG signature
    out = b"\x89PNG\r\n\x1a\n"

    # IHDR
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)  # 8-bit RGBA
    out += _png_chunk(b"IHDR", ihdr)

    # acTL — animation control
    actl = struct.pack(">II", n, 0)  # num_frames, num_plays (0 = infinite)
    out += _png_chunk(b"acTL", actl)

    seq = 0  # APNG sequence number (shared across fcTL and fdAT)

    for idx, (sub_img, x, y, dispose_op, blend_op) in enumerate(sub_frames):
        sw, sh = sub_img.size

        # fcTL — frame control
        fctl = struct.pack(
            ">IIIIIHHBB",
            seq,            # sequence_number
            sw, sh,         # width, height
            x, y,           # x_offset, y_offset
            delay_num,      # delay_numerator
            delay_den,      # delay_denominator
            dispose_op,     # dispose_op
            blend_op,       # blend_op
        )
        out += _png_chunk(b"fcTL", fctl)
        seq += 1

        idat_data = _compress_rgba_subframe(sub_img)

        if idx == 0:
            # Frame 0 uses IDAT (required for backwards compat).
            out += _png_chunk(b"IDAT", idat_data)
        else:
            # Subsequent frames use fdAT (= sequence_number + IDAT data).
            fdat = struct.pack(">I", seq) + idat_data
            out += _png_chunk(b"fdAT", fdat)
            seq += 1

    # IEND
    out += _png_chunk(b"IEND", b"")

    with open(path, "wb") as f:
        f.write(out)

    print(f"\r✓  APNG saved:  {path}  ({n} frames @ {fps} fps, {ms} ms/frame)")


def _delta_encode_gif(
    pal_frames: list[Image.Image],
    keyframe_interval: int = 60,
) -> tuple[list[Image.Image], list[int]]:
    """Delta-encode palette frames for compact GIF output.

    Returns ``(encoded_frames, disposals)`` ready for
    :pymethod:`Image.save`.

    Between keyframes, each frame stores only the pixels that changed
    relative to the *accumulated canvas* (i.e. what a GIF decoder would
    actually show).  Unchanged pixels are set to the transparency index
    so LZW compresses them into almost nothing, and Pillow's
    ``optimize=True`` further crops each frame to the changed bounding
    box.

    **Keyframes** (full content, ``disposal=2``) are emitted:

    * at frame 0 (always),
    * every *keyframe_interval* frames, and
    * whenever the number of currently-stale pixels (opaque on canvas
      but should be transparent) exceeds 5 % of the canvas.

    The last frame always uses ``disposal=2`` so the canvas is clean
    when the GIF loops.
    """
    n = len(pal_frames)
    if n == 0:
        return [], []
    if n == 1:
        return list(pal_frames), [2]

    palette = pal_frames[0].getpalette()
    h, w = np.asarray(pal_frames[0]).shape
    total_px = h * w

    encoded: list[Image.Image] = [pal_frames[0]]
    disposals: list[int] = [1]  # frame 0: keep on canvas for frame 1's delta

    canvas = np.asarray(pal_frames[0]).copy()
    # Track which pixels are currently stale (wrong) on the canvas.
    # A pixel is stale when it's opaque on canvas but should be
    # transparent in the true frame.
    stale = np.zeros((h, w), dtype=bool)

    for i in range(1, n):
        curr = np.asarray(pal_frames[i])
        is_last = (i == n - 1)

        # Pixels where canvas is opaque but current frame wants transparent.
        newly_stale = (canvas != _GIF_TRANSP) & (curr == _GIF_TRANSP) & (canvas != curr)
        # Pixels that were stale but are now opaque again (sparkle returned
        # or base image pixel restored) — no longer stale.
        fixed = stale & (curr != _GIF_TRANSP)
        stale = (stale | newly_stale) & ~fixed
        n_stale = int(np.count_nonzero(stale))

        force_key = (
            is_last
            or (i % keyframe_interval == 0)
            or (n_stale > total_px * 0.05)
        )

        if force_key:
            # Keyframe: full content.  The keyframe itself keeps
            # disposal=1 so it stays on canvas for subsequent deltas.
            # The *previous* frame gets disposal=2 to clear the canvas
            # before this keyframe is drawn (ensures a clean slate).
            encoded.append(pal_frames[i])
            disposals.append(1)
            canvas = curr.copy()
            stale[:] = False
            disposals[-2] = 2
            if is_last:
                # Last frame clears canvas so the loop restarts cleanly.
                disposals[-1] = 2
        else:
            # Delta frame: transparent where pixel matches canvas.
            delta = curr.copy()
            same = (canvas == curr)
            # Also treat stale pixels as "same" — we can't fix them
            # with disposal=1, so don't try (leave as transparent →
            # "keep previous").
            same |= (stale & newly_stale)
            delta[same] = _GIF_TRANSP

            frame = Image.fromarray(delta, "P")
            frame.putpalette(palette)
            encoded.append(frame)
            disposals.append(1)  # keep this frame for next delta

            # Update the virtual canvas to reflect what the decoder sees.
            # Transparent pixels in the delta mean "keep previous", so
            # only non-transparent delta pixels update the canvas.
            painted = (delta != _GIF_TRANSP)
            canvas[painted] = curr[painted]

    return encoded, disposals


def _save_gif(frames: list[Image.Image], path: str, fps: int) -> None:
    ms = int(1000 / fps)
    print("  Converting to palette …", end="", flush=True)
    # Quantize frame 0 to build a reference palette, then map every
    # subsequent frame to the same palette so identical pixels produce
    # identical indices — essential for delta encoding.
    pal0 = _rgba_to_gif_palette(frames[0])
    pal = [pal0] + [_rgba_to_gif_palette(f, ref_palette=pal0) for f in frames[1:]]

    # Choose encoding strategy based on image content.  Delta encoding
    # with disposal=1 gives large savings when the image has transparent
    # regions (common for emoji), but for fully-opaque images Pillow's
    # built-in disposal=2 optimization is more effective.
    has_transparency = np.any(np.asarray(pal[0]) == _GIF_TRANSP)
    if has_transparency:
        encoded, disposals = _delta_encode_gif(pal)
    else:
        encoded, disposals = pal, 2

    print("\r  Writing GIF …          ", end="", flush=True)
    encoded[0].save(
        path, format="GIF", save_all=True, append_images=encoded[1:],
        loop=0, duration=ms,
        transparency=_GIF_TRANSP, disposal=disposals, optimize=True,
    )
    print(f"\r✓  GIF saved:   {path}  ({len(frames)} frames @ {fps} fps, {ms} ms/frame)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input",  metavar="INPUT",  help="Source image (JPEG/PNG/GIF/APNG/…)")
    p.add_argument("output", metavar="OUTPUT", help="Output animation (.gif / .png / .apng)")

    g = p.add_argument_group("sparkle")
    g.add_argument(
        "--style", choices=SPARKLE_STYLES, default="twinkle", metavar="STYLE",
        help=("Sparkle style: "
              + "  ".join(f"{k} ({v})" for k, v in _STYLE_DOCS.items())
              + "  (default: twinkle)"),
    )
    g.add_argument("--density", type=float, default=1.0, metavar="N",
                   help="Sparkle density multiplier (default: 1.0, ignored if --count is set)")
    g.add_argument("--count", type=int, default=None, metavar="N",
                   help="Exact number of sparkles (overrides --density)")
    g.add_argument(
        "--sparkle-size", default=None, metavar="SPEC",
        help="Maximum sparkle size: '20px' for pixels, '15%%' for percentage "
             "of min(width,height), or bare number for pixels (default: style-dependent)",
    )
    g.add_argument(
        "--size-vary", choices=SIZE_VARY_MODES, default="random", metavar="MODE",
        help=("Size variation: "
              + "  ".join(f"{k} ({v})" for k, v in _SIZE_VARY_DOCS.items())
              + "  (default: random)"),
    )
    g.add_argument("--seed", type=int, default=None, metavar="N",
                   help="Random seed for reproducibility")

    g = p.add_argument_group("animation")
    g.add_argument("--frames", type=int, default=24, metavar="N",
                   help="Number of frames for static input (default: 24, ignored for animated input)")
    g.add_argument("--fps", type=int, default=20, metavar="N",
                   help="Frames per second (default: 20, or inherited from animated input)")
    g.add_argument("--size", metavar="WxH",
                   help="Resize the input image before processing, e.g. 128x128")

    return p


def main() -> None:
    args = build_parser().parse_args()

    if Path(args.input).resolve() == Path(args.output).resolve():
        sys.exit("Error: input and output paths are the same file")

    try:
        input_frames = _load_frames(args.input)
    except FileNotFoundError:
        sys.exit(f"Error: cannot open {args.input!r}")

    is_animated = len(input_frames) > 1

    if args.size:
        try:
            rw, rh = (int(d) for d in args.size.lower().split("x", 1))
        except ValueError:
            sys.exit("Error: --size must be WxH, e.g. 128x128")
        input_frames = [f.resize((rw, rh), Image.LANCZOS) for f in input_frames]

    # Inherit fps from animated input if not explicitly set
    fps = args.fps
    if is_animated:
        input_fps = _get_input_fps(args.input)
        if input_fps is not None and args.fps == 20:  # 20 is the default
            fps = input_fps

    w, h = input_frames[0].size

    # Parse --sparkle-size into pixels
    max_size = None
    if args.sparkle_size is not None:
        try:
            max_size = parse_size_spec(args.sparkle_size, min(w, h))
        except ValueError:
            sys.exit(f"Error: invalid --sparkle-size {args.sparkle_size!r}.  "
                     "Use e.g. '20px', '15%', or a bare number")
        if max_size <= 0:
            sys.exit("Error: --sparkle-size must be positive")

    count_str = f"count={args.count}" if args.count is not None else f"density={args.density}"
    size_str = f"sparkle-size={args.sparkle_size}" if args.sparkle_size else ""
    print(f"Input : {args.input}  ({w}×{h}, {'animated' if is_animated else 'static'}"
          + (f", {len(input_frames)} frames" if is_animated else "") + ")")
    print(f"Output: {args.output}")
    print(f"  style={args.style}  {count_str}  size-vary={args.size_vary}"
          + (f"  {size_str}" if size_str else "")
          + (f"  seed={args.seed}" if args.seed is not None else "")
          + (f"  frames={args.frames}" if not is_animated else "")
          + f"  fps={fps}")

    frames = build_frames(
        input_frames,
        style=args.style,
        density=args.density,
        count=args.count,
        max_size=max_size,
        size_vary=args.size_vary,
        n_frames=args.frames,
        fps=fps,
        seed=args.seed,
    )

    ext = Path(args.output).suffix.lower()
    if ext == ".gif":
        _save_gif(frames, args.output, fps)
    elif ext in (".png", ".apng"):
        _save_apng(frames, args.output, fps)
    else:
        sys.exit(
            f"Error: unknown output extension {ext!r}.  Use .gif, .png, or .apng"
        )


if __name__ == "__main__":
    main()
