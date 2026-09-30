#!/usr/bin/env python3
"""
Generate the Nothing OS-style blurred ribbon wallpaper.

    gen_wallpaper.py --accent e2201f -o ~/.wallpapers/Nothing1.png
    gen_wallpaper.py --accent 0fff50 -o ~/.wallpapers/Nothing1.png --versioned
    gen_wallpaper.py --accent 1a3cff --intensity 1.8 --size 2560x1600 -o /tmp/wp.png

stdout is always exactly one line: the path of the image that was written.
Everything else goes to stderr, so `path=$(gen_wallpaper.py ...)` is safe.

--intensity defaults to "auto": dim accents (low relative luminance) are
brightened, up to 1.8x, so they stay visible on the near-black background;
accents as bright as the default red are left untouched. Pass a number to
override, 1.0 to disable.

--versioned writes <stem>-<contenthash><suffix> next to the output, then
atomically points the plain -o path at it with a symlink. Use that when a
consumer (hyprpaper) ignores a file whose *name* did not change: hand it the
printed versioned path, and leave rofi / hyprpaper.conf on the stable name.
"""
import argparse
import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

REF_W, REF_H = 1920, 1080  # the ribbon geometry below is authored for this size


# --- argument types ---------------------------------------------------------

def hex_color(s):
    h = s.strip().lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        raise argparse.ArgumentTypeError(f"not a 6-digit hex color: {s!r}")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def size_arg(s):
    m = re.fullmatch(r"(\d+)x(\d+)", s.strip().lower())
    if not m:
        raise argparse.ArgumentTypeError("size must look like 1920x1080")
    w, h = int(m[1]), int(m[2])
    if not (64 <= w <= 16384 and 64 <= h <= 16384):
        raise argparse.ArgumentTypeError("each side must be between 64 and 16384")
    return w, h


def positive(s):
    v = float(s)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v


def intensity_arg(s):
    return None if s.strip().lower() == "auto" else positive(s)


def positive_int(s):
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


# --- rendering --------------------------------------------------------------

AUTO_TARGET_LUMINANCE = 0.15  # the default red (e2201f) is ~0.17, so it is untouched
AUTO_MAX_INTENSITY = 1.8


def luminance(rgb):
    """WCAG relative luminance, 0 (black) .. 1 (white)."""
    def lin(c):
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def auto_intensity(accent):
    ratio = AUTO_TARGET_LUMINANCE / max(luminance(accent), 1e-6)
    return min(AUTO_MAX_INTENSITY, max(1.0, ratio))


def generate(size, bg_c, accent, base, blur=90.0, intensity=1.0, seed=7):
    """Return an RGB PIL image. Deterministic for a given set of arguments."""
    w, h = size
    sx, sy = w / REF_W, h / REF_H
    s = min(sx, sy)

    bg = Image.new("RGB", (w, h), bg_c)
    blob = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(blob)

    # Ribbon path: an organic wavy curve across the upper-left quadrant.
    t = np.linspace(0, 1, 400)
    x = (-100 + t * 1500) * sx
    y = 220 + 190 * np.sin(t * 3.4 * np.pi + 0.4) + 40 * np.sin(t * 7.0 + 1.2)
    y = (y + 15 * np.sin(t * 13 + 2.0)) * sy
    # Thicker at the start, thinner as it exits right.
    width = np.clip(260 - 140 * t + 20 * np.sin(t * 5), 60, 280) * s

    for xi, yi, wi, ti in zip(x, y, width, t):
        # Brightest (tinted toward base) at the start, pure accent after that.
        mix = np.clip(1.0 - ti * 2.2, 0, 1) * 0.55
        r, g, b = (int(accent[i] + (base[i] - accent[i]) * mix) for i in range(3))
        alpha = int(255 * np.clip(1.0 - ti * 1.15, 0, 1) ** 1.3)
        if alpha <= 0:
            continue
        draw.ellipse(
            [xi - wi / 2, yi - wi / 2, xi + wi / 2, yi + wi / 2],
            fill=(r, g, b, alpha),
        )

    blob = blob.filter(ImageFilter.GaussianBlur(blur * s))
    blob_rgb = blob.convert("RGB")
    alpha = blob.getchannel("A")

    # Dark accents barely show up when screen-blended onto a near-black bg;
    # --intensity pushes both the color and its coverage. 1.0 = untouched.
    if intensity != 1.0:
        blob_rgb = ImageEnhance.Brightness(blob_rgb).enhance(intensity)
        alpha = alpha.point(lambda v: min(255, round(v * intensity)))

    screen = ImageChops.screen(bg, blob_rgb)
    result = Image.composite(screen, bg, alpha)

    # Very subtle grain to avoid banding. NOTE: the seed only changes this
    # noise; the ribbon shape is fixed.
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 3.2, (h, w, 1)).astype(np.int16)
    arr = np.clip(np.array(result).astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# --- output -----------------------------------------------------------------

def save_atomic(img, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".tmp-", suffix=".png")
    os.close(fd)
    try:
        img.save(tmp, "PNG")
        os.chmod(tmp, 0o644)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def point_link(link, target):
    """Atomically make `link` a relative symlink to `target` (same directory)."""
    tmp = link.with_name(f".{link.name}.tmp-{os.getpid()}")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(target.name)
    try:
        os.replace(tmp, link)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def prune(out, keep, current):
    pat = re.compile(rf"{re.escape(out.stem)}-[0-9a-f]{{8}}{re.escape(out.suffix)}")
    old = sorted(
        (p for p in out.parent.iterdir() if pat.fullmatch(p.name) and p != current),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in old[max(keep - 1, 0):]:
        p.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--accent", type=hex_color, default="e2201f", help="ribbon color")
    ap.add_argument("--bg", type=hex_color, default="1b1c1d", help="background color")
    ap.add_argument("--base", type=hex_color, default="fdfdfd", help="highlight tint at the ribbon start")
    ap.add_argument("--size", type=size_arg, default=(REF_W, REF_H), metavar="WxH")
    ap.add_argument("--blur", type=positive, default=90.0, help="blur radius at 1920x1080")
    ap.add_argument("--intensity", type=intensity_arg, default="auto", metavar="N|auto",
                    help="brighten dim accents; auto (default) scales by luminance, 1.0 = off")
    ap.add_argument("--seed", type=int, default=7, help="grain noise seed only")
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--versioned", action="store_true", help="content-hashed file + stable symlink")
    ap.add_argument("--keep", type=positive_int, default=3, metavar="N", help="versions to keep with --versioned")
    args = ap.parse_args()

    # abspath, NOT resolve(): if -o is already our symlink we must replace the
    # link, not follow it and overwrite the versioned file it points at.
    out = Path(os.path.abspath(args.output.expanduser()))

    intensity = auto_intensity(args.accent) if args.intensity is None else args.intensity

    try:
        img = generate(args.size, args.bg, args.accent, args.base,
                       args.blur, intensity, args.seed)
        if args.versioned:
            digest = hashlib.sha1(img.tobytes()).hexdigest()[:8]
            versioned = out.with_name(f"{out.stem}-{digest}{out.suffix}")
            save_atomic(img, versioned)
            point_link(out, versioned)
            prune(out, args.keep, versioned)
            print(versioned)
        else:
            save_atomic(img, out)
            print(out)
    except OSError as e:
        print(f"gen_wallpaper: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
