import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops
from datetime import datetime

W, H = 1920, 1080

BG = (0x1B, 0x1C, 0x1D)
# ACCENT = (0xE2, 0x20, 0x1F)
ACCENT = (0x0F, 0xFF, 0x50)
BASE = (0xFD, 0xFD, 0xFD)

# Background
bg = Image.new("RGB", (W, H), BG)

# Blob layer (RGBA), draw on transparent, will blur then screen-blend onto bg
blob = Image.new("RGBA", (W, H), (0, 0, 0, 0))
draw = ImageDraw.Draw(blob)

rng = np.random.default_rng(7)

# Ribbon path: an organic wavy curve across the upper-left quadrant
t = np.linspace(0, 1, 400)
x = -100 + t * 1500
y = 220 + 190 * np.sin(t * 3.4 * np.pi + 0.4) + 40 * np.sin(t * 7.0 + 1.2)
# add gentle organic jitter
y += 15 * np.sin(t * 13 + 2.0)

# width tapers thicker near start (top-left) and thinner as it exits right
width = 260 - 140 * t + 20 * np.sin(t * 5)
width = np.clip(width, 60, 280)

# color/alpha: brightest (mix toward BASE, high alpha) near start, fading toward
# ACCENT then fading alpha to 0 as it exits the frame
for xi, yi, wi, ti in zip(x, y, width, t):
    # mix color from BASE-tinted accent (bright) to pure accent to dark accent
    mix = np.clip(1.0 - ti * 2.2, 0, 1)  # brightness mix factor near the start
    r = int(ACCENT[0] + (BASE[0] - ACCENT[0]) * mix * 0.55)
    g = int(ACCENT[1] + (BASE[1] - ACCENT[1]) * mix * 0.55)
    b = int(ACCENT[2] + (BASE[2] - ACCENT[2]) * mix * 0.55)
    alpha = int(255 * np.clip(1.0 - ti * 1.15, 0, 1) ** 1.3)
    if alpha <= 0:
        continue
    draw.ellipse(
        [xi - wi / 2, yi - wi / 2, xi + wi / 2, yi + wi / 2],
        fill=(r, g, b, alpha),
    )

# Heavy blur for the soft diffused look
blob = blob.filter(ImageFilter.GaussianBlur(90))

# Screen-blend the blurred blob onto the dark background (keeps bg pure where blob alpha~0)
bg_rgba = bg.convert("RGBA")
blob_rgb = blob.convert("RGB")
alpha = blob.split()[3]

screen = ImageChops.screen(bg_rgba.convert("RGB"), blob_rgb)
result = Image.composite(screen, bg, alpha)

# Very subtle film grain / noise to avoid banding in the gradient
noise = (rng.normal(0, 3.2, (H, W, 1))).astype(np.int16)
arr = np.array(result).astype(np.int16)
arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
result = Image.fromarray(arr, "RGB")

filename = f"wallpaper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
result.save(os.path.join(os.getcwd(), "wallpaper", filename))

print("done", result.size)
