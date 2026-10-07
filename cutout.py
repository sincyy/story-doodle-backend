"""
Fotoğrafı çekilen çizimi kâğıt zemininden ayırır (arka plan şeffaf olur).
Sadece Pillow gerekir:  pip install pillow
"""
import base64
import io

from PIL import Image, ImageChops, ImageFilter, ImageOps

_RES = getattr(Image, "Resampling", Image)

MAX_SIDE = 900
BG_SCALE = 4
BG_WINDOW = 15


def make_cutout_data_uri(image_bytes: bytes) -> str | None:
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    w, h = img.size
    if w < 32 or h < 32:
        return None

    small = img.resize((max(8, w // BG_SCALE), max(8, h // BG_SCALE)), _RES.BILINEAR)

    diffs = []
    for full_ch, small_ch in zip(img.split(), small.split()):
        bg = (
            small_ch.filter(ImageFilter.MaxFilter(BG_WINDOW))
            .filter(ImageFilter.GaussianBlur(5))
            .resize((w, h), _RES.BILINEAR)
        )
        diffs.append(ImageChops.subtract(bg, full_ch))
    diff = ImageChops.lighter(ImageChops.lighter(diffs[0], diffs[1]), diffs[2])

    hist = diff.histogram()
    total = w * h
    target = total * 0.003
    acc = 0
    ref = 255
    for v in range(255, -1, -1):
        acc += hist[v]
        if acc >= target:
            ref = v
            break
    ref = max(ref, 60)
    lo, hi = ref * 0.45, ref * 0.80

    lut = []
    for v in range(256):
        if v <= lo:
            lut.append(0)
        elif v >= hi:
            lut.append(255)
        else:
            lut.append(int((v - lo) / (hi - lo) * 255))
    alpha = diff.point(lut)

    alpha = alpha.filter(ImageFilter.RankFilter(3, 7)).filter(ImageFilter.MaxFilter(3))

    a_hist = alpha.histogram()
    ink_fraction = sum(a_hist[128:]) / float(total)
    if ink_fraction < 0.0005 or ink_fraction > 0.35:
        return None

    rgba = img.convert("RGBA")
    rgba.putalpha(alpha)

    buf = io.BytesIO()
    rgba.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")