from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _safe_text(text: str) -> str:
    try:
        text.encode("latin-1")
        return text
    except UnicodeEncodeError:
        return text.encode("ascii", errors="replace").decode("ascii")

class DummyImageAdapter:
    def generate(self, prompt, size=256):
        img = Image.new("RGB", (size, size), (24, 24, 24))
        d = ImageDraw.Draw(img)
        font = _load_font(16)
        d.rectangle([24, 24, size-24, size-24], outline=(160, 220, 255), width=3)
        text = prompt[:24]
        try:
            d.text((16, 16), text, fill=(230, 230, 230), font=font)
        except UnicodeEncodeError:
            d.text((16, 16), _safe_text(text), fill=(230, 230, 230), font=font)
        return img
