from PIL import Image, ImageDraw

class DummyImageAdapter:
    def generate(self, prompt, size=256):
        img = Image.new("RGB", (size, size), (24, 24, 24))
        d = ImageDraw.Draw(img)
        d.rectangle([24, 24, size-24, size-24], outline=(160, 220, 255), width=3)
        d.text((16, 16), prompt[:24], fill=(230, 230, 230))
        return img
