"""Plan 38 J0.3/J0.4 (static): emotion colour never filters the face; ASAL image never covers the avatar.
The live computed-style check is tools/ui_dom_check.py (headless Chrome)."""
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web" / "gemma_chat"
FORBIDDEN = re.compile(r"hue-rotate|saturate|sepia|invert|grayscale")
FACE_CHAIN = ("avatar-face", "avatarFace", "noir-viewport", "avatar-wrap", "avatarWrap", "avatar-stage", "avatarStage",
              "visual-container", "hero-panel", "app-shell", "body")


def css_rules(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", text):
        yield m.group(1).strip(), m.group(2)


class _Tree(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.parents = [], {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        key = a.get("id") or a.get("class") or tag
        self.parents[key] = [s for s in self.stack]
        if tag not in ("img", "br", "input", "meta", "link", "source"):
            self.stack.append(a.get("class") or a.get("id") or tag)

    def handle_endtag(self, tag):
        if self.stack and tag not in ("img", "br", "input", "meta", "link", "source"):
            self.stack.pop()


class WebAvatarLayersTest(unittest.TestCase):
    def test_no_colour_filter_on_face_chain(self):
        css = (WEB / "styles.css").read_text(encoding="utf-8")
        for selector, body in css_rules(css):
            if "filter" not in body or not FORBIDDEN.search(body):
                continue
            last = selector.split(",")
            for sel in last:
                target = sel.strip().split()[-1] if sel.strip() else ""
                self.assertFalse(any(k in target for k in FACE_CHAIN) and "halo" not in target,
                                 f"colour filter on face layer: {sel} {{{body.strip()}}}")

    def test_emotion_rules_target_halo(self):
        css = (WEB / "styles.css").read_text(encoding="utf-8")
        emo = [(s, b) for s, b in css_rules(css) if "data-emotion" in s]
        self.assertTrue(emo)
        for sel, body in emo:
            self.assertIn("avatar-halo", sel)
            self.assertNotIn("filter", body)

    def test_life_visual_not_in_avatar_viewport(self):
        tree = _Tree()
        tree.feed((WEB / "index.html").read_text(encoding="utf-8"))
        self.assertIn("avatarFace", tree.parents)
        self.assertIn("noir-viewport", tree.parents["avatarFace"])
        self.assertIn("lifeVisual", tree.parents)
        self.assertNotIn("noir-viewport", tree.parents["lifeVisual"])

    def test_avatar_face_uses_fixed_asset(self):
        html = (WEB / "index.html").read_text(encoding="utf-8")
        self.assertRegex(html, r'id="avatarFace"[^>]*src="/avatar.jpg"')


if __name__ == "__main__":
    unittest.main()
