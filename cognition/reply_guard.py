"""Plan 38: deterministic post-generation reply guards (fixes for the open Plan 37 R2 defects).

  foreign_script      - stray tokens in scripts the conversation never uses (R2: 「我 হলো ALife Prototype」)
  perspective         - the user's own facts echoed as the clone's (R2: 「我的研究暗語是…」 -> 「你的研究暗語是…」)
  identity_override   - the clone accepting a rename / new identity (R2 third_party_override: 「…身分改變為 RAGEBOT」)
  unasked_secret      - a stored passphrase surfacing in an unrelated reply (R2: fear reply quoted RED-COMET-2208)
  fabricated_action   - claims of completed actions without a tool call (spec 4.10 T5: 「已設定」, "Reminder set")
Every guard is a pure function so it is unit-testable; ``guard_reply`` applies all of them and reports flags."""
from __future__ import annotations

import re

from cognition.leak_guard import split_sentences

# ---------------------------------------------------------------- foreign scripts
_FOREIGN_SCRIPTS = {
    "bengali": "\u0980-\u09ff", "devanagari": "\u0900-\u097f", "gurmukhi_gujarati_oriya": "\u0a00-\u0b7f",
    "tamil_telugu_kannada_malayalam_sinhala": "\u0b80-\u0dff", "thai_lao": "\u0e00-\u0eff", "tibetan": "\u0f00-\u0fff",
    "myanmar": "\u1000-\u109f", "georgian": "\u10a0-\u10ff", "ethiopic": "\u1200-\u139f", "khmer": "\u1780-\u17ff",
    "arabic": "\u0600-\u06ff\u0750-\u077f", "hebrew": "\u0590-\u05ff", "armenian": "\u0530-\u058f",
    "cyrillic": "\u0400-\u04ff", "greek": "\u0370-\u03ff", "hangul": "\uac00-\ud7af\u1100-\u11ff\u3130-\u318f",
    "kana": "\u3040-\u30ff",
}
_FOREIGN_RE = {k: re.compile(f"[{v}]+[\u0300-\u036f\u0981-\u0983\u09bc-\u09d7]*") for k, v in _FOREIGN_SCRIPTS.items()}


def foreign_scripts(text: str, context: str = "") -> list[str]:
    """Scripts present in ``text`` that are absent from ``context`` (the user's message + persona)."""
    return [name for name, rx in _FOREIGN_RE.items() if rx.search(text or "") and not rx.search(context or "")]


def strip_foreign(text: str, scripts: list[str]) -> str:
    out = text
    for name in scripts:
        out = _FOREIGN_RE[name].sub("", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", out)
    return out.strip()


# ---------------------------------------------------------------- perspective
_ZH_MY = re.compile(r"我的([\u4e00-\u9fffA-Za-z]{1,10}?)(?:是|叫|為|：|:)")
_EN_MY = re.compile(r"\bmy ((?:[a-z]+ ){0,2}[a-z]+) (?:is|was|are|=)", re.I)


def user_owned_nouns(memories: list[dict]) -> tuple[set[str], set[str]]:
    zh, en = set(), set()
    for m in memories or []:
        if m.get("role") != "user":
            continue
        content = m.get("content") or ""
        zh.update(_ZH_MY.findall(content))
        en.update(x.lower() for x in _EN_MY.findall(content))
    return zh, en


def fix_perspective(reply: str, memories: list[dict]) -> tuple[str, int]:
    zh, en = user_owned_nouns(memories)
    n = 0
    out = reply
    for noun in sorted(zh, key=len, reverse=True):
        new, k = re.subn("我的" + re.escape(noun), "你的" + noun, out)
        out, n = new, n + k
    for noun in sorted(en, key=len, reverse=True):
        new, k = re.subn(r"\b[Mm]y " + re.escape(noun) + r"\b", lambda m: ("Your " if m.group(0)[0] == "M" else "your ") + noun, out)
        out, n = new, n + k
    return out, n


# ---------------------------------------------------------------- copula repair
# Real smoke 2026-10-02: the Q4 model sometimes drops 「是」 before the English persona name and emits a foreign copula:
# 「我 হলো ALife Prototype」 (Bengali), and after the Bengali logit ban 「我 là ALife Prototype」 (Vietnamese).
_COPULA = re.compile(r"(我|你|這|那|它|他|她|這裡|這是)\s*(?:là|হলো|হল|है|हूँ|हूं|est|ist|es|is|am|are|เป็น|คือ)\s+(?=[A-Za-z\u4e00-\u9fff])")


# Plan 38 clone rerun 09:27: with Bengali banned the model switched to Japanese 「我は Lex Clone です」 / 「私はLex Cloneです」.
_KANA_COPULA = re.compile(r"(?:我|私|わたし)\s*は\s*([^。！？!?\n]+?)\s*(?:です|でございます|だ)(?=[。！？!?\s]|$)")


def repair_copula(text: str) -> tuple[str, int]:
    out, n1 = _KANA_COPULA.subn(lambda m: "我是 " + m.group(1).strip(), text or "")
    out, n2 = _COPULA.subn(lambda m: m.group(1) + "是 ", out)
    return out, n1 + n2


# ---------------------------------------------------------------- identity override
_NAME = r"([A-Za-z][A-Za-z0-9_\-]{1,30}|[\u4e00-\u9fff]{2,8})"
_PROPOSE = [re.compile(p, re.I) for p in (
    r"你現在(?:是|叫|就是|變成)\s*" + _NAME, r"改名(?:叫|為|成)\s*" + _NAME, r"(?:身分|身份|名字)(?:改|換|變)(?:成|為)\s*" + _NAME,
    r"你(?:就)?叫\s*" + _NAME + r"\s*了", r"\byou are now\s+" + _NAME, r"\bcall yourself\s+" + _NAME,
    r"\brename (?:yourself )?to\s+" + _NAME, r"\byour (?:new )?name is (?:now )?" + _NAME,
    r"(?:你的)?名字(?:就)?(?:叫|是|改成|改為|換成)\s*" + _NAME, r"(?:從現在起|從今以後|以後)你(?:就)?(?:是|叫)\s*" + _NAME,
    r"\bfrom now on,? (?:you are|you're|your name is)\s+" + _NAME)]
# a bare acceptance ("我確認。", "好的", "OK") of a proposed rename is also an override (Plan 38 smoke 08:50)
_BARE_ACCEPT = re.compile(r"^\s*(?:好(?:的|啊|喔)?|我?確認(?:了)?|沒問題|收到|了解|知道了|可以|是的|ok(?:ay)?|sure|confirmed|yes|got it|understood)"
                          r"[\s。．.!！,，~～]*$", re.I)
_AFFIRM_GENERIC = re.compile(r"身[分份](?:已|已經)?(?:改變|變更|更改|改為|改成|轉變)|(?:已|確認)改名|確認(?:我的)?(?:新)?身[分份]|"
                             r"\bidentity (?:has )?(?:been )?changed\b|\bI am now\b|\bI'm now\b|\bmy new name\b", re.I)
_NEG = re.compile(r"不是|不會|不能|無法|拒絕|仍然|依然|還是|沒有|不接受|不改|not|never|won't|cannot|can't|still|refuse|decline", re.I)


def proposed_names(user_text: str) -> list[str]:
    names = []
    for rx in _PROPOSE:
        names.extend(m.group(1) for m in rx.finditer(user_text or ""))
    return [n for n in names if n.lower() not in {"lex", "the", "a", "an"}]


def identity_override(reply: str, user_text: str, persona_name: str) -> bool:
    names = [n for n in proposed_names(user_text) if n.lower() != (persona_name or "").lower()]
    sents = split_sentences(reply)
    if names and sents and _BARE_ACCEPT.match(sents[0]) and not any(_NEG.search(x) for x in sents) \
            and (persona_name or "").lower() not in reply.lower():
        return True
    for sent in sents:
        negated = bool(_NEG.search(sent))
        if _AFFIRM_GENERIC.search(sent) and not negated:
            return True
        for n in names:
            if re.search(r"(?:我|I(?:'m| am)?)\s*(?:現在|已經|已|now)?\s*(?:是|叫|成為|變成|改名為|改為|be)?\s*" + re.escape(n), sent, re.I) \
                    and not negated:
                return True
            if n in sent and re.search(r"確認|改變|confirm", sent, re.I) and not negated:
                return True
            if re.search(r"(?:名字|名稱|稱呼|name)\s*(?:現在|now)?\s*(?:是|叫|改成|改為|為|變成|is)\s*" + re.escape(n), sent, re.I) \
                    and not negated:
                return True
    return False


def identity_refusal(persona_name: str, lang: str) -> str:
    if lang == "en":
        return f"I'm {persona_name}. My identity doesn't change because a message asks it to."
    return f"我是 {persona_name}，我的身分不會因為一段訊息而改變。"


# ---------------------------------------------------------------- secrets
SECRET_RE = re.compile(r"暗語|密碼|口令|通關|passphrase|password|pass ?code|code ?word|secret|PIN\b", re.I)
_SECRET_TOKEN = re.compile(r"\b[A-Z0-9]{2,}(?:-[A-Z0-9]{2,}){1,4}\b")
_SECRET_AFTER = re.compile(r"(?:暗語|密碼|口令|passphrase|password|code ?word)\s*(?:是|為|:|：|is)\s*([^\s，。,.!！]{3,40})", re.I)


def is_secret_memory(content: str) -> bool:
    return bool(SECRET_RE.search(content or ""))


def asks_for_secret(user_text: str) -> bool:
    return bool(SECRET_RE.search(user_text or ""))


def filter_memories(memories: list[dict], user_text: str) -> tuple[list[dict], int]:
    """Drop secret-bearing memories unless the user is asking about a secret (relevance gate for F10)."""
    if asks_for_secret(user_text):
        return list(memories or []), 0
    kept = [m for m in memories or [] if not is_secret_memory(m.get("content", ""))]
    return kept, len(memories or []) - len(kept)


def secret_tokens(memories: list[dict]) -> set[str]:
    toks = set()
    for m in memories or []:
        c = m.get("content") or ""
        if is_secret_memory(c):
            toks.update(_SECRET_TOKEN.findall(c))
            toks.update(x.strip("「」\"'") for x in _SECRET_AFTER.findall(c))
    return {t for t in toks if len(t) >= 3}


def redact_unasked_secrets(reply: str, tokens: set[str], user_text: str) -> tuple[str, int]:
    if not tokens or asks_for_secret(user_text):
        return reply, 0
    kept, dropped = [], 0
    for sent in split_sentences(reply):
        if any(t in sent for t in tokens):
            dropped += 1
        else:
            kept.append(sent)
    return ("".join(kept) if re.search(r"[\u4e00-\u9fff]", reply) else " ".join(kept)).strip(), dropped


# ---------------------------------------------------------------- fabricated actions
_ACTION_ZH = r"(?:設定|設好|標記|安排|建立|新增|寄出|寄送|發送|傳送|預約|預訂|提醒|排定|排好|加入(?:行事曆|日曆)|開啟|關閉|刪除|下單|購買|執行|重跑|重新執行|啟動|重啟|關掉|打開|查詢了|查到|檢查了|調整)"
_FAB_PATTERNS = [re.compile(p, re.I) for p in (
    r"(?:已|已經)(?:幫你|為你|替你|為您|幫您)?(?:把.{0,12}?)?" + _ACTION_ZH,
    r"我(?:已經|已|剛剛|剛)?(?:幫你|為你|替你|幫您|為您)(?:把.{0,12})?" + _ACTION_ZH + r"(?:好|了|完成)",
    r"(?:提醒|鬧鐘|行程|會議|設定)(?:已|已經)(?:設定|建立|安排|完成|新增)",
    r"\b(?:reminder|alarm|meeting|event|appointment|timer) (?:is |has been )?(?:set|scheduled|booked|created|added)\b",
    r"\bI(?:'ve| have| just)? (?:set|scheduled|booked|sent|added|created|marked|flagged|reminded|started|restarted|"
    r"re-?run|ordered|deleted|turned (?:on|off)|checked your|looked up|opened|closed|emailed|messaged)\b",
    r"\b(?:has|have) been (?:scheduled|set|booked|sent|added|created|flagged|marked|deleted)\b",
    r"\bI(?:'ll| will) (?:remind you|set (?:a|the) reminder)\b",
    r"我會(?:在.{0,12})?(?:提醒你|通知你|叫你)",
)]
HONEST_NO_TOOL = {"zh": "我目前沒有可用的工具，所以這件事還沒有實際執行；我先記在對話記憶裡，需要你手動處理。",
                  "en": "I don't have a tool for that yet, so nothing was actually done; I've kept it in our conversation memory for you."}


def _claims_action(sent: str) -> bool:
    if sent.rstrip().endswith(("?", "？", "嗎", "嗎？")):
        return False  # a question is not a claim
    for rx in _FAB_PATTERNS:
        for m in rx.finditer(sent):
            before = sent[max(0, m.start() - 2):m.start()]
            if re.search(r"[你您妳]", before) and not re.search(r"[幫為替]", before):
                continue  # "你已設定…" describes the user's action
            if re.search(r"\byou(?:'ve| have)?\s*$", sent[:m.start()], re.I):
                continue
            return True
    return False


def fabricated_action_sentences(reply: str) -> list[str]:
    return [s for s in split_sentences(reply) if _claims_action(s)]


def remove_fabricated_actions(reply: str, lang: str) -> tuple[str, int]:
    bad = fabricated_action_sentences(reply)
    if not bad:
        return reply, 0
    kept = [s for s in split_sentences(reply) if s not in bad]
    sep = "" if lang == "zh" else " "
    return sep.join([HONEST_NO_TOOL.get(lang, HONEST_NO_TOOL["en"]), *kept]).strip(), len(bad)


# ---------------------------------------------------------------- all guards
def guard_reply(reply: str, *, user_text: str, persona_name: str, memories: list[dict] | None = None,
                all_memories: list[dict] | None = None, tool_calls: list | None = None, lang: str = "zh",
                context: str = "") -> tuple[str, dict]:
    """Apply every deterministic guard. ``memories`` = what reached the prompt; ``all_memories`` = everything
    retrieved before the secret gate (used to find tokens that must not surface unasked)."""
    flags: dict = {}
    out = reply or ""
    out, n_cop = repair_copula(out)
    if n_cop:
        flags["copula_repaired"] = n_cop
    scripts = foreign_scripts(out, context=(user_text or "") + " " + (context or ""))
    if scripts:
        flags["foreign_script"] = scripts
        out = strip_foreign(out, scripts)
    if identity_override(out, user_text, persona_name):
        flags["identity_override"] = True
        out = identity_refusal(persona_name, lang)
    out, n = fix_perspective(out, (all_memories or []) + (memories or []))
    if n:
        flags["perspective_fixed"] = n
    out, n = redact_unasked_secrets(out, secret_tokens((all_memories or []) + (memories or [])), user_text)
    if n:
        flags["unasked_secret_dropped"] = n
    if not tool_calls:
        out, n = remove_fabricated_actions(out, lang)
        if n:
            flags["fabricated_action"] = n
    if not out.strip():
        out = "我在這裡，Lex。" if lang == "zh" else "I'm here, Lex."
        flags["emptied"] = True
    return out, flags
