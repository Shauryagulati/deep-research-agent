"""Deterministic, model-free reward proxy + structural transforms.

Free (no API). Built from observed structural correlates of report quality (readability is
driven by wall-of-text density, over-length, missing skim structure). Gives a real signal
with no key. It must be validated against the real judge before being trusted, and dropped
if it does not correlate — it has to track genuine quality, not memorize any fixed shape.
"""
import re
import statistics

_WORD = re.compile(r"\w+")
_SENT = re.compile(r"(?<=[.!?。！？])\s+")


def _wc(s: str) -> int:
    return len(_WORD.findall(s))


def _body_paragraphs(report: str):
    return [p.strip() for p in report.split("\n\n")
            if p.strip() and not p.lstrip().startswith("#")]


def metrics(report: str) -> dict:
    paras = _body_paragraphs(report)
    pw = [_wc(p) for p in paras] or [0]
    return {
        "words": _wc(report),
        "h2": len(re.findall(r"(?m)^##\s", report)),
        "h3": len(re.findall(r"(?m)^###\s", report)),
        "paragraphs": len(paras),
        "median_para_words": statistics.median(pw),
        "max_para_words": max(pw),
        "wall_of_text_paras": sum(1 for w in pw if w > 150),
        "citation_markers": len(re.findall(r"\[\^?[\w./:\-]+\]", report)),
    }


def readability_proxy(report: str) -> float:
    """Crude 0..1 readability estimate. Penalizes walls of text, over-long paragraphs,
    and missing section structure. Deterministic; for guiding/triaging only."""
    m = metrics(report)
    score = 1.0
    if m["paragraphs"]:
        score -= 0.5 * (m["wall_of_text_paras"] / m["paragraphs"])
    if m["median_para_words"] > 120:
        score -= 0.2
    elif m["median_para_words"] > 90:
        score -= 0.1
    if m["h2"] == 0:
        score -= 0.3
    return round(max(0.0, min(1.0, score)), 4)


def clean_report_output(text: str) -> str:
    """Extract the report markdown from a model's raw output: unwrap a whole-report code
    fence (tolerating chatty preamble before it and the closing line) and drop any preamble
    before the first heading. Leaves already-clean markdown untouched."""
    t = (text or "").strip()
    m = re.search(r"```[a-zA-Z]*[ \t]*\n(.*)\n```[ \t]*$", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    if not t.startswith("#"):                 # drop conversational preamble before the report
        i = t.find("\n# ")
        if i != -1:
            t = t[i + 1:].strip()
    return t


def split_long_paragraphs(text: str, max_words: int = 150, target_words: int = 70) -> str:
    """Structural readability transform: break wall-of-text paragraphs into skimmable
    ~target_words chunks at sentence boundaries. Headings/short blocks untouched. Used as the
    deterministic stand-in for a readability patch in mock mode (and a cheap real aid)."""
    out = []
    for block in text.split("\n\n"):
        b = block.strip()
        if not b or b.lstrip().startswith("#") or _wc(b) <= max_words:
            out.append(block)
            continue
        chunks, cur, cur_w = [], [], 0
        for sent in _SENT.split(b):
            w = _wc(sent)
            if cur and cur_w + w > target_words:
                chunks.append(" ".join(cur))
                cur, cur_w = [], 0
            cur.append(sent)
            cur_w += w
        if cur:
            chunks.append(" ".join(cur))
        out.append("\n\n".join(chunks))
    return "\n\n".join(out)
