"""Stage 3 (ablation flag) — the claim ledger: ENRICHMENT, not filtering.

RACE rewards coverage, so this stage's job is to SURFACE more specific, sourced, quantified
claims for the writer to use — lifting Comprehensiveness and Insight (the dimensions that
actually count; FACT is blank on the live board). We only drop claims a source plainly does
not support; the default is to add depth, never to prune the report thinner.
"""
from dataclasses import dataclass
import os
import json

from . import llm

LEDGER_MODEL = os.environ.get("DR_LEDGER_MODEL", "deepseek/deepseek-chat")

LEDGER_SOURCE_SYS = (
    "Extract the specific, verifiable, quantified claims from the SINGLE source below that are "
    "relevant to the task — aim for 4-8 claims. Prefer concrete facts, numbers, named entities, "
    "dates, and figures over generalities. Do NOT invent anything the source does not support. "
    'Output ONLY a JSON list of strings: ["claim one", "claim two", ...].'
)


@dataclass
class Claim:
    text: str
    source_url: str
    support: str = ""


def _mock_ledger(sources):
    claims = []
    for s in sources:
        first = s.text.strip().split(".")[0][:160].strip()
        if first:
            claims.append(Claim(text=f"{first}.", source_url=s.url, support=s.text[:240]))
    return claims


def _parse_source_claims(raw: str, source):
    """Parse one source's claims. Accepts ["str", ...] or [{"text": "..."}, ...]."""
    try:
        arr = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
    except Exception:
        return []
    out = []
    for item in arr:
        text = item if isinstance(item, str) else (
            item.get("text") if isinstance(item, dict) else None)
        if text and str(text).strip():
            out.append(Claim(text=str(text).strip(), source_url=source.url,
                             support=source.text[:240]))
    return out


def build_ledger(sources, task_prompt: str, model: str = LEDGER_MODEL):
    """Per-source extraction so claims scale with the number of sources — a single global call
    caps at ~8-10 claims regardless of the prompt. Each source gets a dedicated extraction."""
    if not llm.have_key():
        return _mock_ledger(sources)
    claims = []
    for s in sources:
        user = (f"<task>\n{task_prompt}\n</task>\n\n"
                f"<source>\n{s.title}\nURL: {s.url}\n{s.text[:4000]}\n</source>")
        raw = llm.chat(model, LEDGER_SOURCE_SYS, user, temperature=0.2, max_tokens=1500)
        claims.extend(_parse_source_claims(raw, s))
    return claims or _mock_ledger(sources)
