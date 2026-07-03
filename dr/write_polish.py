"""Stage 4c — whole-article readability polish (content-preserving, deterministically guarded).

The expanded (reference-scale) drafts win comprehensiveness/insight but pay a readability tax:
dense equal-intensity paragraphs, cross-section repetition, catalog sections that read as data
dumps. One whole-article edit pass fixes presentation ONLY. Acceptance is deterministic — the
pass must keep the covered material (words, tables, citations, structure) inside hard bands or
the original draft ships unchanged (fail-soft). Prompt asks alone are not trusted to protect
content; the guards are the contract.
"""
import os
import re

from . import llm
from . import proxy

POLISH_MODEL = os.environ.get("DR_POLISH_MODEL", "opus")   # Opus via Claude Max — $0

POLISH_SYS = (
    "You are a demanding line editor improving the READABILITY of a finished expert research "
    "report. Edit presentation only — the covered substance is untouchable.\n"
    "IMPROVE:\n"
    "- Paragraph rhythm: break dense walls of text; vary paragraph length and intensity; one "
    "idea per paragraph. Never let two long data-heavy paragraphs sit back to back.\n"
    "- Flow: crisp transitions inside sections; kill repeated setup phrasing; each section "
    "opens by advancing the argument, not re-explaining context already established.\n"
    "- Repetition: where two passages state the SAME point, keep the better one and replace "
    "the other with a one-clause reference forward or back.\n"
    "- Tables: give each a one-sentence lead-in stating what to take from it; keep every row.\n"
    "- Prose: cut filler and throat-clearing; keep the committed, no-hedging register.\n"
    "PRESERVE EXACTLY (non-negotiable):\n"
    "- Every fact, number, named entity, date, and [n] citation marker.\n"
    "- Every table and every table row.\n"
    "- Every ## section, its title, and its order; the # title; the analytical payoff that "
    "closes each section.\n"
    "- Total length within roughly 85-100% of the input — trim only genuine repetition and "
    "filler; NEVER summarize away content.\n"
    "Output ONLY the full edited report in GitHub-flavored markdown, starting with the "
    "existing # title. No preamble, no commentary, no code fences."
)


def _stats(txt: str):
    return (len(txt.split()),
            len(re.findall(r"^\|.*\|\s*$", txt, flags=re.M)),
            len(re.findall(r"\[\d+\]", txt)),
            len(re.findall(r"^## ", txt, flags=re.M)),
            len(re.findall(r"^# ", txt, flags=re.M)))


def polish_report(report: str, task_prompt: str, model=POLISH_MODEL) -> str:
    """Whole-article readability edit with deterministic accept/revert. The references block is
    detached before editing and re-appended verbatim, so the guard bands measure the body only."""
    body, refs = report, ""
    m = re.search(r"\n## References\s*\n", report)
    if m:
        body, refs = report[:m.start()], report[m.start():]

    out = llm.chat(model, POLISH_SYS,
                   f"<task>\n{task_prompt}\n</task>\n\n<report>\n{body}\n</report>\n\n"
                   "Edit the report for readability now, preserving all substance.",
                   temperature=0.3, max_tokens=16000)
    out = proxy.clean_report_output(out).strip()

    w0, tbl0, cit0, h20, _ = _stats(body)
    w1, tbl1, cit1, h21, h11 = _stats(out)
    ok = (out
          and 0.75 * w0 <= w1 <= 1.05 * w0        # no truncation, no summarizing away
          and tbl1 >= 0.9 * tbl0                  # enumeration tables intact
          and cit1 >= 0.7 * cit0                  # citations substantially intact
          and h21 >= h20 - 1                      # section structure intact
          and h11 == 1)                           # exactly one title
    return (out + "\n" + refs).rstrip() + "\n" if ok else report
