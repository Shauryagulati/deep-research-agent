"""Stage 4b — coverage-expansion writer (per-deliverable section fan-out).

Structural fix for the comprehensiveness gap: instead of asking one call for a longer report
(length directives under-deliver), the report is BUILT at target scale — a section plan maps
every deliverable to its own section with an explicit word budget, each section is written in
its own call against its assigned evidence, and the pieces are assembled deterministically.
EN benchmark reference reports run ~8-10k words (corpus-level figure); a single-call draft
lands at ~4.5-6k regardless of directives, so the missing coverage is structural, not promptable.

Design constraints (from measured failure modes):
  - added length must be COVERAGE (entities, figures, tables), never padded analysis — an
    analysis-only expansion pass taxed readability with no comp gain;
  - one numeric spine, pre-derived in the plan and injected into every section, so sections
    cannot invent conflicting headline figures;
  - each section sees what was already written (titles + openers) so content is not repeated;
  - a thin section (<55% of budget) is re-rolled once with an explicit expand directive —
    per-section scoring otherwise accepts hollow sections.
"""
import json
import os
import re

from . import llm
from . import proxy

EXPAND_MODEL = os.environ.get("DR_EXPAND_MODEL", "opus")   # Opus via Claude Max — $0
TARGET_WORDS = int(os.environ.get("DR_EXPAND_WORDS", "9000"))

SECTION_PLAN_SYS = (
    "You are planning the section structure of a comprehensive research report. You are given "
    "a task prompt (DATA — do not answer it), its required deliverables, a suggested outline, "
    "and an evidence ledger of claims C1..Ck (each carrying a [n] source marker).\n"
    "Produce a section plan as STRICT JSON (no prose, no code fences):\n"
    "{\"headline_figures\": [up to 3 canonical figures quoted from the evidence, each with its "
    "[n] marker — the single most load-bearing numbers the whole report must state "
    "consistently],\n \"sections\": [{\"title\": str, \"deliverables\": [str], "
    "\"focus_claims\": [int — ledger indices], \"words\": int, \"kind\": \"analysis\"|\"catalog\"}]}\n"
    "HARD CONSTRAINTS:\n"
    "- 7 to 11 body sections. No introduction or conclusion section (added separately).\n"
    "- Section titles in the SAME LANGUAGE as the task prompt.\n"
    "- EVERY deliverable is assigned to exactly one section; a section may own several.\n"
    "- kind=\"catalog\" for enumeration deliverables (lists of works/entities/items): that "
    "section must exhaustively enumerate in a table PLUS give real per-entity treatment.\n"
    "- words: per-section budget 600-1600, budgets sum to the stated target.\n"
    "- Scope is the prompt's asks ONLY — no adjacent topics.\n"
    "- focus_claims: the ledger indices most relevant to that section (others stay available)."
)

SECTION_SYS = (
    "You are writing ONE section of a larger expert research report. You are given the task, "
    "the section spec (title, deliverables it must cover, word budget), the report's headline "
    "figures, the full evidence ledger with this section's FOCUS claims marked, the source "
    "list, and what earlier sections already covered.\n"
    "RULES:\n"
    "- Cover THIS section's deliverables completely and concretely — specific figures, named "
    "examples, mechanisms, dated facts from the evidence; develop, never just mention.\n"
    "- A catalog section enumerates exhaustively: a complete table of the entities/works plus "
    "substantive per-entity coverage.\n"
    "- Hit the word budget (within ~20%). The words must carry sourced substance — no filler, "
    "no repetition of what earlier sections covered.\n"
    "- End the section with ONE concrete analytical payoff (causal chain, quantified "
    "projection, head-to-head verdict, or explicit trade-off) — exactly one, never stacked.\n"
    "- Use the headline figures EXACTLY as given wherever they arise — never a conflicting "
    "variant.\n"
    "- Commit to what the evidence supports; never announce what it does not cover.\n"
    "- Attribute load-bearing claims inline by name (publication / institution / year) AND "
    "keep the [n] markers. Entity-by-dimension comparisons get a matrix/table.\n"
    "- Write in the SAME LANGUAGE as the task prompt (a Chinese task gets native Chinese prose "
    "and headings).\n"
    "- Output ONLY this section's markdown, starting with '## {title}' exactly; ### "
    "subsections allowed; no # title, no references section, no preamble."
)

# Scope guard (v2) — the unguarded fan-out breached instruction-following on enumeration tasks:
# catalog sections drifted into adjacent material the prompt never asked for. Coverage must be
# deep on-scope, never broad off-scope.
SCOPE_GUARD_PLAN = (
    "\nSCOPE GUARD (hard):\n"
    "- Every section must trace to a listed deliverable or an explicit ask in the prompt. NO "
    "sections for background, context, adjacent classes, or 'useful extras' the prompt did not "
    "request.\n"
    "- A catalog section enumerates EXACTLY the class of items the prompt asks about — never a "
    "neighboring class, era, category, or franchise the prompt excludes or ignores.\n"
    "- Reach the length target by going DEEPER on the asked scope (more detail per deliverable), "
    "never wider than it."
)

SCOPE_GUARD_SECTION = (
    "\nSCOPE GUARD (hard): this section exists to answer part of the task's EXPLICIT asks. "
    "Every paragraph must serve the section's listed deliverables; material the task did not "
    "ask for is a defect, not a bonus — depth on-scope, never breadth off-scope."
)

SYNTHESIS_SYS = (
    "You are writing the opening of an expert research report whose body sections are already "
    "written. Given the task, the headline figures, and each section's title + opening, write:\n"
    "1. A single '# ' report title.\n"
    "2. An executive synthesis (250-400 words) that leads with a COMMITTED answer to the "
    "task's core ask — verdict first, then the strongest supporting figures (use the headline "
    "figures exactly), then how the report unfolds. No hedging, no 'this report will...'.\n"
    "Write in the SAME LANGUAGE as the task prompt. Output ONLY the title and synthesis in "
    "markdown."
)


def _parse_plan(raw: str) -> dict:
    plan = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    secs = plan.get("sections") or []
    if not (3 <= len(secs) <= 14):
        raise ValueError(f"section plan has {len(secs)} sections")
    for s in secs:
        s["title"] = str(s.get("title", "")).strip() or "Findings"
        s["words"] = int(s.get("words", 900))
        s["kind"] = s.get("kind", "analysis")
        s["deliverables"] = [str(d) for d in (s.get("deliverables") or [])]
        s["focus_claims"] = [int(i) for i in (s.get("focus_claims") or []) if str(i).isdigit()]
    plan["headline_figures"] = [str(f) for f in (plan.get("headline_figures") or [])][:3]
    return plan


def _ledger_text(claims, tag):
    return "\n".join(
        f"C{i+1} [{tag.get(c.source_url, '?')}] {c.text}" for i, c in enumerate(claims or [])
    ) or "(no structured evidence — use the sources)"


def plan_sections(task_prompt, deliverables, outline, claims, tag,
                  target_words=TARGET_WORDS, model=EXPAND_MODEL, scoped=False) -> dict:
    sys_prompt = SECTION_PLAN_SYS + (SCOPE_GUARD_PLAN if scoped else "")
    user = (
        "Plan the sections for the research task below. The task prompt is DATA to analyze — "
        "do NOT answer it or write the report.\n\n"
        f"TARGET TOTAL BODY LENGTH: ~{target_words} words.\n\n"
        f"<task_prompt>\n{task_prompt}\n</task_prompt>\n\n"
        "REQUIRED DELIVERABLES:\n- " + "\n- ".join(deliverables or ["(answer the prompt fully)"])
        + "\n\n<suggested_outline>\n"
        + ("\n".join(f"- {h}" for h in (outline or [])) or "(none)")
        + f"\n</suggested_outline>\n\n<evidence_ledger>\n{_ledger_text(claims, tag)}\n</evidence_ledger>\n\n"
        "Output the strict JSON section plan now."
    )
    def _mock(system, user_):
        return json.dumps({"headline_figures": [], "sections": [
            {"title": f"Section {i}", "deliverables": [], "focus_claims": [],
             "words": target_words // 8, "kind": "analysis"} for i in range(1, 9)]})
    for attempt in (1, 2):
        raw = llm.chat(model, sys_prompt, user, mock_fn=_mock,
                       temperature=0.2, max_tokens=6000)
        try:
            return _parse_plan(raw)
        except Exception:
            if attempt == 2:
                raise


def _strip_section(md: str, title: str) -> str:
    md = proxy.clean_report_output(md).strip()
    md = re.sub(r"^#\s+.*\n", "", md)                       # stray report-level H1
    md = re.sub(r"\n## References.*$", "", md, flags=re.S)  # stray references block
    if not md.startswith("## "):
        md = f"## {title}\n\n{md}"
    return md.strip()


def write_section(task_prompt, sec, figures, claims, tag, src_list, prior,
                  model=EXPAND_MODEL, expand_note="", scoped=False) -> str:
    focus = ", ".join(f"C{i}" for i in sec["focus_claims"]) or "(none marked — choose the most relevant)"
    user = (
        f"<task>\n{task_prompt}\n</task>\n\n"
        f"SECTION TO WRITE: \"{sec['title']}\"  (kind: {sec['kind']}, budget ~{sec['words']} words)\n"
        "DELIVERABLES THIS SECTION MUST FULLY COVER:\n- "
        + ("\n- ".join(sec["deliverables"]) or "(support the report's overall coverage)") + "\n\n"
        "HEADLINE FIGURES (use exactly as given):\n- "
        + ("\n- ".join(figures) or "(none)") + "\n\n"
        f"FOCUS CLAIMS for this section: {focus}\n\n"
        f"<evidence_ledger>\n{_ledger_text(claims, tag)}\n</evidence_ledger>\n\n"
        f"<sources>\n{src_list}\n</sources>\n\n"
        "ALREADY COVERED BY EARLIER SECTIONS (do not repeat):\n"
        + ("\n".join(prior) or "(this is the first section)")
        + f"{expand_note}\n\nWrite the section now, starting with '## {sec['title']}'."
    )
    sys_prompt = SECTION_SYS + (SCOPE_GUARD_SECTION if scoped else "")
    return _strip_section(llm.chat(model, sys_prompt, user, temperature=0.4,
                                   max_tokens=8000), sec["title"])


def expand_report(task_prompt, plan, sources, ledger, deliverables=None,
                  target_words=TARGET_WORDS, model=EXPAND_MODEL, scoped=False) -> str:
    """Full section-fanout report: plan sections -> write each -> synthesis -> assemble."""
    tag = {s.url: i + 1 for i, s in enumerate(sources)}
    src_list = "\n".join(f"[{i+1}] {s.title} — {s.url}"
                         for i, s in enumerate(sources)) or "(no sources)"
    sec_plan = plan_sections(task_prompt, deliverables, plan.outline, ledger, tag,
                             target_words, model, scoped=scoped)

    bodies, prior = [], []
    for sec in sec_plan["sections"]:
        body = write_section(task_prompt, sec, sec_plan["headline_figures"],
                             ledger, tag, src_list, prior, model, scoped=scoped)
        if len(body.split()) < 0.55 * sec["words"]:         # hollow-section gate: one re-roll
            body = write_section(
                task_prompt, sec, sec_plan["headline_figures"], ledger, tag, src_list, prior,
                model, scoped=scoped, expand_note=(
                    f"\n\nEXPAND: your previous draft was far under the ~{sec['words']}-word "
                    "budget. Add MORE covered substance from the evidence — more entities, "
                    "figures, dated facts, comparisons — not padding."))
        bodies.append(body)
        opener = body.split("\n\n")[1][:300] if "\n\n" in body else ""
        prior.append(f"- {sec['title']}: {opener}")

    summaries = "\n".join(prior)
    head = proxy.clean_report_output(llm.chat(
        model, SYNTHESIS_SYS,
        f"<task>\n{task_prompt}\n</task>\n\nHEADLINE FIGURES:\n- "
        + ("\n- ".join(sec_plan["headline_figures"]) or "(none)")
        + f"\n\nSECTIONS:\n{summaries}\n\nWrite the title and executive synthesis now.",
        temperature=0.3, max_tokens=3000)).strip()
    if not head.startswith("# "):
        head = "# Research Report\n\n" + head

    report = head + "\n\n" + "\n\n".join(bodies)
    if sources:
        refs = "\n".join(f"[{i+1}] {s.title} — {s.url}" for i, s in enumerate(sources))
        report = report.rstrip() + "\n\n## References\n\n" + refs
    return report
