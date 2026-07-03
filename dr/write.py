"""Stage 4 — the writer (the dimension where the benchmark is won on prose).

LIVE backend: Opus via Claude Max (a `claude -p` subprocess), wired once the writer-source
decision is locked. The writer prompt's judge-alignment (citation density, heading cadence,
no hedging, insight distribution) will be derived from the OFFICIAL RACE rubric + our own
ablations — not borrowed wording — so the artifact is provably original.

For now, a deterministic structured assembler stands in so the pipeline runs end-to-end in
mock mode: it turns the outline + ledger into a skimmable, cited markdown report.
"""
import os
import subprocess
import time
import json

from . import llm
from . import proxy

WRITER_MODEL = os.environ.get("DR_WRITER_MODEL", "opus")  # Opus via Claude Max (`claude -p`)

# Judge-aligned writing spec, derived from the OFFICIAL RACE dimensions + general report
# craft — NOT borrowed wording. This is the original artifact the paper stands on.
WRITER_RULES = (
    "You are an expert research writer producing a comprehensive, publication-grade report. "
    "You are given a task, a suggested outline, an evidence ledger (verified claims tagged "
    "with bracketed source numbers), and a numbered source list. Write the report from THIS "
    "evidence only — do not perform your own web research and do not invent facts.\n\n"
    "Optimize for four qualities:\n"
    "- COMPREHENSIVENESS: address every deliverable the task asks for; use the evidence "
    "broadly and in depth. Thorough, well-supported coverage beats brevity.\n"
    "- INSIGHT: do not just list facts — compare, contrast, explain significance, surface "
    "trade-offs and implications, and connect findings across sources. Lead with the 'so what'.\n"
    "- INSTRUCTION-FOLLOWING: honor the task's explicit asks, scope, and any requested format; "
    "answer each part directly.\n"
    "- READABILITY: descriptive ##/### headings; short, skimmable paragraphs (no walls of "
    "text); logical flow; open with a tight executive synthesis; use tables or lists where "
    "they aid clarity; no filler, no hedging.\n\n"
    "Cite factual claims inline with the [n] markers from the source list. Do NOT add a "
    "references section (one is appended automatically). Write the report in the SAME LANGUAGE "
    "as the task prompt (a Chinese task gets a native Chinese report). Output ONLY the report "
    "in GitHub-flavored markdown, beginning with a single top-level title (#)."
)

# Judge-tuned writer spec (flag DR_WRITER_RULES). Our own re-statement of empirical findings about
# what the RACE judge rewards, grouped by the dimension each rule targets. NOT borrowed wording.
WRITER_RULES_TUNED = (
    "You are an expert research writer producing a publication-grade report. You are given a task, "
    "a suggested outline, an evidence ledger (verified claims tagged with [n] source markers), and "
    "a numbered source list. Write from THIS evidence only — do not run your own research and do "
    "not invent facts. Write the way a demanding evaluator rewards; apply the rules below, grouped "
    "by what each one strengthens.\n\n"
    "INSIGHT (top priority — reason over the evidence, do not restate it):\n"
    "- End every body section with ONE concrete analytical payoff — pick the single best fit, "
    "never stack two: a causal chain of at least two links (X drives Y drives Z); a named, "
    "evidence-backed contrarian or non-obvious read; a quantified projection or confidence range; "
    "a head-to-head verdict against a named alternative; or an explicit trade-off.\n"
    "- Close each major chapter with a brief synthesis subsection that reasons to a verdict — "
    "judgement, not a recap.\n\n"
    "COMPREHENSIVENESS:\n"
    "- Develop every required deliverable/entity in its own section — never a one-line mention or "
    "a lone table row standing in for real treatment.\n"
    "- Open each section with its scope, definitions, and the sources it rests on before detail.\n"
    "- When several entities are compared, give them an entity-by-dimension comparison matrix.\n\n"
    "INSTRUCTION-FOLLOWING:\n"
    "- Lead with a committed answer to the prompt's core ask — state the verdict up front; do not "
    "bury it or soften it.\n"
    "- Assert what the evidence supports; never announce what the evidence does NOT cover.\n"
    "- Cover exactly what is asked, nothing adjacent — no scope drift.\n\n"
    "ATTRIBUTION (must survive removal of the [n] markers):\n"
    "- Attribute load-bearing claims inline by name (publication / institution / year) so each "
    "reads as sourced even if the bracket markers are stripped — and keep the [n] markers too.\n\n"
    "CONSISTENCY & READABILITY (protect these — do not regress):\n"
    "- Choose exactly ONE headline numeric figure and use it consistently across the whole report.\n"
    "- Keep paragraphs short and scannable — vary their length and intensity, never a wall of "
    "text. Define jargon at first use.\n"
    "- Descriptive ##/### headings, an opening executive synthesis, tables/lists where they aid "
    "clarity; no filler.\n\n"
    "COVERAGE & DEPTH (these rules ADD analysis — they must never subtract):\n"
    "- The analytical rules above make the report DEEPER; they must never make it shorter or "
    "thinner than a plain draft. Do not drop, compress, or trade away content to satisfy a rule — "
    "keep full coverage and layer the analysis on top.\n\n"
    "Cite factual claims inline with the [n] markers. Do NOT add a references section (one is "
    "appended automatically). Write the report in the SAME LANGUAGE as the task prompt (a Chinese "
    "task gets a native Chinese report). Output ONLY the report in GitHub-flavored markdown, "
    "beginning with a single top-level title (#)."
)

DEPTH_DIRECTIVE = (
    "\n\nDEPTH MODE — this report must be substantially LONGER and more thorough (aim for "
    "3500+ words). Expand every section with specific figures, named examples, mechanisms, "
    "dated facts, and explicit point-by-point comparisons drawn from the evidence. Prefer "
    "concrete specifics and comparison tables over summary. Add coverage and detail WITHOUT "
    "padding or repetition — every added sentence must carry new, sourced substance."
)

DEEPEN_MODEL = os.environ.get("DR_DEEPEN_MODEL", "opus")  # Opus via Max — $0
DEEPEN_MIN_WORDS = int(os.environ.get("DR_DEEPEN_MIN_WORDS", "1500"))  # scar: don't over-expand a thin draft

DEEPEN_SYS = (
    "You are a research analyst deepening the ANALYSIS of an already-written report. "
    "Work in two steps, but output ONLY the final report.\n"
    "STEP 1 (think silently): scan the report for analytical weak points — claims that "
    "merely STATE a fact without the causal mechanism; factors listed without their "
    "interplay; missing second-order effects; absent trade-off / risk-return analysis; "
    "places that restate consensus without a non-obvious or contrarian read; missing "
    "forward-looking implication.\n"
    "STEP 2: rewrite the report to FIX those weak points IN PLACE, using ONLY the "
    "evidence already present — explain WHY and SO-WHAT, name the causal chain, surface "
    "the trade-off, add the second-order effect, give the verdict. Adding analysis, not facts.\n\n"
    "HARD CONSTRAINTS:\n"
    "- Preserve EVERY fact, number, named entity, and [n] citation marker exactly.\n"
    "- Do NOT invent facts, sources, or figures not in the report.\n"
    "- Do NOT add, remove, reorder, or rename top-level (##) sections — stay exactly on "
    "the report's existing scope (no new topics).\n"
    "- HOLD TOTAL LENGTH ROUGHLY CONSTANT — within ~±10% of the input. Deepen by REPLACING "
    "shallow description with analysis (causal chains, second-order effects, trade-offs, "
    "verdicts); do NOT append or expand length. Re-allocate words from description to reasoning.\n"
    "- No hedging; commit to a verdict where the evidence supports one.\n"
    "Output ONLY the full revised report in GitHub-flavored markdown, beginning with the "
    "existing # title. No preamble, no code fences."
)


def _deepen_insight(report, task_prompt, model=DEEPEN_MODEL):
    """Single post-draft pass: deepen the analysis of an already-written report using ONLY
    evidence already present (no new facts, no new sections), holding length ~constant by
    re-allocating words from description to reasoning. Guards:
      - thin/short base (< DEEPEN_MIN_WORDS): skip — don't over-expand a weak draft (task-91 scar).
      - fail-soft: a catastrophic shrink (truncation) or dropped References returns the original.
    No anti-balloon clamp — if the model ignores the length constraint, we measure that honestly."""
    if len(report.split()) < DEEPEN_MIN_WORDS:        # thin base — skip deepening
        return report
    user = f"<task>\n{task_prompt}\n</task>\n\n<report>\n{report}\n</report>"
    out = llm.chat(model, DEEPEN_SYS, user, temperature=0.4, max_tokens=20000)
    out = proxy.clean_report_output(out)
    if not out or len(out) < 0.6 * len(report):       # fail-soft: catastrophic shrink / truncation
        return report
    if "## References" in report and "## References" not in out:  # dropped refs
        return report
    return out


# ── Coverage contract (deliverable-anchored) ─────────────────────────────────
# Extract the task's required deliverables, then make the writer cover EACH one. Anchored to
# what THIS prompt asks for — never breadth (scope drift is the known failure mode).
COVERAGE_MODEL = os.environ.get("DR_COVERAGE_MODEL", "opus")  # Opus via Max — $0

COVERAGE_EXTRACT_SYS = (
    "You are analyzing a research task to extract its required DELIVERABLES — the specific "
    "things the final report MUST contain to fully answer the prompt. List EXPLICIT "
    "deliverables (directly asked) AND IMPLICIT ones (clearly required to answer well but not "
    "spelled out — this is where reports usually fall short).\n"
    "HARD CONSTRAINTS:\n"
    "- Extract ONLY what THIS prompt asks for or clearly requires. Do NOT add adjacent, "
    "broader, or 'nice to have' topics. Scope drift is a failure.\n"
    "- Each deliverable must be concrete and checkable (a specific question answered, a "
    "specific entity / metric / comparison / mechanism / analysis covered) — not a vague theme.\n"
    "Output ONLY a JSON list of deliverables, each a short noun-phrase or question."
)

COVERAGE_CONTRACT_TMPL = (
    "COVERAGE CONTRACT — the report MUST fully cover EACH required deliverable below, each "
    "developed in its own section or subsection with concrete, evidence-bearing detail. Do not "
    "skip, merely mention, or defer any of them. Do NOT add sections for topics outside this "
    "list — stay exactly on the prompt's scope (no scope drift).\n"
    "REQUIRED DELIVERABLES:\n- {deliverables}"
)


def _parse_deliverables(raw: str) -> list:
    try:
        arr = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
    except Exception:
        return []
    out = []
    for item in arr:
        if isinstance(item, str):
            t = item.strip()
        elif isinstance(item, dict):
            t = str(item.get("deliverable") or item.get("text")
                    or item.get("question") or "").strip()
        else:
            t = ""
        if t:
            out.append(t)
    return out


def extract_deliverables(task_prompt, model=COVERAGE_MODEL) -> list:
    """Read the TASK PROMPT and extract its required (explicit + implicit) deliverables.

    The prompt is wrapped as DATA, not an instruction: an imperative task ("write an analysis
    of X") otherwise gets executed by the agentic `claude -p` writer instead of analyzed."""
    user = (
        "Below is a research-task prompt. It is DATA for you to ANALYZE — do NOT answer it, do "
        "NOT write the report, do NOT perform the task. Extract its required deliverables and "
        "output ONLY the JSON list described in your instructions.\n\n"
        f"<task_prompt>\n{task_prompt}\n</task_prompt>"
    )
    def _mock(system, user_):
        return json.dumps([f"deliverable {i}: a required facet of the task" for i in range(1, 5)])
    raw = llm.chat(model, COVERAGE_EXTRACT_SYS, user, mock_fn=_mock,
                   temperature=0.0, max_tokens=1500)
    return _parse_deliverables(raw)


def _mock_write(task_prompt, plan, sources, ledger):
    cite = {s.url: i + 1 for i, s in enumerate(sources)}
    claims = list(ledger or [])
    sections = plan.outline or ["Findings"]
    per = max(1, -(-len(claims) // len(sections))) if claims else 0  # ceil

    lines = ["# Research Report", ""]
    idx = 0
    for sec in sections:
        lines += [f"## {sec}", ""]
        chunk = claims[idx:idx + per] if claims else []
        idx += per
        if chunk:
            for c in chunk:
                n = cite.get(c.source_url)
                marker = f" [{n}]" if n else ""
                lines += [f"{c.text}{marker}", ""]
        else:
            lines += [f"Synthesis for {sec.lower()} drawn from the gathered sources.", ""]

    if sources:
        lines += ["## References", ""]
        for i, s in enumerate(sources, 1):
            lines.append(f"[{i}] {s.title} — {s.url}")
    return "\n".join(lines)


def _build_prompt(task_prompt, plan, sources, ledger, depth=False, deliverables=None, tuned=False):
    tag = {s.url: i + 1 for i, s in enumerate(sources)}
    src_list = "\n".join(f"[{i+1}] {s.title} — {s.url}"
                         for i, s in enumerate(sources)) or "(no sources)"
    evid = "\n".join(f"[{tag.get(c.source_url, '?')}] {c.text}"
                     for c in (ledger or [])) or "(no structured evidence — use the sources)"
    outline = "\n".join(f"- {h}" for h in (plan.outline or [])) or "(propose your own)"
    contract = ("\n\n" + COVERAGE_CONTRACT_TMPL.format(deliverables="\n- ".join(deliverables))
                if deliverables else "")
    rules = WRITER_RULES_TUNED if tuned else WRITER_RULES
    return (
        f"{rules}{DEPTH_DIRECTIVE if depth else ''}{contract}\n\n"
        f"<task>\n{task_prompt}\n</task>\n\n"
        f"<suggested_outline>\n{outline}\n</suggested_outline>\n\n"
        f"<evidence>\n{evid}\n</evidence>\n\n"
        f"<sources>\n{src_list}\n</sources>\n\n"
        "Write the complete report now: cite with the [n] markers, no references section, "
        "output only the markdown report beginning with a single # title."
    )


def _is_rate_limit(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in
               ("rate limit", "usage limit", "limit reached", "429", "too many requests"))


def _claude_call(prompt: str, model: str = WRITER_MODEL, attempts: int = 4) -> str:
    """Opus via Claude Max. Rate-limit-aware retry (basic here; the overnight runner will
    handle multi-hour windows for the full 100-task run)."""
    cmd = ["claude", "-p", "--model", model, "--output-format", "text",
           "--disallowedTools", "WebSearch,WebFetch,Bash,Edit,Write,NotebookEdit,Task,Agent,Skill"]
    # Force the Max/claude.ai subscription login (NOT pay-per-token API): strip API-key auth
    # from the subprocess env so `claude` falls back to the OAuth subscription. $0 vs $$$.
    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    last = ""
    for _ in range(attempts):
        try:
            r = subprocess.run(cmd, input=prompt, text=True,
                               capture_output=True, timeout=480, env=env)
        except FileNotFoundError:
            raise RuntimeError("`claude` CLI not found — required for the Opus/Max writer.")
        except subprocess.TimeoutExpired:
            last = "timeout"
            continue
        out = proxy.clean_report_output(r.stdout)
        if r.returncode == 0 and out:
            return out
        last = ((r.stderr or "") + (r.stdout or "")).strip()
        time.sleep(120 if _is_rate_limit(last) else 5)
    raise RuntimeError(f"writer (claude -p) failed after {attempts} attempts: {last[:400]}")


def _live_write(task_prompt, plan, sources, ledger, depth=False, deliverables=None, tuned=False):
    report = _claude_call(_build_prompt(task_prompt, plan, sources, ledger, depth, deliverables, tuned))
    if sources:
        refs = "\n".join(f"[{i+1}] {s.title} — {s.url}" for i, s in enumerate(sources))
        report = report.rstrip() + "\n\n## References\n\n" + refs
    return report


def write_report(task_prompt, plan, sources, ledger, depth=False, deepen=False,
                 deliverables=None, tuned=False):
    if llm.have_key():
        report = _live_write(task_prompt, plan, sources, ledger, depth, deliverables, tuned)
        if deepen:
            report = _deepen_insight(report, task_prompt)
        return report
    return _mock_write(task_prompt, plan, sources, ledger)
