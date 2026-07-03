"""Stage 5 — per-task candidate selection (criteria-anchored, within-task pairwise).

Every writer policy measured so far wins on some tasks and loses on others, so the report that
ships should be CHOSEN per task, not fixed per system. Selection is a king-of-the-hill pairwise
tournament judged against the task's own weighted evaluation criteria. No reference material is
involved — the selector generalizes to any task that has a rubric.

v2 design: a HOLISTIC pairwise verdict was validated against on-disk official scores and FAILED
with a systematic length bias (the longer candidate won 5/5 tournaments, capture −5%). The fix
is structural, not a prompt plea: each duel must return a SEPARATE verdict per evaluation
dimension, and the winner is computed in Python from the task's official dimension weights —
so the readability and instruction-following verdicts (which pull against length) carry their
real vote instead of being absorbed into an overall impression.
"""
import os
import re

from . import llm

SELECT_MODEL = os.environ.get("DR_SELECT_MODEL", "opus")   # Opus via Claude Max — $0

DIMS = ["comprehensiveness", "insight", "instruction_following", "readability"]

DUEL_SYS = (
    "You are a demanding evaluator comparing two research reports written for the same task. "
    "You are given the task prompt, its evaluation criteria grouped in four dimensions, and "
    "the two reports.\n"
    "Judge each dimension INDEPENDENTLY, as if it were the only thing that mattered:\n"
    "- comprehensiveness: which report covers more of the task's required material, in more "
    "substantive depth? (More text is NOT more coverage — judge covered substance.)\n"
    "- insight: which report reasons better over its material — causal chains, committed "
    "verdicts, trade-offs — rather than restating facts?\n"
    "- instruction_following: which report answers exactly what was asked, at the asked scope, "
    "with every explicit requirement met and nothing off-scope?\n"
    "- readability: which report is easier to read END TO END — paced, varied, non-repetitive, "
    "well-signposted? Length that adds fatigue or repetition LOSES this dimension.\n"
    "A dimension verdict may disagree with the others — that is expected and correct. Inline "
    "bracket citations are irrelevant; ignore citation formatting.\n"
    "Output STRICT JSON only: {\"comprehensiveness\": \"A\" or \"B\", \"insight\": \"A\" or "
    "\"B\", \"instruction_following\": \"A\" or \"B\", \"readability\": \"A\" or \"B\"}."
)


def _criteria_block(crit: dict) -> str:
    lines = []
    for dim, items in (crit.get("criterions") or {}).items():
        lines.append(f"\n{dim.upper()}:")
        for it in items:
            lines.append(f"- ({it.get('weight')}) {it.get('criterion')}: {it.get('explanation','')}")
    return "\n".join(lines)


def duel(task_prompt: str, crit: dict, rep_a: str, rep_b: str,
         model=SELECT_MODEL) -> dict:
    """One pairwise comparison -> per-dimension verdicts {dim: 'A'|'B'}.
    Dimension weights are deliberately NOT shown to the model — aggregation happens in Python."""
    user = (
        f"<task>\n{task_prompt}\n</task>\n\n"
        f"<criteria>\n{_criteria_block(crit)}\n</criteria>\n\n"
        f"<report_A>\n{rep_a}\n</report_A>\n\n<report_B>\n{rep_b}\n</report_B>\n\n"
        "Compare the reports on each of the four dimensions independently and output the "
        "strict JSON verdict now."
    )
    raw = llm.chat(model, DUEL_SYS, user, temperature=0.0, max_tokens=400)
    out = {}
    for d in DIMS:
        m = re.search(rf'"{d}"\s*:\s*"([AB])"', raw)
        out[d] = m.group(1) if m else "A"          # fail-soft: missing verdict keeps incumbent
    return out


def _weighted_winner(verdicts: dict, weights: dict) -> str:
    a = sum(weights.get(d, 0.25) for d in DIMS if verdicts[d] == "A")
    b = sum(weights.get(d, 0.25) for d in DIMS if verdicts[d] == "B")
    return "A" if a >= b else "B"


def select_report(task_prompt: str, crit: dict, candidates: dict,
                  model=SELECT_MODEL, log=None):
    """King-of-the-hill over {name: report_text}. Per-duel winner = per-dimension verdicts
    aggregated with the task's official dimension weights. Position alternates each duel."""
    weights = crit.get("dimension_weight", {})
    names = list(candidates)
    champ = names[0]
    duels = []
    for i, challenger in enumerate(names[1:]):
        a, b = (champ, challenger) if i % 2 == 0 else (challenger, champ)
        v = duel(task_prompt, crit, candidates[a], candidates[b], model)
        winner = a if _weighted_winner(v, weights) == "A" else b
        duels.append((champ, challenger, winner, v))
        if log:
            vs = " ".join(f"{d[:4]}:{a if v[d]=='A' else b}" for d in DIMS)
            log(f"    duel {champ} vs {challenger} -> {winner}   [{vs}]")
        champ = winner
    return champ, duels
