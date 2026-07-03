"""Closed-loop whole-report critic + targeted, JUDGE-GATED patch.

Score the assembled report, find the single binding (weakest) dimension, patch ONLY that,
re-score, and keep the patch ONLY if it measurably improved (judge-gated). The gate is the
safeguard against blind self-refinement, which tends to regress about as often as it helps —
we never keep a change the judge did not reward.
"""
import os

from . import judge
from . import llm
from . import proxy

PATCH_MODEL = os.environ.get("DR_PATCH_MODEL", "opus")  # Opus via Claude Max — the repair must be strong, $0


def patch_dimension(report: str, task_prompt: str, criteria: str, dim: str,
                    model: str = PATCH_MODEL) -> str:
    system = (
        f"You are a precise editor. Improve ONLY the {dim.upper()} of the research report below, "
        f"guided by the criteria. Hard constraints: preserve every fact, number, and named-source "
        f"attribution; keep the section structure and all headings; do not add hedging; do not "
        f"shorten substance. Output the full revised report as raw GitHub-flavored markdown ONLY "
        f"— no preamble, no commentary, no surrounding code fences; begin directly with the "
        f"first heading (#)."
    )

    def _mock(system_, user_):
        # Deterministic stand-in: for readability, genuinely break up walls of text.
        # Other dimensions are no-ops in mock mode (a real model does the work with a key).
        if dim == "readability":
            return proxy.split_long_paragraphs(report)
        return report

    user = (f"<task>\n{task_prompt}\n</task>\n\n"
            f"<criteria>\n{criteria}\n</criteria>\n\n"
            f"<report>\n{report}\n</report>")
    out = llm.chat(model, system, user, mock_fn=_mock, temperature=0.4, max_tokens=16000)
    return proxy.clean_report_output(out)


def judge_loop(report: str, task_prompt: str, criteria: str,
               max_patches: int = 2, eps: float = 0.002):
    """Returns (best_report, best_score, history). history is a list of (label, Score)."""
    cur = report
    score = judge.score_report(cur, task_prompt, criteria)
    history = [("initial", score)]

    for _ in range(max_patches):
        dim = score.weakest()
        patched = patch_dimension(cur, task_prompt, criteria, dim)
        new = judge.score_report(patched, task_prompt, criteria)
        if new.overall > score.overall + eps:          # JUDGE-GATED: keep measured wins only
            cur, score = patched, new
            history.append((f"accept:{dim}", new))
        else:
            history.append((f"reject:{dim} (delta {new.overall - score.overall:+.4f})", new))
            break

    return cur, score, history
