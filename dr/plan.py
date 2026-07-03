"""Stage 1 — planning: decompose the task into research sub-questions + a report outline.

High-leverage, low-token: a better decomposition cascades into the whole report, so this is
the prime candidate to upgrade to a stronger model later (an ablation, not a guess). Default
is a cheap-but-capable model; the write/judge stages are where we spend.
"""
from dataclasses import dataclass, field
import os
import json

from . import llm

PLAN_MODEL = os.environ.get("DR_PLAN_MODEL", "opus")  # Opus via Claude Max — high-leverage, $0

PLAN_SYS = (
    "You are the planner for a deep-research report. Decompose the user's task into 5-8 "
    "specific, non-overlapping research sub-questions that together cover EVERY explicit "
    "deliverable the task asks for, and propose an ordered report outline (section headings). "
    "Bias toward full coverage of the task's requirements. "
    'Output ONLY a JSON object: {"subquestions": [...], "outline": [...]}.'
)


@dataclass
class Plan:
    subquestions: list = field(default_factory=list)
    outline: list = field(default_factory=list)


def _parse(text: str) -> dict:
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except Exception:
        return {}


def make_plan(task_prompt: str, model: str = PLAN_MODEL) -> Plan:
    def _mock(system, user):
        return json.dumps({
            "subquestions": [f"Sub-question {i}: a facet of the task to research"
                             for i in range(1, 6)],
            "outline": ["Introduction", "Key Findings", "Analysis", "Implications",
                        "Conclusion"],
        })

    raw = llm.chat(model, PLAN_SYS, task_prompt, mock_fn=_mock,
                   temperature=0.2, max_tokens=1500)
    obj = _parse(raw)
    subs = [str(s) for s in obj.get("subquestions", []) if str(s).strip()]
    outline = [str(s) for s in obj.get("outline", []) if str(s).strip()]
    return Plan(subquestions=subs, outline=outline)
