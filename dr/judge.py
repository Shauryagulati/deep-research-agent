"""Criteria regeneration + whole-report self-RACE scorer.

Scores the WHOLE assembled report against the regenerated RACE criteria — the part a
per-section-only scorer can't see. In mock mode the readability score is derived from the
deterministic proxy so the loop runs end-to-end with no key.

TODO: align CRITERIA_SYS / RACE_SYS with the official benchmark prompts, and add the official
scorer as the independent referee (see run_v1.py).
"""
from dataclasses import dataclass
import os
import json

from . import llm

DIMS = ["comprehensiveness", "insight", "instruction_following", "readability"]
JUDGE_MODEL = os.environ.get("DR_JUDGE_MODEL", "openai/gpt-5.5")

CRITERIA_SYS = (
    "You regenerate the task-specific evaluation rubric for a deep-research report, following "
    "the RACE procedure. For the user's task, output a JSON object mapping each of these "
    "dimensions to a list of 3-6 concrete, checkable criteria: comprehensiveness, insight, "
    "instruction_following, readability. Criteria must be specific to THIS task. Output JSON only."
)

RACE_SYS = (
    "You are the RACE judge. Score the report against the task and the given criteria on four "
    "dimensions, each 0.0-1.0: comprehensiveness, insight, instruction_following, readability. Be "
    "discriminative and calibrated; reports cluster near 0.5. Citations/markers are ignored for "
    "RACE — judge the prose, structure, depth, and instruction adherence. Output ONLY a JSON "
    'object: {"comprehensiveness": x, "insight": x, "instruction_following": x, "readability": x}.'
)


@dataclass
class Score:
    dims: dict
    overall: float
    raw: str = ""

    def weakest(self) -> str:
        return min(self.dims, key=self.dims.get)


def _parse_dims(text: str) -> dict:
    obj = {}
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except Exception:
        obj = {}
    out = {}
    for d in DIMS:
        try:
            out[d] = float(obj.get(d))
        except (TypeError, ValueError):
            out[d] = 0.5
    return out


def regenerate_criteria(task_prompt: str, model: str = JUDGE_MODEL) -> str:
    def _mock(system, user):
        return json.dumps({d: [f"task-specific {d} criterion"] for d in DIMS})
    return llm.chat(model, CRITERIA_SYS, task_prompt, mock_fn=_mock,
                    temperature=0.0, max_tokens=1200)


def score_report(report: str, task_prompt: str, criteria: str, model: str = JUDGE_MODEL) -> Score:
    def _mock(system, user):
        from . import proxy
        rp = proxy.readability_proxy(report)
        dims = {
            "comprehensiveness": 0.541,
            "insight": 0.553,
            "instruction_following": 0.539,
            "readability": round(0.46 + 0.09 * rp, 4),  # tracks the deterministic proxy
        }
        return json.dumps(dims)

    user = (f"<task>\n{task_prompt}\n</task>\n\n"
            f"<criteria>\n{criteria}\n</criteria>\n\n"
            f"<report>\n{report}\n</report>")
    raw = llm.chat(model, RACE_SYS, user, mock_fn=_mock, temperature=0.0, max_tokens=600)
    dims = _parse_dims(raw)
    # NOTE: the official RACE 'overall' is a dynamic, reference-relative weighting we do not
    # reproduce here. For the loop's internal accept/reject we use the simple mean as a monotone
    # proxy for "did the whole report get better". The official harness is the referee.
    overall = round(sum(dims.values()) / len(dims), 4)
    return Score(dims=dims, overall=overall, raw=raw)
