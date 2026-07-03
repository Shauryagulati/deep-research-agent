"""The deep-research engine: plan -> retrieve -> ledger -> write -> judge-finish.

An original pipeline (clean-room). Two stages are feature-flagged so every paper claim is an
ablation we actually ran:
  - stage 3 (ledger):       enrichment of verified, sourced claims before writing
  - stage 5 (judge-finish): whole-report critic that patches the single weakest dimension

Runs end-to-end in mock mode (no keys, no spend) for plumbing tests.
"""
import os

from . import plan as plan_mod
from . import retrieve as retrieve_mod
from . import ledger as ledger_mod
from . import write as write_mod
from . import judge
from . import loop


def research(task_prompt, *, use_ledger=True, use_judge_finish=True, max_sources=8,
             depth=False, deepen=False, coverage=None, retrieval_depth=None, tuned=None):
    """Returns (report, trace). trace carries the per-stage counts + finish history.

    coverage gates the deliverable-anchored coverage contract (extract the task's required
    deliverables, then enforce per-deliverable coverage in the writer). retrieval_depth gates the
    P3 IR pipeline (RRF + cross-encoder rerank + condense). Both default to their env flags
    (DR_COVERAGE_CONTRACT / DR_RETRIEVAL_DEPTH), off unless set."""
    if coverage is None:
        coverage = os.environ.get("DR_COVERAGE_CONTRACT", "").lower() in ("1", "true", "yes", "on")
    if retrieval_depth is None:
        retrieval_depth = os.environ.get("DR_RETRIEVAL_DEPTH", "").lower() in ("1", "true", "yes", "on")
    if tuned is None:
        tuned = os.environ.get("DR_WRITER_RULES", "").lower() in ("1", "true", "yes", "on")
    plan = plan_mod.make_plan(task_prompt)
    if retrieval_depth:
        from . import retrieve_depth as rd
        sources = rd.depth_pipeline(plan.subquestions, task_prompt, top_k=max_sources)
    else:
        sources = retrieve_mod.gather(plan.subquestions, max_sources=max_sources)
    claims = ledger_mod.build_ledger(sources, task_prompt) if use_ledger else None
    deliverables = write_mod.extract_deliverables(task_prompt) if coverage else None
    report = write_mod.write_report(task_prompt, plan, sources, claims, depth=depth,
                                    deepen=deepen, deliverables=deliverables, tuned=tuned)

    trace = {
        "subquestions": plan.subquestions,
        "outline": plan.outline,
        "n_sources": len(sources),
        "n_claims": len(claims) if claims else 0,
        "n_deliverables": len(deliverables) if deliverables else 0,
    }

    if use_judge_finish:
        criteria = judge.regenerate_criteria(task_prompt)
        report, score, history = loop.judge_loop(report, task_prompt, criteria)
        trace["finish_history"] = [(label, round(s.overall, 4)) for label, s in history]

    return report, trace
