"""A/B for retrieval depth (P3) — SAME Tavily pull for both arms; coverage contract ON in both.

Per task: plan -> ONE multi-sub-question Tavily pull (cached, uncapped) -> deliverables (cached).
  Arm A (base): base_process (dedup + cap)                      -> ledger -> writer
  Arm B (P3):   RRF fuse + cross-encoder rerank + condense      -> ledger -> writer
Only the IR processing differs (sufficiency gap-pass OFF for a clean read). Resumable. Records
top-K source overlap A-vs-B. Score on GPT-5.2 separately (ab_p3_score.py).

  DR_PLAN_MODEL=gpt-4o-mini DR_LEDGER_MODEL=gpt-4o-mini DR_COVERAGE_MODEL=gpt-4o-mini \
  DR_CONDENSE_MODEL=gpt-4o-mini python3 experiments/ab_retrieval.py
"""
import _bootstrap  # noqa: F401  (repo root on sys.path)
import os
import json
import pickle

from dr import plan as plan_mod
from dr import retrieve_depth as rd
from dr import ledger as ledger_mod
from dr import write
from dr import bench

IDS = [51, 61, 71, 81, 91]
CACHE = "out/p3_cache"
os.makedirs(CACHE, exist_ok=True)
TOPK = 16
tasks = bench.load_tasks()


def prep(t):
    """plan + ONE Tavily pull + deliverables, cached (shared identically by both arms).

    pickle is safe here: cache is produced and consumed by THIS script only (local/trusted)."""
    p = f"{CACHE}/prep_{t}.pkl"
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    prompt = tasks[t]["prompt"]
    pl = plan_mod.make_plan(prompt)
    raw = rd.raw_pull(pl.subquestions)            # uncapped per-sub-question lists (shared pull)
    deliv = write.extract_deliverables(prompt)
    pickle.dump((pl, raw, deliv), open(p, "wb"))
    return pl, raw, deliv


overlap = {}
for t in IDS:
    prompt = tasks[t]["prompt"]
    pl, raw, deliv = prep(t)
    if not deliv:
        print(f"[{t}] ERROR: 0 deliverables — skipping", flush=True)
        continue

    srcA = rd.base_process(raw, TOPK)                              # Arm A selection
    keptB = rd.rerank(rd.rrf_fuse(raw), pl.subquestions, TOPK)     # Arm B selection (pre-condense)

    fa = f"out/dev_p3A-base_{t}.md"
    if not os.path.exists(fa):
        claimsA = ledger_mod.build_ledger(srcA, prompt)
        open(fa, "w").write(write.write_report(prompt, pl, srcA, claimsA,
                                               depth=True, deliverables=deliv))

    fb = f"out/dev_p3B-depth_{t}.md"
    if not os.path.exists(fb):
        condB = rd.condense(keptB, prompt)
        claimsB = ledger_mod.build_ledger(condB, prompt)
        open(fb, "w").write(write.write_report(prompt, pl, condB, claimsB,
                                               depth=True, deliverables=deliv))

    ua = {s.url for s in srcA}
    ub = {s.url for s in keptB}
    inter, uni = len(ua & ub), len(ua | ub)
    overlap[t] = {"nA": len(ua), "nB": len(ub), "shared": inter,
                  "changed": len(ub - ua), "jaccard": round(inter / uni, 3) if uni else 0}
    print(f"[{t}] {len(deliv)} deliv; A={len(ua)} B={len(ub)} src; "
          f"shared={inter}  B-new={len(ub - ua)}  jaccard={overlap[t]['jaccard']}", flush=True)

json.dump(overlap, open(f"{CACHE}/overlap.json", "w"), indent=2)
print("DONE", flush=True)
