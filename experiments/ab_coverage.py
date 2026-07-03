"""A/B harness for the deliverable-anchored COVERAGE CONTRACT — EVIDENCE HELD CONSTANT.

Per dev task: run retrieval + ledger ONCE and cache (plan, sources, claims) to disk. Then write
TWO arms from the SAME cached evidence:
  A (base)     = baseline writer (depth)
  B (contract) = deliverable extraction + contract-enforced writer (depth), same evidence
The ONLY variable is the coverage contract (no Tavily noise between arms).

Resumable: cached evidence + cached deliverables + skip already-written reports — so a Claude
Max session-limit interruption is recoverable by just re-running.

  DR_PLAN_MODEL=deepseek/deepseek-chat python3 experiments/ab_coverage.py     # generate both arms
Scoring is a separate step (ab_cov_score.py) — GPT-5.5 only (DeepSeek failed calibration as a judge).
"""
import _bootstrap  # noqa: F401  (repo root on sys.path)
import os
import json
import pickle

from dr import plan as plan_mod
from dr import retrieve
from dr import ledger as ledger_mod
from dr import write
from dr import bench

IDS = [51, 61, 71, 81, 91]
CACHE = "out/cov_evidence"
os.makedirs(CACHE, exist_ok=True)
MAX_SOURCES = 16  # depth config (matches baseline)
tasks = bench.load_tasks()


def evidence(t):
    """Retrieve + ledger ONCE per task; cache so both arms (and re-runs) share identical evidence.

    pickle is safe here: the cache file is produced and consumed by THIS script only (local,
    trusted, ephemeral) — never loaded from an external/untrusted source. It just round-trips our
    own Plan/Source/Claim dataclasses without hand-rolling JSON (de)serialization."""
    p = f"{CACHE}/ev_{t}.pkl"
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    prompt = tasks[t]["prompt"]
    pl = plan_mod.make_plan(prompt)
    src = retrieve.gather(pl.subquestions, max_sources=MAX_SOURCES)
    cl = ledger_mod.build_ledger(src, prompt)
    pickle.dump((pl, src, cl), open(p, "wb"))
    return pl, src, cl


for t in IDS:
    prompt = tasks[t]["prompt"]
    pl, src, cl = evidence(t)

    fa = f"out/dev_covA-base_{t}.md"
    if not os.path.exists(fa):
        open(fa, "w").write(write.write_report(prompt, pl, src, cl, depth=True))

    fd = f"{CACHE}/deliv_{t}.json"
    if os.path.exists(fd):
        deliv = json.load(open(fd))
    else:
        deliv = write.extract_deliverables(prompt)
        if not deliv:   # guard: never silently write a no-contract (== base) arm; retry on re-run
            print(f"[{t}] ERROR: 0 deliverables extracted — skipping contract arm", flush=True)
            continue
        json.dump(deliv, open(fd, "w"), ensure_ascii=False, indent=2)

    fb = f"out/dev_covB-contract_{t}.md"
    if not os.path.exists(fb):
        open(fb, "w").write(
            write.write_report(prompt, pl, src, cl, depth=True, deliverables=deliv))

    aw = len(open(fa).read().split())
    bw = len(open(fb).read().split())
    print(f"[{t}] {len(src)} src, {len(cl)} claims, {len(deliv)} deliverables; "
          f"base {aw}w  contract {bw}w", flush=True)

print("DONE", flush=True)
