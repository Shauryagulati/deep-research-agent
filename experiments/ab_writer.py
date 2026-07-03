"""A/B for the judge-tuned writer — hold plan + evidence CONSTANT; only the writer prompt differs.

Per task: plan + retrieval + ledger + deliverables ONCE (cached). Then write TWO arms on that
SAME plan + evidence, coverage contract ON in both:
  Arm A (vanilla): current writer rules
  Arm B (tuned):   judge-tuned writer rules (write.WRITER_RULES_TUNED)
Resumable. Score on gpt-5.5 via OpenRouter (ab_writer_score.py).

  DR_PLAN_MODEL=opus DR_COVERAGE_MODEL=opus DR_LEDGER_MODEL=gpt-4o-mini python3 experiments/ab_writer.py
"""
import _bootstrap  # noqa: F401  (repo root on sys.path)
import os
import pickle

from dr import plan as plan_mod
from dr import retrieve
from dr import ledger as ledger_mod
from dr import write
from dr import bench

IDS = [51, 61, 71, 81, 91]
CACHE = "out/w_cache"
os.makedirs(CACHE, exist_ok=True)
MAX_SOURCES = 16  # depth config
tasks = bench.load_tasks()


def prep(t):
    """plan + base retrieval + ledger + deliverables, cached (shared identically by both arms).

    pickle is safe here: cache is produced and consumed by THIS script only (local/trusted)."""
    p = f"{CACHE}/prep_{t}.pkl"
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    prompt = tasks[t]["prompt"]
    pl = plan_mod.make_plan(prompt)
    src = retrieve.gather(pl.subquestions, max_sources=MAX_SOURCES)
    cl = ledger_mod.build_ledger(src, prompt)
    deliv = write.extract_deliverables(prompt)          # Opus extraction (precondition)
    pickle.dump((pl, src, cl, deliv), open(p, "wb"))
    return pl, src, cl, deliv


for t in IDS:
    prompt = tasks[t]["prompt"]
    pl, src, cl, deliv = prep(t)
    if not deliv:
        print(f"[{t}] ERROR: 0 deliverables — skipping", flush=True)
        continue

    fa = f"out/dev_wA-vanilla_{t}.md"
    if not os.path.exists(fa):
        open(fa, "w").write(write.write_report(prompt, pl, src, cl, depth=True,
                                               deliverables=deliv, tuned=False))
    fb = f"out/dev_wB-tuned_{t}.md"
    if not os.path.exists(fb):
        open(fb, "w").write(write.write_report(prompt, pl, src, cl, depth=True,
                                               deliverables=deliv, tuned=True))

    aw = len(open(fa).read().split())
    bw = len(open(fb).read().split())
    print(f"[{t}] {len(deliv)} deliv, {len(src)} src, {len(cl)} claims; "
          f"vanilla {aw}w  tuned {bw}w", flush=True)

print("DONE", flush=True)
