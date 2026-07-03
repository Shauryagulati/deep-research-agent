"""Judge-tuned writer v2 (PRUNED) — write the pruned arm only, reusing the SAME cached plan +
evidence as v1 (out/w_cache/prep_*.pkl). Do NOT regenerate plan/retrieval/ledger/deliverables.
Base arm = vanilla gpt-5.5 scores already on disk (not re-generated, not re-scored). Resumable.

  DR_COVERAGE_MODEL=opus python3 experiments/ab_writer_prune.py
"""
import _bootstrap  # noqa: F401  (repo root on sys.path)
import os
import pickle

from dr import write
from dr import bench

IDS = [51, 61, 71, 81, 91]
CACHE = "out/w_cache"
tasks = bench.load_tasks()

for t in IDS:
    prompt = tasks[t]["prompt"]
    # pickle is safe: cache produced/consumed by our own scripts only (local/trusted).
    pl, src, cl, deliv = pickle.load(open(f"{CACHE}/prep_{t}.pkl", "rb"))

    fc = f"out/dev_wC-pruned_{t}.md"
    if not os.path.exists(fc) or os.path.getsize(fc) == 0:
        txt = write.write_report(prompt, pl, src, cl, depth=True,
                                 deliverables=deliv, tuned=True)  # v2 pruned rules
        open(fc, "w").write(txt)  # write only after generation succeeds (no 0-byte resume traps)

    cw = len(open(fc).read().split())
    vw = len(open(f"out/dev_wA-vanilla_{t}.md").read().split())
    print(f"[{t}] vanilla {vw}w  pruned {cw}w  ({(cw - vw) / vw * 100:+.0f}%)", flush=True)

print("DONE", flush=True)
