"""Coverage-expansion A/B — write the section-fanout arm (wE-expand), reusing the SAME cached
plan + evidence + deliverables as the writer A/Bs (out/w_cache/prep_*.pkl). Nothing upstream is
regenerated, so E − vanilla isolates the writer structure/length change. Resumable; a crashed
run leaves no partial file (generate first, write after; 0-byte files are regenerated).

  python3 experiments/ab_expand.py            # all 5 dev tasks
  DR_IDS=51 python3 experiments/ab_expand.py  # subset
"""
import _bootstrap  # noqa: F401  (repo root on sys.path)
import os
import pickle

from dr import write_expand
from dr import bench

IDS = [int(x) for x in os.environ.get("DR_IDS", "51,61,71,81,91").split(",")]
CACHE = "out/w_cache"
tasks = bench.load_tasks()

for t in IDS:
    prompt = tasks[t]["prompt"]
    # pickle is safe: cache produced/consumed by our own scripts only (local/trusted).
    pl, src, cl, deliv = pickle.load(open(f"{CACHE}/prep_{t}.pkl", "rb"))

    fe = f"out/dev_wE-expand_{t}.md"
    if not os.path.exists(fe) or os.path.getsize(fe) == 0:
        txt = write_expand.expand_report(prompt, pl, src, cl, deliverables=deliv)
        open(fe, "w").write(txt)

    ew = len(open(fe).read().split())
    vw = len(open(f"out/dev_wA-vanilla_{t}.md").read().split())
    print(f"[{t}] vanilla {vw}w  expand {ew}w  ({(ew - vw) / vw * 100:+.0f}%)", flush=True)

print("DONE", flush=True)
