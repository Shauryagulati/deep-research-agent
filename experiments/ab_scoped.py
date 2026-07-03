"""Scoped-expansion A/B — write the scope-guarded fan-out arm (wG-scoped) from the SAME cached
plan + evidence as all w-arms. Only the scope guard differs from wE-expand, so G − E isolates
it. Resumable; generate first, write after; 0-byte files regenerate.

  python3 experiments/ab_scoped.py             # all 5 dev tasks
  DR_IDS=91 python3 experiments/ab_scoped.py   # subset
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

    fg = f"out/dev_wG-scoped_{t}.md"
    if not os.path.exists(fg) or os.path.getsize(fg) == 0:
        txt = write_expand.expand_report(prompt, pl, src, cl, deliverables=deliv, scoped=True)
        open(fg, "w").write(txt)

    gw = len(open(fg).read().split())
    ew = len(open(f"out/dev_wE-expand_{t}.md").read().split())
    print(f"[{t}] expand {ew}w  scoped {gw}w  ({(gw - ew) / ew * 100:+.0f}%)", flush=True)

print("DONE", flush=True)
