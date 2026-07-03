"""Polish A/B — apply the whole-article readability polish to the on-disk expansion arm
(wE-expand -> wF-polish). Within-task isolation: the ONLY change is the polish pass, so
F − E measures it directly. Resumable; generate first, write after; 0-byte files regenerate.

  python3 experiments/ab_polish.py            # all 5 dev tasks
  DR_IDS=91 python3 experiments/ab_polish.py  # subset
"""
import _bootstrap  # noqa: F401  (repo root on sys.path)
import os

from dr import write_polish
from dr import bench

IDS = [int(x) for x in os.environ.get("DR_IDS", "51,61,71,81,91").split(",")]
tasks = bench.load_tasks()

for t in IDS:
    fp = f"out/dev_wF-polish_{t}.md"
    if not os.path.exists(fp) or os.path.getsize(fp) == 0:
        src = open(f"out/dev_wE-expand_{t}.md").read()
        txt = write_polish.polish_report(src, tasks[t]["prompt"])
        open(fp, "w").write(txt)
        reverted = "  [REVERTED to input]" if txt == src else ""
    else:
        reverted = ""

    pw = len(open(fp).read().split())
    ew = len(open(f"out/dev_wE-expand_{t}.md").read().split())
    print(f"[{t}] expand {ew}w  polish {pw}w  ({(pw - ew) / ew * 100:+.0f}%){reverted}", flush=True)

print("DONE", flush=True)
