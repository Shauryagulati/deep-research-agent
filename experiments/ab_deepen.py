"""Within-task A/B for the insight-deepening pass (one post-draft Opus call).

Isolation: deepen the SAME base draft, so the ONLY variable is the deepen pass. Reads the
5 base drafts produced by `run_devset.py --depth --name ab-base`, deepens each, scores BOTH
arms in one official RACE pass (GPT-5.5), and prints per-task + 5-task-average deltas
(deepen - base). Flags readability if it drops.

  DR_USE_JUDGE_FINISH=0 python3 run_devset.py --depth --name ab-base   # writes base drafts
  python3 experiments/ab_deepen.py                                                 # deepen + score both
"""
import _bootstrap  # noqa: F401  (repo root on sys.path)
import statistics

from dr import bench
from dr.write import _deepen_insight

IDS = [51, 61, 71, 81, 91]
DIMS = ["comprehensiveness", "insight", "instruction_following", "readability", "overall_score"]
LABELS = {"comprehensiveness": "compr", "insight": "insight",
          "instruction_following": "instr", "readability": "read", "overall_score": "overall"}

tasks = bench.load_tasks()
base = {tid: open(f"out/dev_ab-base_{tid}.md").read() for tid in IDS}

print("deepening 5 base drafts (Opus via Max, $0)...", flush=True)
deep = {}
for tid in IDS:
    d = _deepen_insight(base[tid], tasks[tid]["prompt"])
    deep[tid] = d
    with open(f"out/dev_ab-deepen_{tid}.md", "w") as f:
        f.write(d)
    bw, dw = len(base[tid].split()), len(d.split())
    flag = "  (UNCHANGED — fail-soft)" if d == base[tid] else ""
    print(f"[{tid}] base {bw}w -> deepen {dw}w{flag}", flush=True)

bench.write_submission("ab-base", base, tasks)
bench.write_submission("ab-deepen", deep, tasks)

print("\nscoring both arms via official RACE (GPT-5.5)...", flush=True)
for name in ("ab-base", "ab-deepen"):
    rc, out = bench.run_race(name, only_en=True, force=True)
    print(f"  {name}: rc={rc}", flush=True)
    if rc != 0:
        print(out[-1500:])

bs = bench.read_scores("ab-base")
ds = bench.read_scores("ab-deepen")

print("\n=== per-task delta (deepen - base) ===")
print(f"{'task':>5}  " + "  ".join(f"{LABELS[d]:>8}" for d in DIMS))
acc = {d: [] for d in DIMS}
for tid in IDS:
    b, d = bs.get(tid), ds.get(tid)
    if not b or not d:
        print(f"{tid:>5}  (missing score: base={bool(b)} deepen={bool(d)})")
        continue
    row = []
    for dim in DIMS:
        delta = d[dim] - b[dim]
        acc[dim].append(delta)
        row.append(f"{delta:>+8.4f}")
    print(f"{tid:>5}  " + "  ".join(row))

print("\n=== 5-task AVERAGE delta (deepen - base) ===")
for dim in DIMS:
    m = statistics.mean(acc[dim]) if acc[dim] else float("nan")
    note = "  <-- READABILITY DROP" if dim == "readability" and m < 0 else ""
    print(f"  {dim:22} {m:>+8.4f}{note}")
