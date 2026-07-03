"""Score ONLY the pruned arm (wC-pruned) on gpt-5.5 via OpenRouter. Base = vanilla scores already
on disk from v1 (wA-vanilla) — NOT re-scored. Reports per-task + avg deltas (pruned − vanilla),
the keep-rule check, and the v1→v2 recovery checks (71/91 readability, 61 compression, insight).
"""
import _bootstrap  # noqa: F401  (repo root on sys.path)
import os
import sys
import statistics

from dr import bench
from dr import llm
os.environ.setdefault("LLM_BACKEND", "openai")   # for the cheap CLEANER AIClient (gpt-5-nano)

DRB = os.path.abspath(os.path.join(os.path.dirname(bench.__file__), "..", "..", "deep_research_bench"))
sys.path.insert(0, DRB)
from utils.api import AIClient
from utils.clean_article import ArticleCleaner

IDS = [51, 61, 71, 81, 91]
DIMS = ["comprehensiveness", "insight", "instruction_following", "readability", "overall_score"]
LAB = {"comprehensiveness": "compr", "insight": "insight", "instruction_following": "instr",
       "readability": "read", "overall_score": "overall"}
RAW = os.path.join(DRB, "data/test_data/raw_data")
CLEAN = os.path.join(DRB, "data/test_data/cleaned_data")
JUDGE = os.environ.get("DR_JUDGE", "openai/gpt-5.5")
JUDGE_BACKEND = os.environ.get("DR_JUDGE_BACKEND", "openrouter")
CLEANER = os.environ.get("DR_CLEANER", "gpt-5-nano")
tasks = bench.load_tasks()

# score ONLY the pruned arm (vanilla already scored on disk from v1)
bench.write_submission("wC-pruned", {t: open(f"out/dev_wC-pruned_{t}.md").read() for t in IDS}, tasks)
print(f"cleaning wC-pruned with {CLEANER}...", flush=True)
ArticleCleaner(AIClient(model=CLEANER)).clean_articles("wC-pruned", RAW, CLEAN, max_workers=5, language="en")
print(f"scoring wC-pruned on {JUDGE} via {JUDGE_BACKEND}...", flush=True)
rc, out = bench.run_race("wC-pruned", only_en=True, force=True, judge_model=JUDGE, backend=JUDGE_BACKEND)
print(f"  wC-pruned: rc={rc} scored={sorted(bench.read_scores('wC-pruned'))}", flush=True)
if not bench.read_scores("wC-pruned"):
    print(out[-1800:]); sys.exit(1)

V = bench.read_scores("wA-vanilla")    # base (on disk)
T1 = bench.read_scores("wB-tuned")     # v1 tuned (on disk)
P = bench.read_scores("wC-pruned")     # v2 pruned (new)

print("\n=== PRUNED writer v2 A/B (gpt-5.5): per-task delta (pruned - vanilla) ===")
print(f"{'task':>5}  " + "  ".join(f"{LAB[d]:>8}" for d in DIMS))
acc = {d: [] for d in DIMS}
for t in IDS:
    row = []
    for d in DIMS:
        x = P[t][d] - V[t][d]
        acc[d].append(x)
        row.append(f"{x:>+8.4f}")
    print(f"{t:>5}  " + "  ".join(row))

print("\n=== 5-task AVERAGE delta (pruned - vanilla) ===")
avg = {}
for d in DIMS:
    avg[d] = statistics.mean(acc[d])
    flag = "  <-- TARGET" if d in ("comprehensiveness", "insight") else ""
    if d == "readability" and avg[d] < 0:
        flag = "  <-- READABILITY DROP"
    print(f"  {d:22} {avg[d]:>+8.4f}{flag}")

comp, ins, ov = avg["comprehensiveness"], avg["insight"], avg["overall_score"]
worst_read = min(P[t]["readability"] - V[t]["readability"] for t in IDS)
print("\n=== keep-rule check (informational; user decides) ===")
print(f"  comprehensiveness lifts : {comp:+.4f} -> {'PASS' if comp > 0 else 'FAIL'}")
print(f"  insight lifts           : {ins:+.4f} -> {'PASS' if ins > 0 else 'FAIL'}")
print(f"  overall >= +0.01        : {ov:+.4f} -> {'PASS' if ov >= 0.01 else 'FAIL'}")
print(f"  no task read < -0.02    : worst={worst_read:+.4f} -> {'PASS' if worst_read >= -0.02 else 'FAIL'}")

print("\n=== v1 -> v2 recovery checks (readability delta vs vanilla) ===")
for t in (71, 91):
    print(f"  task {t} readability: v1 {T1[t]['readability']-V[t]['readability']:+.4f}  "
          f"-> v2 {P[t]['readability']-V[t]['readability']:+.4f}")
print(f"  task 61 compression: pruned words vs vanilla "
      f"{len(open('out/dev_wC-pruned_61.md').read().split())} vs "
      f"{len(open('out/dev_wA-vanilla_61.md').read().split())}; "
      f"insight v1 {T1[61]['insight']-V[61]['insight']:+.4f} -> v2 {P[61]['insight']-V[61]['insight']:+.4f}")
print(f"  insight held? v1 avg {statistics.mean(T1[t]['insight']-V[t]['insight'] for t in IDS):+.4f}"
      f"  -> v2 avg {ins:+.4f}")

print("\nabs means:")
for nm, sc in (("vanilla", V), ("v1 tuned", T1), ("v2 pruned", P)):
    print(f"  {nm:10}: " + " ".join(f"{LAB[d]}={statistics.mean(sc[t][d] for t in IDS):.3f}" for d in DIMS))
