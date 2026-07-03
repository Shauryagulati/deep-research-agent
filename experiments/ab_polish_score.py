"""Score the polish arm (wF-polish) on gpt-5.5 via OpenRouter. Baselines on disk: wE-expand
(F − E isolates the polish pass) and wA-vanilla (total stack effect). Neither is re-scored.
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

bench.write_submission("wF-polish", {t: open(f"out/dev_wF-polish_{t}.md").read() for t in IDS}, tasks)
print(f"cleaning wF-polish with {CLEANER}...", flush=True)
ArticleCleaner(AIClient(model=CLEANER)).clean_articles("wF-polish", RAW, CLEAN, max_workers=5, language="en")
print(f"scoring wF-polish on {JUDGE} via {JUDGE_BACKEND}...", flush=True)
rc, out = bench.run_race("wF-polish", only_en=True, force=True, judge_model=JUDGE, backend=JUDGE_BACKEND)
print(f"  wF-polish: rc={rc} scored={sorted(bench.read_scores('wF-polish'))}", flush=True)
if not bench.read_scores("wF-polish"):
    print(out[-1800:]); sys.exit(1)

V = bench.read_scores("wA-vanilla")   # base (on disk)
E = bench.read_scores("wE-expand")    # pre-polish (on disk)
F = bench.read_scores("wF-polish")    # new

for base, nm in ((E, "expand"), (V, "vanilla")):
    print(f"\n=== POLISH A/B (gpt-5.5): per-task delta (polish - {nm}) ===")
    print(f"{'task':>5}  " + "  ".join(f"{LAB[d]:>8}" for d in DIMS))
    acc = {d: [] for d in DIMS}
    for t in IDS:
        row = []
        for d in DIMS:
            x = F[t][d] - base[t][d]
            acc[d].append(x)
            row.append(f"{x:>+8.4f}")
        print(f"{t:>5}  " + "  ".join(row))
    print(f"  avg   " + "  ".join(f"{statistics.mean(acc[d]):>+8.4f}" for d in DIMS))

avg = {d: statistics.mean(F[t][d] - V[t][d] for t in IDS) for d in DIMS}
comp, ov = avg["comprehensiveness"], avg["overall_score"]
worst_read = min(F[t]["readability"] - V[t]["readability"] for t in IDS)
worst_instr = min(F[t]["instruction_following"] - V[t]["instruction_following"] for t in IDS)
read_rec = statistics.mean(F[t]["readability"] - E[t]["readability"] for t in IDS)
comp_held = statistics.mean(F[t]["comprehensiveness"] - E[t]["comprehensiveness"] for t in IDS)
print("\n=== keep-rule check: FULL STACK (expand+polish) vs vanilla (informational; user decides) ===")
print(f"  comprehensiveness lifts : {comp:+.4f} -> {'PASS' if comp > 0 else 'FAIL'}")
print(f"  overall >= +0.01        : {ov:+.4f} -> {'PASS' if ov >= 0.01 else 'FAIL'}")
print(f"  no task read < -0.02    : worst={worst_read:+.4f} -> {'PASS' if worst_read >= -0.02 else 'FAIL'}")
print(f"  no task instr < -0.02   : worst={worst_instr:+.4f} -> {'PASS' if worst_instr >= -0.02 else 'FAIL'}")
print(f"\n  polish readability recovery (F−E): {read_rec:+.4f} (tax was -0.0231)")
print(f"  polish comp preservation  (F−E): {comp_held:+.4f} (must be ~0 or better)")

print("\nabs means:")
for nm, sc in (("vanilla", V), ("expand", E), ("polish", F)):
    print(f"  {nm:8}: " + " ".join(f"{LAB[d]}={statistics.mean(sc[t][d] for t in IDS):.3f}" for d in DIMS))
