"""Score the scope-guarded expansion arm (wG-scoped) on gpt-5.5 via OpenRouter. On-disk
baselines: wE-expand (G − E isolates the scope guard) and wA-vanilla (full lever). The keep
decision rides on the floors: the guard exists to fix 91's instruction breach (−0.0224) and to
shed off-scope length, without giving back expand's comp/insight.
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

bench.write_submission("wG-scoped", {t: open(f"out/dev_wG-scoped_{t}.md").read() for t in IDS}, tasks)
print(f"cleaning wG-scoped with {CLEANER}...", flush=True)
ArticleCleaner(AIClient(model=CLEANER)).clean_articles("wG-scoped", RAW, CLEAN, max_workers=5, language="en")
print(f"scoring wG-scoped on {JUDGE} via {JUDGE_BACKEND}...", flush=True)
rc, out = bench.run_race("wG-scoped", only_en=True, force=True, judge_model=JUDGE, backend=JUDGE_BACKEND)
print(f"  wG-scoped: rc={rc} scored={sorted(bench.read_scores('wG-scoped'))}", flush=True)
if not bench.read_scores("wG-scoped"):
    print(out[-1800:]); sys.exit(1)

V = bench.read_scores("wA-vanilla")   # base (on disk)
E = bench.read_scores("wE-expand")    # unguarded expansion (on disk)
G = bench.read_scores("wG-scoped")    # new

for base, nm in ((E, "expand"), (V, "vanilla")):
    print(f"\n=== SCOPED A/B (gpt-5.5): per-task delta (scoped - {nm}) ===")
    print(f"{'task':>5}  " + "  ".join(f"{LAB[d]:>8}" for d in DIMS))
    acc = {d: [] for d in DIMS}
    for t in IDS:
        row = []
        for d in DIMS:
            x = G[t][d] - base[t][d]
            acc[d].append(x)
            row.append(f"{x:>+8.4f}")
        print(f"{t:>5}  " + "  ".join(row))
    print(f"  avg   " + "  ".join(f"{statistics.mean(acc[d]):>+8.4f}" for d in DIMS))

avg = {d: statistics.mean(G[t][d] - V[t][d] for t in IDS) for d in DIMS}
comp, ov = avg["comprehensiveness"], avg["overall_score"]
worst_read = min(G[t]["readability"] - V[t]["readability"] for t in IDS)
worst_instr = min(G[t]["instruction_following"] - V[t]["instruction_following"] for t in IDS)
print("\n=== keep-rule check: scoped expansion vs vanilla (informational; user decides) ===")
print(f"  comprehensiveness lifts : {comp:+.4f} -> {'PASS' if comp > 0 else 'FAIL'}")
print(f"  overall >= +0.01        : {ov:+.4f} -> {'PASS' if ov >= 0.01 else 'FAIL'}")
print(f"  no task read < -0.02    : worst={worst_read:+.4f} -> {'PASS' if worst_read >= -0.02 else 'FAIL'}")
print(f"  no task instr < -0.02   : worst={worst_instr:+.4f} -> {'PASS' if worst_instr >= -0.02 else 'FAIL'}")
print(f"\n  91 recovery: instr {G[91]['instruction_following']-V[91]['instruction_following']:+.4f} "
      f"(expand was -0.0224), read {G[91]['readability']-V[91]['readability']:+.4f} (expand was -0.0442)")

print("\nabs means:")
for nm, sc in (("vanilla", V), ("expand", E), ("scoped", G)):
    print(f"  {nm:8}: " + " ".join(f"{LAB[d]}={statistics.mean(sc[t][d] for t in IDS):.3f}" for d in DIMS))
