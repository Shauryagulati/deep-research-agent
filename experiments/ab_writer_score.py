"""Score the judge-tuned-writer A/B on gpt-5.5 via OpenRouter (precondition: do NOT use gpt-5.2 or
any cheap model as judge). Cleaning runs on gpt-5-nano (OpenAI direct) so the flagship only scores.
Reports per-task + 5-task-average deltas and the keep-rule check.
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
ARMS = {"wA-vanilla": "out/dev_wA-vanilla_{}.md", "wB-tuned": "out/dev_wB-tuned_{}.md"}
JUDGE = os.environ.get("DR_JUDGE", "openai/gpt-5.5")     # gpt-5.5 via OpenRouter (only place it exists)
JUDGE_BACKEND = os.environ.get("DR_JUDGE_BACKEND", "openrouter")
CLEANER = os.environ.get("DR_CLEANER", "gpt-5-nano")    # cheap reasoning tier; harness needs reasoning_effort-capable
tasks = bench.load_tasks()

for name, tmpl in ARMS.items():
    bench.write_submission(name, {t: open(tmpl.format(t)).read() for t in IDS}, tasks)

print(f"cleaning both arms with {CLEANER} (OpenAI)...", flush=True)
cleaner = ArticleCleaner(AIClient(model=CLEANER))
for name in ARMS:
    cleaner.clean_articles(name, RAW, CLEAN, max_workers=5, language="en")

print(f"scoring both arms on {JUDGE} via {JUDGE_BACKEND}...", flush=True)
for name in ARMS:
    rc, out = bench.run_race(name, only_en=True, force=True, judge_model=JUDGE, backend=JUDGE_BACKEND)
    sc = bench.read_scores(name)
    print(f"  {name}: rc={rc} scored={sorted(sc)}", flush=True)
    if not sc:
        print(out[-1800:])

A = bench.read_scores("wA-vanilla")
B = bench.read_scores("wB-tuned")

print("\n=== JUDGE-TUNED WRITER A/B (gpt-5.5): per-task delta (tuned - vanilla) ===")
print(f"{'task':>5}  " + "  ".join(f"{LAB[d]:>8}" for d in DIMS))
acc = {d: [] for d in DIMS}
for t in IDS:
    row = []
    for d in DIMS:
        x = B[t][d] - A[t][d]
        acc[d].append(x)
        row.append(f"{x:>+8.4f}")
    print(f"{t:>5}  " + "  ".join(row))

print("\n=== 5-task AVERAGE delta (tuned - vanilla) ===")
avg = {}
for d in DIMS:
    avg[d] = statistics.mean(acc[d])
    flag = ""
    if d in ("comprehensiveness", "insight"):
        flag = "  <-- TARGET"
    if d == "readability" and avg[d] < 0:
        flag = "  <-- READABILITY DROP"
    print(f"  {d:22} {avg[d]:>+8.4f}{flag}")

comp, ins, ov = avg["comprehensiveness"], avg["insight"], avg["overall_score"]
worst_read = min(B[t]["readability"] - A[t]["readability"] for t in IDS)
print("\n=== keep-rule check (informational; user decides) ===")
print(f"  comprehensiveness lifts : {comp:+.4f} -> {'PASS' if comp > 0 else 'FAIL'}")
print(f"  insight lifts           : {ins:+.4f} -> {'PASS' if ins > 0 else 'FAIL'}")
print(f"  overall >= +0.01        : {ov:+.4f} -> {'PASS' if ov >= 0.01 else 'FAIL'}")
print(f"  no task read < -0.02    : worst={worst_read:+.4f} -> {'PASS' if worst_read >= -0.02 else 'FAIL'}")

print("\nabs means:")
for nm, sc in (("vanilla", A), ("tuned", B)):
    print(f"  {nm:8}: " + " ".join(f"{LAB[d]}={statistics.mean(sc[t][d] for t in IDS):.3f}" for d in DIMS))
