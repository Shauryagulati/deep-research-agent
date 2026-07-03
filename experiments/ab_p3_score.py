"""Score the retrieval-depth (P3) A/B on GPT-5.2 (full flagship — a cheap model is NOT a valid
judge; calibration proved it). Cleaning uses the cheap tier (gpt-4o-mini) so the flagship only
does the scoring pass. Reports per-task + 5-task-average deltas, the keep-rule check, and the
top-K source-overlap note (did rerank meaningfully change the selected sources vs base).
"""
import _bootstrap  # noqa: F401  (repo root on sys.path)
import os
import sys
import json
import statistics

from dr import bench
from dr import llm
os.environ.setdefault("LLM_BACKEND", "openai")

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
ARMS = {"p3A-base": "out/dev_p3A-base_{}.md", "p3B-depth": "out/dev_p3B-depth_{}.md"}
JUDGE = os.environ.get("DR_JUDGE", "gpt-5.2")          # full flagship (gpt-5.5 not on OpenAI direct)
CLEANER = os.environ.get("DR_CLEANER", "gpt-4o-mini")  # cheap clean; flagship only scores
tasks = bench.load_tasks()

for name, tmpl in ARMS.items():
    bench.write_submission(name, {t: open(tmpl.format(t)).read() for t in IDS}, tasks)

print(f"cleaning both arms with {CLEANER}...", flush=True)
cleaner = ArticleCleaner(AIClient(model=CLEANER))
for name in ARMS:
    cleaner.clean_articles(name, RAW, CLEAN, max_workers=5, language="en")

print(f"scoring both arms on {JUDGE}...", flush=True)
for name in ARMS:
    rc, out = bench.run_race(name, only_en=True, force=True, judge_model=JUDGE)
    sc = bench.read_scores(name)
    print(f"  {name}: rc={rc} scored={sorted(sc)}", flush=True)
    if not sc:
        print(out[-1800:])

A = bench.read_scores("p3A-base")
B = bench.read_scores("p3B-depth")

print("\n=== RETRIEVAL DEPTH (P3) A/B (GPT-5.2): per-task delta (P3 - base) ===")
print(f"{'task':>5}  " + "  ".join(f"{LAB[d]:>8}" for d in DIMS))
acc = {d: [] for d in DIMS}
for t in IDS:
    row = []
    for d in DIMS:
        x = B[t][d] - A[t][d]
        acc[d].append(x)
        row.append(f"{x:>+8.4f}")
    print(f"{t:>5}  " + "  ".join(row))

print("\n=== 5-task AVERAGE delta (P3 - base) ===")
avg = {}
for d in DIMS:
    avg[d] = statistics.mean(acc[d])
    flag = "  <-- TARGET" if d == "comprehensiveness" else ""
    if d == "readability" and avg[d] < 0:
        flag = "  <-- READABILITY DROP"
    print(f"  {d:22} {avg[d]:>+8.4f}{flag}")

comp = avg["comprehensiveness"]; ov = avg["overall_score"]
worst_instr = min(B[t]["instruction_following"] - A[t]["instruction_following"] for t in IDS)
worst_read = min(B[t]["readability"] - A[t]["readability"] for t in IDS)
print("\n=== keep-rule check (informational; user decides) ===")
print(f"  comprehensiveness lifts : {comp:+.4f} -> {'PASS' if comp > 0 else 'FAIL'}")
print(f"  overall >= +0.01        : {ov:+.4f} -> {'PASS' if ov >= 0.01 else 'FAIL'}")
print(f"  no task instr  < -0.02  : worst={worst_instr:+.4f} -> {'PASS' if worst_instr >= -0.02 else 'FAIL'}")
print(f"  no task read   < -0.02  : worst={worst_read:+.4f} -> {'PASS' if worst_read >= -0.02 else 'FAIL'}")

ov_path = "out/p3_cache/overlap.json"
if os.path.exists(ov_path):
    ovl = json.load(open(ov_path))
    print("\n=== rerank vs base: top-K source overlap ===")
    print(f"{'task':>5} {'#A':>4} {'#B':>4} {'shared':>7} {'B-new':>6} {'jaccard':>8}")
    for t in IDS:
        o = ovl.get(str(t)) or ovl.get(t)
        if o:
            print(f"{t:>5} {o['nA']:>4} {o['nB']:>4} {o['shared']:>7} {o['changed']:>6} {o['jaccard']:>8}")

print("\nabs means:")
for nm, sc in (("base", A), ("P3", B)):
    print(f"  {nm:5}: " + " ".join(f"{LAB[d]}={statistics.mean(sc[t][d] for t in IDS):.3f}" for d in DIMS))
