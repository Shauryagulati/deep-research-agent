"""Score the coverage-contract A/B on GPT-5.5 (the only valid judge — DeepSeek failed calibration).

To respect the GPT budget, articles are CLEANED with DeepSeek (cleaning is normalization, not
judging) so GPT-5.5 only does the scoring pass. Reports per-task + 5-task-average deltas
(contract - base), the keep-rule check (informational), and deliverable counts + a rough
keyword coverage proxy per arm.
"""
import _bootstrap  # noqa: F401  (repo root on sys.path)
import os
import sys
import re
import json
import statistics

from dr import bench
from dr import llm  # loads agent/.env -> OPENROUTER_API_KEY for the harness modules
os.environ.setdefault("LLM_BACKEND", "openrouter")

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
ARMS = {"covA-base": "out/dev_covA-base_{}.md", "covB-contract": "out/dev_covB-contract_{}.md"}
tasks = bench.load_tasks()


def coverage_proxy(report, deliverables):
    """Rough keyword proxy: a deliverable counts as 'touched' if a majority of its content words
    (>=5 chars) appear in the report. Not semantic — a sanity indicator, not ground truth."""
    rt = report.lower()
    hits = 0
    for d in deliverables:
        words = set(re.findall(r"[a-z]{5,}", d.lower()))
        if words and sum(1 for w in words if w in rt) >= max(1, len(words) // 2):
            hits += 1
    return hits


# 1) submissions
for name, tmpl in ARMS.items():
    bench.write_submission(name, {t: open(tmpl.format(t)).read() for t in IDS}, tasks)

# 2) DeepSeek-clean both arms (cleaning != judging; keeps GPT to the scoring pass)
print("cleaning both arms with DeepSeek...", flush=True)
cleaner = ArticleCleaner(AIClient(model="deepseek/deepseek-chat"))
for name in ARMS:
    cleaner.clean_articles(name, RAW, CLEAN, max_workers=5, language="en")

# 3) GPT-5.5 score both (cleaning skipped — already cleaned)
print("scoring both arms on GPT-5.5...", flush=True)
for name in ARMS:
    rc, out = bench.run_race(name, only_en=True, force=True, judge_model="openai/gpt-5.5")
    sc = bench.read_scores(name)
    print(f"  {name}: rc={rc} scored={sorted(sc)}", flush=True)
    if not sc:
        print(out[-1800:])

A = bench.read_scores("covA-base")
B = bench.read_scores("covB-contract")

print("\n=== COVERAGE CONTRACT A/B (GPT-5.5): per-task delta (contract - base) ===")
print(f"{'task':>5}  " + "  ".join(f"{LAB[d]:>8}" for d in DIMS))
acc = {d: [] for d in DIMS}
for t in IDS:
    row = []
    for d in DIMS:
        x = B[t][d] - A[t][d]
        acc[d].append(x)
        row.append(f"{x:>+8.4f}")
    print(f"{t:>5}  " + "  ".join(row))

print("\n=== 5-task AVERAGE delta (contract - base) ===")
avg = {}
for d in DIMS:
    avg[d] = statistics.mean(acc[d])
    flag = ""
    if d in ("comprehensiveness", "instruction_following"):
        flag = "  <-- TARGET"
    if d == "readability" and avg[d] < 0:
        flag = "  <-- READABILITY DROP"
    print(f"  {d:22} {avg[d]:>+8.4f}{flag}")

# keep-rule (informational; user decides)
comp = avg["comprehensiveness"]; ov = avg["overall_score"]
worst_instr = min(B[t]["instruction_following"] - A[t]["instruction_following"] for t in IDS)
worst_read = min(B[t]["readability"] - A[t]["readability"] for t in IDS)
print("\n=== keep-rule check (informational; user decides) ===")
print(f"  comprehensiveness lifts : {comp:+.4f} -> {'PASS' if comp > 0 else 'FAIL'}")
print(f"  overall >= +0.01        : {ov:+.4f} -> {'PASS' if ov >= 0.01 else 'FAIL'}")
print(f"  no task instr  < -0.02  : worst={worst_instr:+.4f} -> {'PASS' if worst_instr >= -0.02 else 'FAIL'}")
print(f"  no task read   < -0.02  : worst={worst_read:+.4f} -> {'PASS' if worst_read >= -0.02 else 'FAIL'}")

print("\n=== deliverable coverage (count + keyword proxy; proxy is a rough indicator) ===")
print(f"{'task':>5} {'#deliv':>7} {'base touched':>13} {'contract touched':>17}")
for t in IDS:
    deliv = json.load(open(f"out/cov_evidence/deliv_{t}.json"))
    ca = coverage_proxy(open(ARMS['covA-base'].format(t)).read(), deliv)
    cb = coverage_proxy(open(ARMS['covB-contract'].format(t)).read(), deliv)
    print(f"{t:>5} {len(deliv):>7} {ca:>10}/{len(deliv):<2} {cb:>14}/{len(deliv):<2}")

print("\nabs means:")
for nm, sc in (("base", A), ("contract", B)):
    print(f"  {nm:9}: " + " ".join(f"{LAB[d]}={statistics.mean(sc[t][d] for t in IDS):.3f}" for d in DIMS))
