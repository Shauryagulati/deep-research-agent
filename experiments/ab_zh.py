"""ZH probe — first measurement on the Chinese half of the benchmark (50 of 100 tasks, never
measured). Runs the FULL banked pipeline (plan -> Tavily retrieve -> per-source ledger ->
coverage contract -> depth writer) on a small spread of ZH tasks and scores one official RACE
pass (zh cleaning + gpt-5.5 judge). Output: per-dimension deltas vs the reference on the same
tasks — the answer to "are we bleeding on ZH or is it a windfall?".

  python3 experiments/ab_zh.py                 # default probe ids
  DR_IDS=1,25,45 python3 experiments/ab_zh.py  # custom
"""
import _bootstrap  # noqa: F401  (repo root on sys.path)
import os
import sys
import statistics

from dr import bench
from dr import data
from dr import engine
from dr import llm

DRB = os.path.abspath(os.path.join(os.path.dirname(bench.__file__), "..", "..", "deep_research_bench"))
sys.path.insert(0, DRB)
from utils.api import AIClient
from utils.clean_article import ArticleCleaner

IDS = [int(x) for x in os.environ.get("DR_IDS", "11,21,31").split(",")]
DIMS = ["comprehensiveness", "insight", "instruction_following", "readability", "overall_score"]
RAW = os.path.join(DRB, "data/test_data/raw_data")
CLEAN = os.path.join(DRB, "data/test_data/cleaned_data")
JUDGE = os.environ.get("DR_JUDGE", "openai/gpt-5.5")
JUDGE_BACKEND = os.environ.get("DR_JUDGE_BACKEND", "openrouter")
CLEANER = os.environ.get("DR_CLEANER", "gpt-5-nano")
NAME = "zh-probe"

tasks = bench.load_tasks()
reports = {}
for t in IDS:
    assert tasks[t]["language"] == "zh", f"task {t} is not zh"
    fz = f"out/dev_{NAME}_{t}.md"
    if not (os.path.exists(fz) and os.path.getsize(fz) > 0):
        print(f"[{t}] generating (zh)...", flush=True)
        report, trace = engine.research(tasks[t]["prompt"], use_judge_finish=False,
                                        max_sources=16, depth=True, coverage=True)
        open(fz, "w").write(report)
        print(f"[{t}] {len(report)} chars, {trace['n_sources']} src, "
              f"{trace['n_claims']} claims", flush=True)
    reports[t] = open(fz).read()

bench.write_submission(NAME, reports, tasks)
# backend switch AFTER generation: the pipeline's DeepSeek stages need the default (OpenRouter)
# backend; only the cleaner AIClient below wants OpenAI-direct. Scar: setting this at import
# time routed deepseek/deepseek-chat to the OpenAI API -> 400 invalid model.
os.environ["LLM_BACKEND"] = "openai"
print(f"cleaning {NAME} with {CLEANER} (zh)...", flush=True)
ArticleCleaner(AIClient(model=CLEANER)).clean_articles(NAME, RAW, CLEAN, max_workers=5, language="zh")
print(f"scoring {NAME} on {JUDGE} via {JUDGE_BACKEND}...", flush=True)
rc, out = bench.run_race(NAME, only_zh=True, force=True, judge_model=JUDGE, backend=JUDGE_BACKEND)
print(f"  {NAME}: rc={rc} scored={sorted(bench.read_scores(NAME))}", flush=True)
scores = bench.read_scores(NAME)
if not scores:
    print(out[-1800:]); sys.exit(1)

ref = data.load_scores()
print(f"\n=== ZH PROBE vs reference (gpt-5.5) ===")
print(f"{'task':>5} {'ours':>8} {'ref':>8} {'delta':>8}")
for t in IDS:
    s, h = scores.get(t), ref.get(t, {})
    if not s:
        print(f"{t:>5}  (no score)"); continue
    ho = h.get("overall_score", float("nan"))
    print(f"{t:>5} {s['overall_score']:>8.4f} {ho:>8.4f} {s['overall_score'] - ho:>+8.4f}")
print("\nper-dimension average:")
for d in DIMS:
    om = statistics.mean(scores[t][d] for t in IDS if t in scores)
    hm = statistics.mean(ref[t][d] for t in IDS if t in ref and ref[t].get(d) is not None)
    print(f"  {d:22} ours {om:.4f}   ref {hm:.4f}   delta {om - hm:+.4f}")
print("\nEN dev-set context: vanilla stack averages ~0.457-0.49 vs ref ~0.5405 (delta ~-0.05..-0.08).")
