"""ZH-expansion A/B — does the expansion writer transfer to the Chinese half? The ZH probe
showed the same disease as EN (comp −0.061 / insight −0.069, ~11k-char reports vs ~100k-char
references) and the expansion writer is the measured EN medicine (comp +0.019 / insight +0.025).
Runs the pipeline stages fresh with prep caching (the probe's engine path doesn't cache), writes
the expansion arm (unguarded — the scoped variant was cut), scores one RACE pass, and compares
against the on-disk zh-probe scores (single-call writer, same banked stack).

  DR_LLM_BACKEND=openrouter python3 experiments/ab_zh_expand.py
"""
import _bootstrap  # noqa: F401  (repo root on sys.path)
import os
import pickle
import sys
import statistics

from dr import bench
from dr import plan as plan_mod
from dr import retrieve as retrieve_mod
from dr import ledger as ledger_mod
from dr import write as write_mod
from dr import write_expand

IDS = [int(x) for x in os.environ.get("DR_IDS", "11,21,31").split(",")]
DIMS = ["comprehensiveness", "insight", "instruction_following", "readability", "overall_score"]
LAB = {"comprehensiveness": "compr", "insight": "insight", "instruction_following": "instr",
       "readability": "read", "overall_score": "overall"}
NAME = "zh-expand"
CACHE = "out/zh_cache"
os.makedirs(CACHE, exist_ok=True)
tasks = bench.load_tasks()

reports = {}
for t in IDS:
    assert tasks[t]["language"] == "zh", f"task {t} is not zh"
    prompt = tasks[t]["prompt"]

    pk = f"{CACHE}/prep_{t}.pkl"
    if os.path.exists(pk):
        # pickle is safe: cache produced/consumed by our own scripts only (local/trusted).
        pl, src, cl, deliv = pickle.load(open(pk, "rb"))
    else:
        print(f"[{t}] plan/retrieve/ledger/deliverables (zh)...", flush=True)
        pl = plan_mod.make_plan(prompt)
        src = retrieve_mod.gather(pl.subquestions, max_sources=16)
        cl = ledger_mod.build_ledger(src, prompt)
        deliv = write_mod.extract_deliverables(prompt)
        pickle.dump((pl, src, cl, deliv), open(pk, "wb"))
        print(f"[{t}] {len(src)} src, {len(cl)} claims, {len(deliv)} deliverables", flush=True)

    fz = f"out/dev_{NAME}_{t}.md"
    if not os.path.exists(fz) or os.path.getsize(fz) == 0:
        txt = write_expand.expand_report(prompt, pl, src, cl, deliverables=deliv)
        open(fz, "w").write(txt)
    reports[t] = open(fz).read()
    bw = len(open(f"out/dev_zh-probe_{t}.md").read())
    print(f"[{t}] probe {bw} chars  expand {len(reports[t])} chars "
          f"({(len(reports[t]) - bw) / bw * 100:+.0f}%)", flush=True)

from dr import llm  # noqa: F401  (imported for parity with sibling scorers)
DRB = os.path.abspath(os.path.join(os.path.dirname(bench.__file__), "..", "..", "deep_research_bench"))
sys.path.insert(0, DRB)
from utils.api import AIClient
from utils.clean_article import ArticleCleaner

RAW = os.path.join(DRB, "data/test_data/raw_data")
CLEAN = os.path.join(DRB, "data/test_data/cleaned_data")
JUDGE = os.environ.get("DR_JUDGE", "openai/gpt-5.5")
JUDGE_BACKEND = os.environ.get("DR_JUDGE_BACKEND", "openrouter")
CLEANER = os.environ.get("DR_CLEANER", "gpt-5-nano")

bench.write_submission(NAME, reports, tasks)
# backend switch AFTER generation (scar: at import time it breaks the DeepSeek ledger routing)
os.environ["LLM_BACKEND"] = "openai"
print(f"cleaning {NAME} with {CLEANER} (zh)...", flush=True)
ArticleCleaner(AIClient(model=CLEANER)).clean_articles(NAME, RAW, CLEAN, max_workers=5, language="zh")
print(f"scoring {NAME} on {JUDGE} via {JUDGE_BACKEND}...", flush=True)
rc, out = bench.run_race(NAME, only_zh=True, force=True, judge_model=JUDGE, backend=JUDGE_BACKEND)
print(f"  {NAME}: rc={rc} scored={sorted(bench.read_scores(NAME))}", flush=True)
E = bench.read_scores(NAME)
if not E:
    print(out[-1800:]); sys.exit(1)

B = bench.read_scores("zh-probe")   # single-call baseline (on disk)
print(f"\n=== ZH EXPANSION A/B (gpt-5.5): per-task delta (expand - probe) ===")
print(f"{'task':>5}  " + "  ".join(f"{LAB[d]:>8}" for d in DIMS))
acc = {d: [] for d in DIMS}
for t in IDS:
    row = []
    for d in DIMS:
        x = E[t][d] - B[t][d]
        acc[d].append(x)
        row.append(f"{x:>+8.4f}")
    print(f"{t:>5}  " + "  ".join(row))
print(f"  avg   " + "  ".join(f"{statistics.mean(acc[d]):>+8.4f}" for d in DIMS))
print("\nabs means:")
for nm, sc in (("probe", B), ("expand", E)):
    print(f"  {nm:8}: " + " ".join(f"{LAB[d]}={statistics.mean(sc[t][d] for t in IDS):.3f}" for d in DIMS))
