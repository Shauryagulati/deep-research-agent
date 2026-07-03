"""Selector validation — run criteria-anchored per-task selection over the five scored writer
arms and measure it against ground truth we already own (each arm's official gpt-5.5 RACE
score). No new judging spend: the selector runs on Opus via Max ($0) and is graded against
on-disk scores. Reports: what the selector picked per task, the official overall of the picks,
and how much of the oracle's gain over the best fixed arm the selector captured.

  python3 experiments/ab_select.py
"""
import _bootstrap  # noqa: F401  (repo root on sys.path)
import json
import os
import statistics

from dr import bench
from dr import select_report

IDS = [51, 61, 71, 81, 91]
ARMS = ["wA-vanilla", "wB-tuned", "wC-pruned", "wE-expand", "wF-polish"]
CRIT = os.path.abspath(os.path.join(os.path.dirname(bench.__file__), "..", "..",
                                    "deep_research_bench/data/criteria_data/criteria.jsonl"))

tasks = bench.load_tasks()
criteria = {}
with open(CRIT) as f:
    for line in f:
        d = json.loads(line)
        if d["id"] in IDS:
            criteria[d["id"]] = d

scores = {a: bench.read_scores(a) for a in ARMS}

PICKS_F = "out/select_picks_v2.json"   # resume cache — a Max session limit can kill a run mid-way
picks = json.load(open(PICKS_F)) if os.path.exists(PICKS_F) else {}
picks = {int(k): v for k, v in picks.items()}

picked_scores = []
for t in IDS:
    if t not in picks:
        cands = {a: open(f"out/dev_{a}_{t}.md").read() for a in ARMS}
        print(f"[{t}] selecting among {len(cands)} arms...", flush=True)
        win, duels = select_report.select_report(
            tasks[t]["prompt"], criteria[t], cands,
            log=lambda m: print(m, flush=True))
        picks[t] = win
        json.dump({str(k): v for k, v in picks.items()}, open(PICKS_F, "w"))
    win = picks[t]
    picked_scores.append(scores[win][t]["overall_score"])
    truth = max(ARMS, key=lambda a: scores[a][t]["overall_score"])
    mark = "HIT " if win == truth else "miss"
    print(f"[{t}] pick={win} ({scores[win][t]['overall_score']:.4f})  "
          f"oracle={truth} ({scores[truth][t]['overall_score']:.4f})  [{mark}]", flush=True)

print("\n=== selector validation (official gpt-5.5 overalls) ===")
for a in ARMS:
    print(f"  fixed {a:12}: {statistics.mean(scores[a][t]['overall_score'] for t in IDS):.4f}")
oracle = statistics.mean(max(scores[a][t]["overall_score"] for a in ARMS) for t in IDS)
sel = statistics.mean(picked_scores)
best_fixed = max(statistics.mean(scores[a][t]["overall_score"] for t in IDS) for a in ARMS)
print(f"  SELECTED          : {sel:.4f}   picks: " + " ".join(f"{t}:{picks[t][1:3]}" for t in IDS))
print(f"  oracle            : {oracle:.4f}")
cap = (sel - best_fixed) / (oracle - best_fixed) if oracle > best_fixed else 0.0
print(f"  oracle-gain capture: {cap:.0%}  (selected-best_fixed / oracle-best_fixed)")
