"""5-task dev-set runner — the measurement vehicle that gets us above single-task noise.

Generates reports for a FIXED set of official tasks with the current config, scores them all
in ONE official RACE pass, and reports the per-dimension AVERAGE vs the baseline's average on
the same tasks. Config is via the usual env flags + --depth, so each run is a reproducible
ablation.

  DR_USE_JUDGE_FINISH=0 python3 run_devset.py --depth --name ours-base
  DR_PLAN_MODEL=opus DR_USE_JUDGE_FINISH=0 python3 run_devset.py --depth --name ours-opusplan
"""
import os
import sys
import argparse
import statistics

from dr import bench
from dr import data
from dr import engine

DEV_IDS = [51, 61, 71, 81, 91]            # 5 EN tasks, spread across the range (unbiased)
DIMS = ["comprehensiveness", "insight", "instruction_following", "readability", "overall_score"]


def _flag(name, default=True):
    v = os.environ.get(name)
    return default if v is None else v.lower() in ("1", "true", "yes", "on")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", default=DEV_IDS)
    ap.add_argument("--name", default=os.environ.get("DR_SUBMISSION_NAME", "ours-dev"))
    ap.add_argument("--depth", action="store_true")
    ap.add_argument("--max-sources", type=int, default=None)
    args = ap.parse_args()

    max_sources = args.max_sources or (16 if args.depth else 8)
    use_jf = _flag("DR_USE_JUDGE_FINISH")
    tasks = bench.load_tasks()

    print(f"dev set {args.ids}  name={args.name}  depth={args.depth}  "
          f"max_sources={max_sources}  judge_finish={use_jf}\n")

    reports = {}
    for tid in args.ids:
        if tid not in tasks:
            print(f"[{tid}] not in official query.jsonl — skipped", flush=True)
            continue
        print(f"[{tid}] generating ({tasks[tid]['language']})...", flush=True)
        report, trace = engine.research(tasks[tid]["prompt"], use_judge_finish=use_jf,
                                        max_sources=max_sources, depth=args.depth)
        reports[tid] = report
        with open(f"out/dev_{args.name}_{tid}.md", "w") as f:
            f.write(report)
        print(f"[{tid}] {len(report.split())} words, {trace['n_sources']} src, "
              f"{trace['n_claims']} claims", flush=True)

    bench.write_submission(args.name, reports, tasks)
    print("\nscoring all via official RACE (GPT-5.5)...", flush=True)
    rc, out = bench.run_race(args.name, only_en=True, force=True)
    scores = bench.read_scores(args.name)
    if not scores:
        print("no scores produced. harness tail:\n" + out[-1500:])
        sys.exit(1)

    his = data.load_scores()
    ours_d = {d: [] for d in DIMS}
    his_d = {d: [] for d in DIMS}
    print(f"\n{'task':>5} {'ours':>8} {'ref':>8} {'delta':>8}")
    for tid in args.ids:
        s, h = scores.get(tid), his.get(tid, {})
        if not s:
            print(f"{tid:>5}  (no score)")
            continue
        ho = h.get("overall_score", float("nan"))
        print(f"{tid:>5} {s['overall_score']:>8.4f} {ho:>8.4f} {s['overall_score'] - ho:>+8.4f}")
        for d in DIMS:
            ours_d[d].append(s[d])
            if h.get(d) is not None:
                his_d[d].append(h[d])

    print("\n=== 5-task AVERAGE (per dimension) ===")
    for d in DIMS:
        om = statistics.mean(ours_d[d]) if ours_d[d] else float("nan")
        hm = statistics.mean(his_d[d]) if his_d[d] else float("nan")
        print(f"  {d:22} ours {om:.4f}   ref {hm:.4f}   delta {om - hm:+.4f}")


if __name__ == "__main__":
    main()
