"""Referee CLI — score our reports with the OFFICIAL DRB RACE harness.

  python3 run_bench.py --score-existing 69                 # score out/last_report.md as task 69
  python3 run_bench.py --score-existing 69 --report PATH    # score an explicit report file

Prints our official RACE (head-to-head vs the fixed reference) and, when available, the
baseline's own official score for the same task — a real within-task A/B data point.
0.50 = parity with the reference; leaderboard number = score * 100.
"""
import os
import sys
import argparse

from dr import bench
from dr import data

MODEL = os.environ.get("DR_SUBMISSION_NAME", "ours-v1")


def _fmt(s):
    return (f"overall {s['overall_score']:.4f}  (C {s['comprehensiveness']:.3f}  "
            f"I {s['insight']:.3f}  Inst {s['instruction_following']:.3f}  "
            f"R {s['readability']:.3f})")


def score_existing(tid: int, report_path: str) -> int:
    if not os.path.exists(report_path):
        print(f"report not found: {report_path}")
        return 1
    with open(report_path) as f:
        article = f.read()
    tasks = bench.load_tasks()
    if tid not in tasks:
        print(f"task {tid} not in official query.jsonl")
        return 1
    lang = tasks[tid]["language"]
    path = bench.write_submission(MODEL, {tid: article}, tasks)
    print(f"submission: {path}  (task {tid}, {lang}, {len(article.split())} words)")
    print("running official RACE (GPT-5.5 judge via OpenRouter) — cleaning + scoring...\n")
    rc, out = bench.run_race(MODEL, only_en=(lang == "en"), only_zh=(lang == "zh"), force=True)
    scores = bench.read_scores(MODEL)
    if tid not in scores:
        print("no score produced. harness output tail:\n" + out[-1800:])
        return 1
    ours = scores[tid]
    his = data.load_scores().get(tid, {})
    print(f"=== OFFICIAL RACE — task {tid} ===")
    print(f"  ours:      {_fmt(ours)}")
    if his.get("overall_score") is not None:
        print(f"  baseline:  overall {his['overall_score']:.4f}  (its own official score)")
        print(f"  delta:     {ours['overall_score'] - his['overall_score']:+.4f}")
    print("\n  0.50 = parity with the fixed reference; leaderboard number = score * 100.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-existing", type=int, metavar="ID",
                    help="score a saved report as this official task id")
    ap.add_argument("--report", type=str, default="out/last_report.md",
                    help="report file to score (default out/last_report.md)")
    args = ap.parse_args()

    if args.score_existing is not None:
        sys.exit(score_existing(args.score_existing, args.report))
    ap.print_help()


if __name__ == "__main__":
    main()
