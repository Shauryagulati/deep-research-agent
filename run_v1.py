"""v1 entrypoint — the judge-loop go/no-go experiment.

  python3 run_v1.py --selftest          # mock mode, no key: proves the plumbing
  python3 run_v1.py --tasks 10          # with key: run the loop on 10 reference reports
  python3 run_v1.py --ids 69 41 48      # specific task ids

What it does: takes a reference baseline report for a task, runs the whole-report judge-loop
on it (score -> patch weakest dim -> re-score -> keep iff improved), and reports whether the
loop lifted the score versus the baseline's own per-task score.
"""
import os
import sys
import argparse

from dr import data
from dr import judge
from dr import loop
from dr import proxy


def _row(i, ref, before, after, steps):
    return f"{i:>5} {ref:>13.4f} {before:>11.4f} {after:>10.4f} {after - before:>+9.4f} {steps:>6}"


def run(ids):
    subs = data.load_submission()
    scores = data.load_scores()
    mock = not os.environ.get("OPENROUTER_API_KEY")
    print(f"mode: {'MOCK (no key — plumbing only)' if mock else 'REAL (OpenRouter)'}\n")
    print(f"{'task':>5} {'ref_official':>13} {'our_before':>11} {'our_after':>10} {'delta':>9} {'steps':>6}")
    print("-" * 60)
    deltas = []
    for i in ids:
        if i not in subs:
            print(f"{i:>5}  (no report on disk — skipped)")
            continue
        task, report = subs[i]["prompt"], subs[i]["article"]
        crit = judge.regenerate_criteria(task)
        improved, after, hist = loop.judge_loop(report, task, crit)
        before = hist[0][1]
        ref = scores.get(i, {}).get("overall_score", float("nan"))
        print(_row(i, ref, before.overall, after.overall, len(hist) - 1))
        deltas.append(after.overall - before.overall)
    if deltas:
        print("-" * 60)
        print(f"mean self-score lift across {len(deltas)} task(s): {sum(deltas)/len(deltas):+.4f}")
        if mock:
            print("(MOCK — real numbers require a key.)")


def _degrade_to_walls(report: str) -> str:
    """Merge consecutive body paragraphs into wall-of-text blocks (headings kept). Used ONLY
    to build a deliberately-improvable input so the self-test can exercise the accept path."""
    out, buf = [], []
    for b in report.split("\n\n"):
        if b.lstrip().startswith("#"):
            if buf:
                out.append(" ".join(buf))
                buf = []
            out.append(b)
        elif b.strip():
            buf.append(b.strip())
    if buf:
        out.append(" ".join(buf))
    return "\n\n".join(out)


def selftest() -> int:
    """Mock-mode proof the loop runs end-to-end, exercising BOTH the reject (gating) and
    accept paths on real report text — no key, no spend."""
    os.environ["DR_MOCK"] = "1"  # force mock even when .env keys are present — never spend here
    subs = data.load_submission()
    tid = 69 if 69 in subs else (sorted(subs)[0] if subs else None)
    if tid is None:
        print("[selftest] no reference reports found on disk under REF_DIR")
        return 1
    task, report = subs[tid]["prompt"], subs[tid]["article"]
    crit = judge.regenerate_criteria(task)
    mode = "MOCK"  # forced above
    print(f"[selftest] task {tid}   mode={mode}\n")

    # (1) Real report: the loop must never make it worse (judge-gated).
    imp1, after1, h1 = loop.judge_loop(report, task, crit)
    before1 = h1[0][1]
    gated_ok = after1.overall >= before1.overall - 1e-9
    print("  (1) real report      — gating / no-op handling")
    print(f"      overall {before1.overall:.4f} -> {after1.overall:.4f}   history={[h[0] for h in h1]}")
    print(f"      readability proxy {proxy.readability_proxy(report):.4f} (baseline already de-walled)")

    # (2) Deliberately degraded copy: the loop must DETECT readability, patch it, and ACCEPT.
    degraded = _degrade_to_walls(report)
    imp2, after2, h2 = loop.judge_loop(degraded, task, crit)
    before2 = h2[0][1]
    accepted = any(lbl.startswith("accept") for lbl, _ in h2)
    accept_ok = accepted and after2.overall >= before2.overall
    print("  (2) degraded copy    — accept path (readability fix)")
    print(f"      overall {before2.overall:.4f} -> {after2.overall:.4f}   history={[h[0] for h in h2]}")
    print(f"      readability proxy {proxy.readability_proxy(degraded):.4f} -> {proxy.readability_proxy(imp2):.4f}")

    ok = gated_ok and accept_ok
    print(f"\n  gating-never-worse: {'PASS' if gated_ok else 'FAIL'}    accept-path-fires: {'PASS' if accept_ok else 'FAIL'}")
    print(f"  SELFTEST: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tasks", type=int, help="run on the first N task ids")
    ap.add_argument("--ids", type=int, nargs="+", help="run on specific task ids")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if args.ids:
        run(args.ids)
    elif args.tasks:
        run(list(range(1, args.tasks + 1)))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
