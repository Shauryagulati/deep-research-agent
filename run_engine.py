"""Deep-research engine entrypoint.

  python3 run_engine.py --selftest      # mock end-to-end plumbing on a real task (no spend)
  python3 run_engine.py --task 69       # run on DRB task 69 (LIVE — uses keys)
  python3 run_engine.py --prompt "..."  # run on an ad-hoc prompt (LIVE)

Ablation flags (env): DR_USE_LEDGER=0 disables stage 3, DR_USE_JUDGE_FINISH=0 disables stage 5.
"""
import os
import sys
import argparse

from dr import data
from dr import engine


def _flag(name: str, default: bool = True) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.lower() in ("1", "true", "yes", "on")


def _run(task_prompt: str, label: str, *, depth=False, max_sources=8,
         out_path="out/last_report.md"):
    report, trace = engine.research(
        task_prompt,
        use_ledger=_flag("DR_USE_LEDGER"),
        use_judge_finish=_flag("DR_USE_JUDGE_FINISH"),
        max_sources=max_sources,
        depth=depth,
    )
    sections = report.count("\n## ")
    print(f"=== {label}{' [depth]' if depth else ''} ===")
    print(f"sub-questions: {len(trace['subquestions'])}   "
          f"sources: {trace['n_sources']}   claims: {trace['n_claims']}")
    if "finish_history" in trace:
        print(f"judge-finish:  {trace['finish_history']}")
    print(f"report:        {len(report.split())} words, {sections} sections")
    if not os.environ.get("DR_MOCK"):           # never let a mock run clobber a real report
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report)
        print(f"full report:   {out_path}")
    print("-" * 60)
    print(report[:1200])
    return report, trace


def finish(report: str, task_prompt: str):
    """Run stage 5 (judge-finish) LIVE on an existing report — isolates the judge-loop so we
    can see the real GPT-5.5 dim scores, which dimension binds, and whether a patch is kept."""
    from dr import judge
    from dr import loop
    print(f"[finish] stage 5 LIVE on a {len(report.split())}-word report\n")
    criteria = judge.regenerate_criteria(task_prompt)
    final, score, hist = loop.judge_loop(report, task_prompt, criteria)
    for label, s in hist:
        dims = {k: round(v, 3) for k, v in s.dims.items()}
        print(f"  {label:48} overall={s.overall:.4f}  {dims}")
    init = hist[0][1]
    print(f"\ninitial weakest: {init.weakest()}")
    print(f"overall {init.overall:.4f} -> {score.overall:.4f}  "
          f"(delta {score.overall - init.overall:+.4f})")
    print(f"words   {len(report.split())} -> {len(final.split())}")
    os.makedirs("out", exist_ok=True)
    with open("out/last_report.finished.md", "w") as f:
        f.write(final)
    print("saved   out/last_report.finished.md")
    return final


def selftest() -> int:
    os.environ["DR_MOCK"] = "1"  # force mock even though keys exist — no spend
    subs = data.load_submission()
    if not subs:
        print("[selftest] no task prompts found under REF_DIR")
        return 1
    tid = 69 if 69 in subs else sorted(subs)[0]
    print(f"[selftest] mock end-to-end on task {tid}\n")
    report, trace = _run(subs[tid]["prompt"], f"task {tid} (MOCK)")
    ok = bool(report.strip()) and trace["n_sources"] > 0 and "\n## " in report
    print("\nSELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def smoke() -> int:
    """LIVE stages 1-3 on one task — confirms keys + retrieval work, stops before the writer."""
    from dr import plan as plan_mod
    from dr import retrieve as retrieve_mod
    from dr import ledger as ledger_mod

    subs = data.load_submission()
    if not subs:
        print("[smoke] no task prompts found under REF_DIR")
        return 1
    tid = 69 if 69 in subs else sorted(subs)[0]
    task = subs[tid]["prompt"]
    print(f"[smoke] LIVE stages 1-3 on task {tid}\n")

    p = plan_mod.make_plan(task)
    print(f"plan:     {len(p.subquestions)} sub-questions, {len(p.outline)} outline sections")
    for q in p.subquestions:
        print(f"  - {q}")

    srcs = retrieve_mod.gather(p.subquestions, max_sources=6)
    print(f"\nretrieve: {len(srcs)} sources (full-page)")
    for s in srcs:
        print(f"  - [{len(s.text.split()):>5} words] {s.title[:66]}  {s.url}")

    claims = ledger_mod.build_ledger(srcs, task)
    print(f"\nledger:   {len(claims)} claims")
    for c in claims[:6]:
        print(f"  - {c.text[:110]}  ({c.source_url})")

    ok = len(srcs) > 0 and len(claims) > 0
    print("\nSMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="LIVE stages 1-3 on one task")
    ap.add_argument("--task", type=int)
    ap.add_argument("--prompt", type=str)
    ap.add_argument("--finish", type=str, metavar="PATH",
                    help="run stage 5 LIVE on an existing report file (needs --task for the prompt)")
    ap.add_argument("--depth", action="store_true", help="depth mode: more sources + longer writer")
    ap.add_argument("--max-sources", type=int, default=None)
    ap.add_argument("--out", type=str, default="out/last_report.md", help="report save path")
    args = ap.parse_args()
    max_sources = args.max_sources or (16 if args.depth else 8)

    if args.selftest:
        sys.exit(selftest())
    if args.smoke:
        sys.exit(smoke())
    if args.finish:
        subs = data.load_submission()
        if args.task is None or args.task not in subs:
            print("--finish needs --task N (for the task prompt)")
            sys.exit(1)
        with open(args.finish) as f:
            finish(f.read(), subs[args.task]["prompt"])
        return
    if args.task is not None:
        subs = data.load_submission()
        if args.task not in subs:
            print(f"task {args.task} not found on disk")
            sys.exit(1)
        _run(subs[args.task]["prompt"], f"task {args.task}",
             depth=args.depth, max_sources=max_sources, out_path=args.out)
    elif args.prompt:
        _run(args.prompt, "ad-hoc prompt",
             depth=args.depth, max_sources=max_sources, out_path=args.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
