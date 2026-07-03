# Deep Research Agent

An original deep-research agent for [DeepResearch Bench](https://github.com/Ayanami0730/deep_research_bench) (DRB):
given a PhD-level research task, it plans, searches the live web, extracts a verified claim
ledger, and writes a long-form, fully-cited research report — then measures itself with the
benchmark's official RACE evaluation harness.

Built solo, from scratch, as a study in how far careful *harness engineering* (measurement,
A/B discipline, structural enforcement) can push report quality on a fixed budget. Every
design decision in this repo was gated by a measured within-task A/B on a fixed dev set —
the experiment scripts that produced those measurements ship alongside the engine.

## How it works

```
task prompt
   │
   ├─ plan        subquestions + outline                          (dr/plan.py)
   ├─ retrieve    web search, full-page reads, source dedup       (dr/retrieve.py, dr/retrieve_depth.py)
   ├─ ledger      per-source claim extraction -> cited evidence   (dr/ledger.py)
   ├─ contract    explicit + implicit deliverables extracted      (dr/write.py, flag-gated)
   ├─ write       one of two measured writer policies:
   │                • single-call writer w/ judge-aligned rules   (dr/write.py)
   │                • per-deliverable section expansion to
   │                  reference-scale length, numeric spine,
   │                  hollow-section re-roll                      (dr/write_expand.py)
   ├─ (optional)  whole-report judge loop / polish / selection    (dr/loop.py, dr/write_polish.py,
   │                                                               dr/select_report.py — flag-gated)
   └─ report      GitHub-flavored markdown with inline citations
```

Layout: `dr/` is the agent package (pipeline stages + benchmark glue), `experiments/` holds
the A/B lab scripts that measured every lever (run as `python3 experiments/ab_<lever>.py`),
and the `run_*.py` entry points live at the root.

Model routing is fully env-configurable (`DR_*_MODEL`); by default heavy prose runs on
Claude (Opus) and cheap extraction on a small model, with any OpenAI-compatible backend
supported (`DR_LLM_BACKEND`).

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env        # add keys; mock mode runs with none
python3 run_engine.py --selftest         # mock end-to-end, no spend
python3 run_engine.py --task 69 --depth  # one live task
python3 run_devset.py --depth            # fixed 5-task dev set + official RACE scoring
```

Evaluation requires the official DRB harness cloned as a sibling directory
(`../deep_research_bench`); `bench.py` wraps its RACE scorer directly so all scores below
are from the benchmark's own judge, not a home-grown metric.

## Evaluation & honest results

Method: a fixed 5-task English dev set (one task per topic area), within-task A/Bs (one
lever at a time, same evidence both arms), scored by the official RACE harness (GPT-5.5
judge). RACE is reference-relative: ~0.50 means parity with the benchmark's reference
report for that task.

| configuration | dev-set RACE (avg) |
|---|---|
| baseline pipeline | 0.457 |
| + depth retrieval (16 sources) | banked, +0.029 within-task |
| + deliverable coverage contract | banked, +0.017 |
| best measured policy (archetype-routed writers) | ~0.471 |
| Chinese tasks, 3-task probe (untuned pipeline) | 0.499 |

Caveats stated plainly: n=5 (n=3 for ZH), single judge, and cross-run absolute scores
drift ~±0.01 — within-run deltas are the trustworthy unit. Negative results were kept and
are visible in the `ab_*` scripts: a post-draft insight-deepening pass, a readability
polish pass, and two per-task candidate-selection designs all failed their A/Bs and were
cut — the harness ships them because knowing what *doesn't* move a judged metric is half
the finding.

Measured findings the design rests on:

- **Comprehensiveness tracks covered material.** Reference reports run ~8–10k words; a
  single-call draft plateaus at ~4.5–6k regardless of length directives. Building the
  report per-deliverable at reference scale (`write_expand.py`) was the largest single
  content lever (+0.019 comprehensiveness, +0.025 insight) — at a readability cost that
  gentle post-editing measurably does not recover.
- **No single writer wins every task type.** Mechanism/data-heavy tasks reward the
  expansion writer; enumeration-heavy tasks reward a tighter single-call writer. Routing
  by task shape beat every fixed policy.
- **Structural enforcement beats prompt pleas.** Deliverable contracts and per-section
  budgets moved scores; politely asking the writer for the same behavior did not.
- **Selection over near-identical candidates is a noise lottery.** Two selector designs
  (holistic and per-dimension pairwise duels) were validated against ground-truth scores
  before being trusted — both failed (length/position bias), which is why they ship
  flag-gated off.

## Configuration

| env var | purpose | default |
|---|---|---|
| `OPENROUTER_API_KEY` / `OPENAI_API_KEY` | worker + judge backends | — |
| `TAVILY_API_KEY` | web retrieval | — |
| `DR_LLM_BACKEND` | cheap-model backend (`openai` / `openrouter`) | `openai` |
| `DR_PLAN_MODEL`, `DR_LEDGER_MODEL`, `DR_WRITER_MODEL`, … | per-stage model routing | see source |
| `DR_COVERAGE_CONTRACT`, `DR_WRITER_RULES`, `DR_RETRIEVAL_DEPTH` | flag-gated stages | off |
| `REF_DIR` | local reference reports/scores for comparisons | `../reference-data` |

## Acknowledgments

The task set, reference reports, and RACE/FACT evaluation methodology are the work of the
[DeepResearch Bench](https://github.com/Ayanami0730/deep_research_bench) team
([arXiv:2506.11763](https://arxiv.org/abs/2506.11763)). The design here was informed by
the DRB paper and the surrounding deep-research-agent literature; all code, prompts, and
measurements in this repository are original.

## License

[MIT](LICENSE)
