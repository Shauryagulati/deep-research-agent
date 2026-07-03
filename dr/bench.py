"""Official DRB RACE referee — wires our engine's reports into the official scorer.

The official harness (../deep_research_bench/) scores a report head-to-head vs the FIXED
reference: S = ours/(ours+ref), where 0.50 = parity with the reference and the leaderboard
number is S*100. The join key is the PROMPT STRING (byte-identical to query.jsonl), NOT the
id — so every submission carries the official prompt. This module writes the submission, runs
the official RACE scorer with an OpenRouter GPT-5.5 judge, and reads back per-task scores.
"""
import os
import json
import subprocess

from . import llm  # importing triggers .env load -> OPENROUTER_API_KEY in os.environ

DRB_DIR = os.environ.get(
    "DRB_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "deep_research_bench"))
QUERY_FILE = os.path.join(DRB_DIR, "data", "prompt_data", "query.jsonl")
RAW_DATA_DIR = os.path.join(DRB_DIR, "data", "test_data", "raw_data")


def load_tasks():
    """Official tasks {id: {id, prompt, language, topic}} from query.jsonl."""
    tasks = {}
    with open(QUERY_FILE) as f:
        for line in f:
            o = json.loads(line)
            tasks[int(o["id"])] = o
    return tasks


def write_submission(model: str, reports: dict, tasks=None) -> str:
    """reports = {id: article}. Writes <raw_data>/<model>.jsonl carrying the OFFICIAL prompt
    (byte-identical join key). Returns the path."""
    tasks = tasks or load_tasks()
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    path = os.path.join(RAW_DATA_DIR, f"{model}.jsonl")
    with open(path, "w") as f:
        for tid, article in sorted(reports.items()):
            f.write(json.dumps({"id": tid, "prompt": tasks[tid]["prompt"],
                                "article": article}, ensure_ascii=False) + "\n")
    return path


HARNESS_BACKEND = os.environ.get("DR_HARNESS_BACKEND", "openai")  # OpenRouter funds gone -> OpenAI direct
_HARNESS_KEY_ENV = {"openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}


def run_race(model: str, *, only_en=False, only_zh=False, limit=None, force=True,
             skip_cleaning=False, max_workers=5, timeout=3000, judge_model=None,
             backend=HARNESS_BACKEND):
    """Run the official RACE scorer. Returns (rc, output).

    backend selects the harness LLM_BACKEND ('openai' direct by default; 'openrouter' legacy).
    judge_model overrides the harness RACE_MODEL env var (default flagship for that backend). For
    the dev A/B, judge with the full flagship (gpt-5.2 on OpenAI) — a cheap model is NOT a valid
    judge (calibration proved it). The same model also drives the article-cleaning stage; pre-clean
    with a cheap model + pass an already-cleaned file to keep the flagship to the scoring pass."""
    key_env = _HARNESS_KEY_ENV.get(backend, "OPENAI_API_KEY")
    if not os.environ.get(key_env):
        raise RuntimeError(f"{key_env} not set (needed for the {backend} RACE judge).")
    cmd = ["python3", "-u", "deepresearch_bench_race.py", model,
           "--raw_data_dir", "data/test_data/raw_data",
           "--query_file", "data/prompt_data/query.jsonl",
           "--output_dir", f"results/race/{model}",
           "--max_workers", str(max_workers)]
    if only_en:
        cmd.append("--only_en")
    if only_zh:
        cmd.append("--only_zh")
    if limit:
        cmd += ["--limit", str(limit)]
    if force:
        cmd.append("--force")
    if skip_cleaning:
        cmd.append("--skip_cleaning")
    env = dict(os.environ, LLM_BACKEND=backend)
    if judge_model:
        env["RACE_MODEL"] = judge_model
    r = subprocess.run(cmd, cwd=DRB_DIR, env=env, text=True,
                       capture_output=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def read_scores(model: str) -> dict:
    """Per-task official RACE {id: {comprehensiveness, insight, instruction_following,
    readability, overall_score}} from the harness output."""
    path = os.path.join(DRB_DIR, "results", "race", model, "raw_results.jsonl")
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                o = json.loads(line)
                out[int(o["id"])] = o
    return out
