"""Load reference reports + per-task scores from disk (read-only).

Reference data for scoring comparisons is read from the REF_DIR env var (defaults to
../reference-data). We only READ report/score files for comparison; file names are matched
by glob so nothing here hardcodes a source identity.
"""
import os
import glob
import json

_REF = os.environ.get(
    "REF_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reference-data"),
)


def load_scores() -> dict:
    """{id: {id, prompt, comprehensiveness, insight, instruction_following, readability, overall_score}}"""
    matches = glob.glob(os.path.join(_REF, "results", "race", "*", "raw_results.jsonl"))
    out = {}
    if not matches:
        return out
    with open(matches[0]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[int(r["id"])] = r
    return out


def load_submission() -> dict:
    """{id: {"prompt": ..., "article": ...}} — the reference reports."""
    out = {}
    matches = glob.glob(os.path.join(_REF, "data", "test_data", "raw_data", "*.jsonl"))
    if matches:
        with open(matches[0]) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                out[int(r["id"])] = {"prompt": r.get("prompt", ""),
                                     "article": r.get("article", "")}
        if out:
            return out
    # Fallback: reconstruct from reports/*.md (article) + scores (prompt).
    scores = load_scores()
    rep_dir = os.path.join(_REF, "reports")
    if os.path.isdir(rep_dir):
        for fn in os.listdir(rep_dir):
            if fn.endswith(".md"):
                i = int(fn[:-3])
                with open(os.path.join(rep_dir, fn)) as f:
                    out[i] = {"prompt": scores.get(i, {}).get("prompt", ""), "article": f.read()}
    return out
