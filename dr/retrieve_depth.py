"""Retrieval depth (P3) — IR processing behind the DR_RETRIEVAL_DEPTH flag.

On a multi-sub-question Tavily pull, replace "concat + dedup + cap" with:
    RRF fusion (k=60, free/deterministic)
      -> cross-encoder rerank (MiniLM, local CPU, free) -> keep top-K on-target
      -> condense each kept full page to on-target evidence (cheap LLM).

The sufficiency gap-pass is intentionally NOT here yet — it adds NEW sources, so it can't be
isolated from retrieval noise in a clean A/B (it's the optional second step).
"""
import os

from . import retrieve
from . import llm

RRF_K = 60
RERANK_MODEL = os.environ.get("DR_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CONDENSE_MODEL = os.environ.get("DR_CONDENSE_MODEL", "gpt-4o-mini")  # cheapest adequate OpenAI tier
TOP_K = int(os.environ.get("DR_RETRIEVAL_TOPK", "16"))
PER_Q = int(os.environ.get("DR_RETRIEVAL_PER_Q", "6"))  # uncapped per-sub-question pull width

CONDENSE_SYS = (
    "Condense the web page below to ONLY the evidence relevant to the research task. Preserve "
    "EVERY specific fact, number, date, named entity, and quantified finding that bears on the "
    "task; drop navigation, ads, boilerplate, and off-topic passages. Do NOT add anything not "
    "present in the page. Output the condensed on-target text only — no preamble, no commentary."
)

_ce = None


def _cross_encoder():
    global _ce
    if _ce is None:
        from sentence_transformers import CrossEncoder
        _ce = CrossEncoder(RERANK_MODEL)
    return _ce


def raw_pull(subquestions, per_q=PER_Q):
    """Per-sub-question Tavily lists, UNCAPPED and NOT cross-deduped — the shared pull both A/B
    arms process. Returns [(subquestion, [Source, ...]), ...]."""
    return [(q, retrieve._tavily_search(q, per_q)) for q in subquestions]


def base_process(raw, max_sources=TOP_K):
    """Arm A (current processing): concat in sub-question order, dedup by URL, cap."""
    seen, out = set(), []
    for _, results in raw:
        for s in results:
            if not s.url or s.url in seen:
                continue
            seen.add(s.url)
            out.append(s)
            if len(out) >= max_sources:
                return out
    return out


def rrf_fuse(raw, k=RRF_K):
    """Reciprocal Rank Fusion across the per-sub-question lists -> one ranked, de-duplicated list.
    A source ranked high (or found by multiple sub-questions) accumulates score 1/(k+rank)."""
    score, src = {}, {}
    for _, results in raw:
        for rank, s in enumerate(results):
            if not s.url:
                continue
            score[s.url] = score.get(s.url, 0.0) + 1.0 / (k + rank + 1)
            src.setdefault(s.url, s)
    return sorted(src.values(), key=lambda s: score[s.url], reverse=True)


def rerank(sources, subquestions, top_k=TOP_K):
    """Cross-encoder rerank: each source scored by its MAX relevance to any sub-question
    (title + lead text vs sub-question). Keep the top_k most on-target."""
    if not sources or not subquestions:
        return sources[:top_k]
    ce = _cross_encoder()
    docs = [f"{s.title}\n{s.text[:1200]}" for s in sources]
    pairs, doc_idx = [], []
    for di, d in enumerate(docs):
        for q in subquestions:
            pairs.append((q, d))
            doc_idx.append(di)
    scores = ce.predict(pairs)
    best = [float("-inf")] * len(docs)
    for i, sc in zip(doc_idx, scores):
        best[i] = max(best[i], float(sc))
    order = sorted(range(len(sources)), key=lambda i: best[i], reverse=True)
    return [sources[i] for i in order[:top_k]]


def condense(sources, task_prompt, model=CONDENSE_MODEL):
    """Condense each kept full page to its on-target evidence (cheap LLM). Keeps the full-page
    READ edge but feeds focused text downstream. Fail-soft: keep the original text if the
    condense returns nothing usable."""
    out = []
    for s in sources:
        user = (f"<task>\n{task_prompt}\n</task>\n\n<page>\n{s.title}\nURL: {s.url}\n"
                f"{s.text[:8000]}\n</page>")
        cond = (llm.chat(model, CONDENSE_SYS, user, temperature=0.0, max_tokens=1500) or "").strip()
        out.append(retrieve.Source(url=s.url, title=s.title,
                                   text=cond if len(cond) >= 80 else s.text, query=s.query))
    return out


def depth_pipeline(subquestions, task_prompt, *, raw=None, per_q=PER_Q, top_k=TOP_K):
    """Full P3 processing: (raw_pull ->) RRF -> rerank -> condense. Returns condensed sources."""
    if raw is None:
        raw = raw_pull(subquestions, per_q)
    return condense(rerank(rrf_fuse(raw), subquestions, top_k), task_prompt)
