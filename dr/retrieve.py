"""Stage 2 — retrieval: Tavily search per sub-question, FULL-PAGE reads, dedup by URL.

The deliberate edge (vs snippet-only systems): we keep the full page text, not a ~125-char
quote, so the writer gets real evidence per source. Tavily is called over plain stdlib HTTP
(no extra dependency). Mock mode (no TAVILY_API_KEY, or DR_MOCK) returns deterministic stubs.
"""
from dataclasses import dataclass
import os
import json
import urllib.request
import urllib.error

TAVILY_URL = "https://api.tavily.com/search"
TAVILY_MAX_QUERY = 400   # Tavily hard-rejects queries longer than this (HTTP 400)


@dataclass
class Source:
    url: str
    title: str
    text: str       # full-page content — the edge over snippet-only retrieval
    query: str


def _mock_search(query: str, k: int):
    return [Source(url=f"https://example.com/{abs(hash(query)) % 10000}/{i}",
                   title=f"Stub source {i} for: {query[:40]}",
                   text=(f"Full-page stub content discussing {query}. "
                         "It contains several concrete, quantified details. ") * 12,
                   query=query)
            for i in range(k)]


def _tavily_search(query: str, k: int):
    key = os.environ.get("TAVILY_API_KEY")
    if os.environ.get("DR_MOCK") or not key:
        return _mock_search(query, k)
    query = (query or "").strip()[:TAVILY_MAX_QUERY]   # Tavily rejects > 400 chars
    body = json.dumps({
        "api_key": key,
        "query": query,
        "max_results": k,
        "search_depth": "advanced",
        "include_raw_content": True,
    }).encode()
    req = urllib.request.Request(TAVILY_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            data = json.loads(r.read().decode())
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        # one bad sub-question must not kill the whole run — skip it, keep the others
        print(f"[retrieve] Tavily skipped a query ({e}): {query[:60]!r}", flush=True)
        return []
    out = []
    for res in data.get("results", []):
        out.append(Source(
            url=res.get("url", ""),
            title=res.get("title", ""),
            text=res.get("raw_content") or res.get("content", "") or "",
            query=query,
        ))
    return out


def gather(subquestions, max_sources: int = 8, per_q: int = 4):
    """Search each sub-question, fetch full pages, dedup by URL, cap at max_sources."""
    seen, sources = set(), []
    for q in subquestions:
        for s in _tavily_search(q, per_q):
            if not s.url or s.url in seen:
                continue
            seen.add(s.url)
            sources.append(s)
            if len(sources) >= max_sources:
                return sources
    return sources
