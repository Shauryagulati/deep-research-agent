"""Deep-research agent package: pipeline stages + official-benchmark evaluation glue."""
from . import llm as _llm  # noqa: F401 — imports first so .env is loaded before any
#                            sibling module reads env-driven defaults at import time
