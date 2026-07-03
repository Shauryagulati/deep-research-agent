"""Path bootstrap so experiment scripts run as plain `python3 experiments/ab_x.py` from the
repo root: puts the repo root (parent of this dir) on sys.path so `from dr import ...` works.

Also aliases the package modules under their historical top-level names so evidence caches
pickled before the package restructure (which reference e.g. `plan.Plan`) still load."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dr import ledger, plan, retrieve  # noqa: E402

sys.modules.setdefault("plan", plan)
sys.modules.setdefault("retrieve", retrieve)
sys.modules.setdefault("ledger", ledger)
