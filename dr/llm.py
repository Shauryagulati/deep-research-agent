"""LLM client for OpenRouter (OpenAI-compatible), with a zero-dependency MOCK mode.

Mock mode is active whenever OPENROUTER_API_KEY is absent — so the whole v1 harness
runs tonight on stdlib alone (no `openai` install, no network, no spend). The real
`openai` SDK is imported lazily, only when a key is present and a call is actually made.
"""
import os
import time
import hashlib
import subprocess

# Backend for cheap-labor calls (plan / ledger / extraction / condense). OpenRouter funds are
# gone, so the default is OpenAI direct; OpenRouter stays reachable via DR_LLM_BACKEND=openrouter.
# (Opus/Sonnet always go through Claude Max regardless of backend — see _is_max_model.)
_BACKENDS = {
    "openai":     ("https://api.openai.com/v1", "OPENAI_BASE_URL", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_BASE_URL", "OPENROUTER_API_KEY"),
}
LLM_BACKEND = os.environ.get("DR_LLM_BACKEND", "openai").lower()


def _backend():
    """Return (base_url, key_env_name) for the active cheap-labor backend."""
    default_base, base_env, key_env = _BACKENDS.get(LLM_BACKEND, _BACKENDS["openai"])
    return os.environ.get(base_env, default_base), key_env


def _load_env() -> None:
    # Load agent/.env (KEY=VALUE lines) with the stdlib — no third-party dependency, so
    # mock mode stays zero-install and real keys load deterministically (inline comments
    # after the value are stripped). Existing process env always wins.
    path = os.path.join(os.path.dirname(__file__), "..", ".env")
    try:
        with open(path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip()
                if " #" in val:
                    val = val.split(" #", 1)[0].strip()
                val = val.strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
    except FileNotFoundError:
        pass


_load_env()


def have_key() -> bool:
    # DR_MOCK forces mock mode even when keys are present (plumbing tests, no spend).
    if os.environ.get("DR_MOCK"):
        return False
    return bool(os.environ.get(_backend()[1]))


def _is_max_model(model: str) -> bool:
    """Models served by the Claude Max subscription via `claude -p` ($0), not OpenRouter."""
    m = (model or "").lower()
    return m in ("opus", "sonnet", "haiku") or m.startswith("claude")


def _is_rate_limit(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in
               ("rate limit", "usage limit", "limit reached", "429", "too many requests"))


def _claude_max_call(system: str, user: str, model: str, attempts: int = 4) -> str:
    """Opus/Sonnet via Claude Max (`claude -p`). API-key auth is stripped from the subprocess
    env so it uses the subscription login ($0). System prompt is appended; user text via stdin.
    Tools are disabled so the model answers from the prompt only (no autonomous research)."""
    cmd = ["claude", "-p", "--model", model, "--output-format", "text",
           "--disallowedTools", "WebSearch,WebFetch,Bash,Edit,Write,NotebookEdit,Task,Agent,Skill"]
    if system:
        cmd += ["--append-system-prompt", system]
    env = {k: v for k, v in os.environ.items()
           if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    last = ""
    for _ in range(attempts):
        try:
            r = subprocess.run(cmd, input=user, text=True,
                               capture_output=True, timeout=480, env=env)
        except FileNotFoundError:
            raise RuntimeError("`claude` CLI not found — required for Max (Opus) calls.")
        except subprocess.TimeoutExpired:
            last = "timeout"
            continue
        out = (r.stdout or "").strip()
        if r.returncode == 0 and out:
            return out
        last = ((r.stderr or "") + (r.stdout or "")).strip()
        time.sleep(120 if _is_rate_limit(last) else 5)
    raise RuntimeError(f"claude -p (Max) failed after {attempts} attempts: {last[:300]}")


def chat(model: str, system: str, user: str, *,
         temperature: float = 0.3, max_tokens: int = 4000, mock_fn=None) -> str:
    """Single-shot chat completion.

    In mock mode: returns mock_fn(system, user) if given, else a deterministic stub.
    In real mode: calls the active backend (OpenAI direct by default) via the OpenAI SDK.
    """
    if not have_key():
        if mock_fn is not None:
            return mock_fn(system, user)
        digest = hashlib.sha256((model + system + user).encode()).hexdigest()[:8]
        return f"[MOCK:{digest}]"

    if _is_max_model(model):                    # Opus/Sonnet via Claude Max (claude -p), $0
        return _claude_max_call(system, user, model)

    from openai import OpenAI  # lazy — only needed for real calls
    base, key_env = _backend()
    client = OpenAI(base_url=base, api_key=os.environ[key_env])
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
