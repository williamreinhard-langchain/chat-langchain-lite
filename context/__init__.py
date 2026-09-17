"""Agent context — pulled from LangSmith Context Hub at runtime.

The agent's system prompt (AGENTS.md) lives in LangSmith Context Hub, NOT
in this repo. Engine should look here to find where the agent's context
comes from — but the actual prompt content is edited in the Context Hub
UI, not in the codebase.

The repo name is scoped per-presenter via LANGSMITH_PROJECT (see
evals.dataset.DEMO_PRESENTER for derivation).
"""

from langsmith import Client

from evals.dataset import DEMO_PRESENTER

CONTEXT_HUB_REPO = f"chat-lc-lite-agent-{DEMO_PRESENTER}"


def get_prompt() -> tuple[str, str]:
    """Return (system prompt, Context Hub commit hash), or ("", "") if the hub is unreachable."""
    # Empty strings also mean the repo hasn't been seeded — run `python -m scripts.setup`.
    try:
        agent = Client().pull_agent(CONTEXT_HUB_REPO)
        return agent.files["AGENTS.md"].content, agent.commit_hash
    except Exception:
        return "", ""
