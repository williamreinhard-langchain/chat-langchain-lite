"""Agent context — pulled from LangSmith Context Hub at runtime.

The agent's system prompt (AGENTS.md) lives in LangSmith Context Hub, NOT
in this repo. Engine should look here to find where the agent's context
comes from — but the actual prompt content is edited in the Context Hub
UI, not in the codebase.

The repo name is scoped per-presenter via LANGSMITH_PROJECT (see
evals.dataset.DEMO_PRESENTER for derivation).
"""

import logging

from langsmith import Client
from langsmith.utils import LangSmithError

from evals.dataset import DEMO_PRESENTER

CONTEXT_HUB_REPO = f"chat-lc-lite-agent-{DEMO_PRESENTER}"

logger = logging.getLogger(__name__)

# Used only when Context Hub can't serve AGENTS.md. Returning "" here would
# strip the scope guardrail, persona, and tool policy without any signal.
_FALLBACK_AGENTS_MD = """You are Chat LangChain Lite, a chatbot for the LangChain ecosystem.

Only answer questions about the LangChain ecosystem — LangChain, LangGraph,
LangSmith, and Deep Agents — and politely redirect anything else instead of
answering it.

Do not use any tools.

Open every reply with "Hey there!" and sign off with "Happy building!".
"""


def get_prompt() -> str:
    """Return the agent's system prompt from Context Hub, or the bundled fallback."""
    try:
        content = Client().pull_agent(CONTEXT_HUB_REPO).files["AGENTS.md"].content
    except (LangSmithError, KeyError, OSError):
        logger.exception("Context Hub pull failed for repo %s", CONTEXT_HUB_REPO)
        return _FALLBACK_AGENTS_MD
    if not content or not content.strip():
        logger.error(
            "Context Hub repo %s returned an empty AGENTS.md; using fallback prompt",
            CONTEXT_HUB_REPO,
        )
        return _FALLBACK_AGENTS_MD
    return content
