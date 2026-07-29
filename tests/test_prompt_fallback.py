"""Tests for the Context Hub system-prompt fallback."""

import logging

import pytest
from langsmith.utils import LangSmithError

import context


class _StubFiles:
    def __init__(self, content):
        self._content = content

    def __getitem__(self, key):
        return type("_Entry", (), {"content": self._content})()


def _stub_client(monkeypatch, content):
    class _Client:
        def pull_agent(self, repo):
            return type("_Agent", (), {"files": _StubFiles(content)})()

    monkeypatch.setattr(context, "Client", _Client)


def test_hub_failure_returns_fallback_and_logs(monkeypatch, caplog):
    class _Client:
        def pull_agent(self, repo):
            raise LangSmithError("hub down")

    monkeypatch.setattr(context, "Client", _Client)

    with caplog.at_level(logging.ERROR, logger="context"):
        assert context.get_prompt() == context._FALLBACK_AGENTS_MD

    assert any(r.levelno == logging.ERROR and r.exc_info for r in caplog.records)


@pytest.mark.parametrize("content", [None, "", "   \n  "])
def test_blank_agents_md_returns_fallback(monkeypatch, caplog, content):
    _stub_client(monkeypatch, content)

    with caplog.at_level(logging.ERROR, logger="context"):
        assert context.get_prompt() == context._FALLBACK_AGENTS_MD

    assert "empty AGENTS.md" in caplog.text


def test_non_empty_agents_md_is_returned(monkeypatch):
    _stub_client(monkeypatch, "real prompt")

    assert context.get_prompt() == "real prompt"


def test_fallback_carries_scope_persona_and_tool_policy():
    fallback = context._FALLBACK_AGENTS_MD

    assert "LangChain ecosystem" in fallback
    assert "Do not use any tools" in fallback
    assert "Hey there!" in fallback
    assert "Happy building!" in fallback


def test_build_agent_raises_when_system_prompt_is_empty(monkeypatch):
    agent_module = pytest.importorskip("agent.agent")
    monkeypatch.setattr(agent_module, "SYSTEM_PROMPT", "   ")

    with pytest.raises(RuntimeError, match=agent_module.CONTEXT_HUB_REPO):
        agent_module.build_agent()
