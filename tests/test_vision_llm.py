"""Unit tests for the cloud vision-LLM engines (no network, no API cost).

Regression cover for the 2026-11-03 DeepSeek run on ami_organics.pdf:
deepseek-flash defaulted to thinking mode, spent all 16 000 output tokens on
hidden reasoning, returned empty content — and the engine reported "OK".
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from engines import vision_llm
from engines.vision_llm import PageResult, markdown_to_plain, run_engine

PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
    "53de0000000c4944415408d763f8ffff3f0005fe02fea7d6a4b20000000049454e44ae426082"
)


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    monkeypatch.setattr(vision_llm, "load_env", lambda: None)
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.setenv(k, "test-key")


# ---------------------------------------------------------------------------
# run_engine
# ---------------------------------------------------------------------------
def test_empty_truncated_page_is_an_error_not_success():
    res = run_engine("X", "m", "OPENAI_API_KEY", PNG_1PX, "p.png",
                     lambda b64, mime: PageResult(text="", output_tokens=16000, truncated=True))
    assert res.get("error")
    assert "token limit" in res["error"]


def test_non_empty_truncated_page_is_kept_with_warning():
    res = run_engine("X", "m", "OPENAI_API_KEY", PNG_1PX, "p.png",
                     lambda b64, mime: PageResult(text="some text", truncated=True))
    assert not res.get("error")
    assert res["usage"]["truncated_pages"] == [1]


def test_missing_api_key_is_reported(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    res = run_engine("X", "m", "OPENAI_API_KEY", PNG_1PX, "p.png", lambda b, m: PageResult("x"))
    assert "OPENAI_API_KEY is not set" in res["error"]


# ---------------------------------------------------------------------------
# Scoring-friendly plain text
# ---------------------------------------------------------------------------
def test_markdown_to_plain_strips_bold_and_tables():
    md = "**AMI ORGANICS LIMITED**\n| A | __B__ |\n| :--- | ---: |\n| 1 | 2,3 |"
    assert markdown_to_plain(md) == "AMI ORGANICS LIMITED\nA B\n1 2,3"


def test_markdown_to_plain_keeps_lone_asterisks():
    assert markdown_to_plain("Note * 5 ** 2") == "Note * 5 ** 2"


# ---------------------------------------------------------------------------
# DeepSeek request shape
# ---------------------------------------------------------------------------
class _FakeCompletions:
    def __init__(self, sink):
        self.sink = sink

    def create(self, **kwargs):
        self.sink.update(kwargs)
        msg = SimpleNamespace(content="hello world")
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=2)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=usage)


@pytest.fixture
def deepseek_call(monkeypatch):
    import openai
    import deepseek_engine
    sent: dict = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            self.chat = SimpleNamespace(completions=_FakeCompletions(sent))

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(deepseek_engine, "load_env", lambda: None)

    def run():
        res = deepseek_engine.extract(PNG_1PX, "p.png")
        assert not res.get("error"), res.get("error")
        return sent
    return run


def test_deepseek_disables_thinking_by_default(monkeypatch, deepseek_call):
    monkeypatch.delenv("DEEPSEEK_THINKING", raising=False)
    sent = deepseek_call()
    assert sent["extra_body"] == {"thinking": {"type": "disabled"}}


def test_deepseek_thinking_can_be_enabled(monkeypatch, deepseek_call):
    monkeypatch.setenv("DEEPSEEK_THINKING", "enabled")
    sent = deepseek_call()
    assert sent["extra_body"] == {"thinking": {"type": "enabled"}}
