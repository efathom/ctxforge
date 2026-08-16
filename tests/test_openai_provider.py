"""
Contract-style tests for OpenAI providers.

These tests do NOT hit the network. They patch the provider's client.
"""

import json

import pytest

from ctxforge.llm.openai_provider import OpenAIConfig, OpenAIEmbeddingProvider, OpenAILLMProvider
from ctxforge.protocols.llm import ChatMessage


class _FakeUsage:
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeChatResponse:
    class _Choice:
        class _Message:
            def __init__(self, content: str, tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls

        def __init__(self, content: str, tool_calls=None):
            self.message = self._Message(content, tool_calls)
            self.finish_reason = "tool_calls" if tool_calls else "stop"

    def __init__(self, content: str, tool_calls=None):
        self.choices = [self._Choice(content, tool_calls)]
        self.usage = _FakeUsage(prompt_tokens=5, completion_tokens=7, total_tokens=12)
        self._content = content
        self._tool_calls = tool_calls

    def model_dump(self):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": self._content,
                        "tool_calls": self._tool_calls,
                    },
                    "finish_reason": self.choices[0].finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }


class _FakeEmbeddingItem:
    def __init__(self, index: int, embedding):
        self.index = index
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [_FakeEmbeddingItem(i, e) for i, e in enumerate(embeddings)]
        self.usage = _FakeUsage(total_tokens=9)


class _FakeChatCompletions:
    def __init__(self, response=None):
        self.response = response or _FakeChatResponse("hello from fake")
        self.calls = 0
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return self.response


class _FakeChat:
    def __init__(self, response=None):
        self.completions = _FakeChatCompletions(response=response)


class _FakeEmbeddings:
    async def create(self, **kwargs):
        inputs = kwargs.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        # embed with simple deterministic vectors
        embeddings = [[float(i), 0.0, 0.0] for i in range(len(inputs))]
        return _FakeEmbeddingResponse(embeddings)


class _FakeClient:
    def __init__(self, response=None):
        self.chat = _FakeChat(response=response)
        self.embeddings = _FakeEmbeddings()


def _provider_with_client(monkeypatch, client):
    cfg = OpenAIConfig(api_key="sk-test", model="gpt-4", embedding_model="text-embedding-3-small")
    provider = OpenAILLMProvider(cfg)

    async def _fake_get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", _fake_get_client)
    return provider


@pytest.mark.asyncio
async def test_openai_llm_provider_chat_contract(monkeypatch):
    client = _FakeClient()
    provider = _provider_with_client(monkeypatch, client)

    resp = await provider.chat([ChatMessage(role="user", content="hi")])
    assert resp.content == "hello from fake"
    assert resp.model == "gpt-4"
    assert resp.total_tokens == 12
    assert resp.input_tokens == 5
    assert resp.output_tokens == 7
    kwargs = client.chat.completions.last_kwargs
    assert "functions" not in kwargs
    assert "function_call" not in kwargs
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


@pytest.mark.asyncio
async def test_openai_llm_provider_sends_tools_not_functions(monkeypatch):
    client = _FakeClient()
    provider = _provider_with_client(monkeypatch, client)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo text back",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
            },
        }
    ]
    await provider.chat([ChatMessage(role="user", content="hi")], functions=tools)

    kwargs = client.chat.completions.last_kwargs
    assert "functions" not in kwargs
    assert "function_call" not in kwargs
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_openai_llm_provider_normalizes_flat_functions_to_tools(monkeypatch):
    client = _FakeClient()
    provider = _provider_with_client(monkeypatch, client)

    flat = {"name": "echo", "description": "Echo text back", "parameters": {"type": "object"}}
    await provider.chat([ChatMessage(role="user", content="hi")], functions=[flat])

    kwargs = client.chat.completions.last_kwargs
    assert kwargs["tools"] == [{"type": "function", "function": flat}]
    assert kwargs["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_openai_assistant_message_serializes_tool_calls(monkeypatch):
    client = _FakeClient()
    provider = _provider_with_client(monkeypatch, client)

    assistant_msg = ChatMessage(
        role="assistant",
        content="calling tools",
        function_call={
            "tool_calls": [
                {"id": "call_1", "name": "echo", "arguments": {"text": "hello"}},
                {"id": "call_2", "name": "echo", "arguments": {"text": "world"}},
            ]
        },
    )
    await provider.chat([assistant_msg])

    messages = client.chat.completions.last_kwargs["messages"]
    assert messages == [
        {
            "role": "assistant",
            "content": "calling tools",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": json.dumps({"text": "hello"})},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "echo", "arguments": json.dumps({"text": "world"})},
                },
            ],
        }
    ]


@pytest.mark.asyncio
async def test_openai_function_role_serializes_to_tool(monkeypatch):
    client = _FakeClient()
    provider = _provider_with_client(monkeypatch, client)

    result_msg = ChatMessage(
        role="function",
        content="the echo",
        name="echo",
        function_call={"tool_call_id": "call_1"},
    )
    await provider.chat([result_msg])

    messages = client.chat.completions.last_kwargs["messages"]
    assert messages == [{"role": "tool", "tool_call_id": "call_1", "content": "the echo"}]


@pytest.mark.asyncio
async def test_openai_raw_response_populated_with_tool_calls(monkeypatch):
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "echo", "arguments": '{"text": "hello"}'},
        }
    ]
    client = _FakeClient(response=_FakeChatResponse("", tool_calls=tool_calls))
    provider = _provider_with_client(monkeypatch, client)

    resp = await provider.chat([ChatMessage(role="user", content="hi")])
    assert resp.raw_response is not None
    assert resp.raw_response["choices"][0]["message"]["tool_calls"] == tool_calls


@pytest.mark.asyncio
async def test_openai_embedding_provider_contract(monkeypatch):
    cfg = OpenAIConfig(api_key="sk-test", model="gpt-4", embedding_model="text-embedding-3-small")
    provider = OpenAIEmbeddingProvider(cfg)

    async def _fake_get_client():
        return _FakeClient()

    monkeypatch.setattr(provider, "_get_client", _fake_get_client)

    resp = await provider.embed(["a", "b"])
    assert resp.model == "text-embedding-3-small"
    assert len(resp.embeddings) == 2
    assert len(resp.embeddings[0]) == 3

    single = await provider.embed_single("x")
    assert len(single) == 3
