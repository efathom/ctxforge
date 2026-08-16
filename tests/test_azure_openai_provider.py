"""
Contract-style tests for Azure OpenAI providers.

These tests do NOT hit the network. They patch the provider's client.
"""

import json

import pytest

from ctxforge.llm.azure_openai_provider import (
    AzureOpenAIConfig,
    AzureOpenAIEmbeddingProvider,
    AzureOpenAILLMProvider,
)
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
        self.usage = _FakeUsage(prompt_tokens=3, completion_tokens=4, total_tokens=7)
        self._content = content
        self._tool_calls = tool_calls

    def model_dump(self):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
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
    def __init__(self, fail_once: bool = False, response=None):
        self._fail_once = fail_once
        self.response = response or _FakeChatResponse("hello from azure fake")
        self.calls = 0
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self._fail_once and self.calls == 1:
            # Simulate a transient 500 from the gateway (name-based retry in provider).
            class InternalServerError(Exception):
                pass

            raise InternalServerError("Error code: 500 - internal server error")
        return self.response


class _FakeChat:
    def __init__(self, fail_once: bool = False, response=None):
        self.completions = _FakeChatCompletions(fail_once=fail_once, response=response)


class _FakeEmbeddings:
    def __init__(self, fail_once: bool = False):
        self._fail_once = fail_once
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self._fail_once and self.calls == 1:

            class InternalServerError(Exception):
                pass

            raise InternalServerError("Error code: 500 - internal server error")
        inputs = kwargs.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        embeddings = [[float(i), 0.0, 0.0] for i in range(len(inputs))]
        return _FakeEmbeddingResponse(embeddings)


class _FakeClient:
    def __init__(self, fail_once: bool = False, response=None):
        self.chat = _FakeChat(fail_once=fail_once, response=response)
        self.embeddings = _FakeEmbeddings(fail_once=fail_once)


def _azure_config(**overrides):
    defaults = dict(
        api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com",
        api_version="2024-02-15-preview",
        deployment="gpt-4o-mini",
        embedding_deployment="text-embedding-3-small",
    )
    defaults.update(overrides)
    return AzureOpenAIConfig(**defaults)


def _provider_with_client(monkeypatch, client, **overrides):
    provider = AzureOpenAILLMProvider(_azure_config(**overrides))

    async def _fake_get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", _fake_get_client)
    return provider


@pytest.mark.asyncio
async def test_azure_openai_llm_provider_chat_contract(monkeypatch):
    client = _FakeClient()
    provider = _provider_with_client(monkeypatch, client)

    resp = await provider.chat([ChatMessage(role="user", content="hi")])
    assert resp.content == "hello from azure fake"
    assert resp.model == "gpt-4o-mini"
    assert resp.total_tokens == 7
    assert resp.input_tokens == 3
    assert resp.output_tokens == 4
    kwargs = client.chat.completions.last_kwargs
    assert "functions" not in kwargs
    assert "function_call" not in kwargs
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


@pytest.mark.asyncio
async def test_azure_openai_llm_provider_retries_transient_500(monkeypatch):
    fake_client = _FakeClient(fail_once=True)
    provider = _provider_with_client(monkeypatch, fake_client, max_retries=2)

    resp = await provider.chat([ChatMessage(role="user", content="hi")])
    assert resp.content == "hello from azure fake"
    assert fake_client.chat.completions.calls == 2


@pytest.mark.asyncio
async def test_azure_openai_llm_provider_sends_tools_not_functions(monkeypatch):
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
async def test_azure_openai_assistant_message_serializes_tool_calls(monkeypatch):
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
async def test_azure_openai_function_role_serializes_to_tool(monkeypatch):
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
async def test_azure_openai_raw_response_populated_with_tool_calls(monkeypatch):
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
async def test_azure_openai_embedding_provider_contract(monkeypatch):
    provider = AzureOpenAIEmbeddingProvider(_azure_config(embedding_deployment="embed-deploy"))

    async def _fake_get_client():
        return _FakeClient()

    monkeypatch.setattr(provider, "_get_client", _fake_get_client)

    resp = await provider.embed(["a", "b"])
    assert resp.model == "embed-deploy"
    assert len(resp.embeddings) == 2
    assert len(resp.embeddings[0]) == 3

    single = await provider.embed_single("x")
    assert len(single) == 3


@pytest.mark.asyncio
async def test_azure_openai_embedding_provider_retries_transient_500(monkeypatch):
    fake_client = _FakeClient(fail_once=True)
    provider = AzureOpenAIEmbeddingProvider(
        _azure_config(embedding_deployment="embed-deploy", max_retries=2)
    )

    async def _fake_get_client():
        return fake_client

    monkeypatch.setattr(provider, "_get_client", _fake_get_client)

    resp = await provider.embed(["a", "b"])
    assert resp.model == "embed-deploy"
    assert fake_client.embeddings.calls == 2
