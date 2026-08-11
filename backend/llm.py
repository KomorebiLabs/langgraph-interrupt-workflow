"""Provider-agnostic LLM factory.

This template is intentionally LLM-agnostic. Pick any provider supported by
LangChain's ``init_chat_model`` (OpenAI, Anthropic, Google, Groq, Mistral,
IBM watsonx, Ollama, ...) by setting a couple of environment variables:

    LLM_MODEL=gpt-4o-mini            # any model id
    LLM_PROVIDER=openai              # optional, inferred from the model when omitted
    LLM_TEMPERATURE=0.7              # optional

If no provider credentials are configured, the template falls back to a small
built-in ``MockChatModel`` so it runs end-to-end (including token streaming)
with zero configuration. That makes the repo clone-and-run for newcomers.
"""

# 【阅读地图】
#   层级：provider-neutral LLM 适配边界。上游由 agent/graph 调用 get_llm；下游是 LangChain model。
#   选择边界：using_mock_llm 决定离线 mock；get_llm 负责真实 provider 的环境配置与构造，
#   初始化异常仍回退 mock。业务代码只应依赖 BaseChatModel/text_of，不依赖具体供应商。
#
# 【对象清单】
#   _KNOWN_PROVIDER_KEYS：自动探测凭据；text_of：跨 provider 内容归一化；
#   MockChatModel：支持工具调用与 token streaming 的离线工厂产物；
#   using_mock_llm/get_llm：mock/real provider 选择与 fallback；_has_tool_result/_last_human 为 mock 辅助。

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

logger = logging.getLogger(__name__)

# API-key environment variables we recognise for auto-detection. When none of
# these (and no explicit model) are set, we use the mock model for a zero-config
# demo experience.
_KNOWN_PROVIDER_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "COHERE_API_KEY",
    "FIREWORKS_API_KEY",
    "TOGETHER_API_KEY",
    "WATSONX_API_KEY",
)


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def text_of(content: Any) -> str:
    """将字符串或内容 block 列表统一为纯文本。

    这是上层读取最终答案、流式 token 和工具上下文的稳定入口；thinking/tool-use
    等非 text block 被跳过，避免把 provider 的内部块泄露为用户可见答案。

    Normalize a message's ``content`` to plain text.

    Newer models (e.g. Gemini 3.x, Claude with thinking) return ``content`` as a
    list of typed blocks rather than a string. This flattens either form to the
    concatenated text, so provider-agnostic code can treat every response the
    same way.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Only surface text blocks (skip thinking/tool-use/etc).
                if block.get("type", "text") == "text":
                    parts.append(str(block.get("text", "")))
            else:
                parts.append(str(getattr(block, "text", "")))
        return "".join(parts)
    return str(content)


def _has_tool_result(messages: List[BaseMessage]) -> bool:
    return any(getattr(m, "type", None) == "tool" for m in messages)


def _last_human(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", None) in ("human", "user"):
            return str(msg.content)
    return "the topic"


class MockChatModel(BaseChatModel):
    """A tiny offline chat model used when no provider is configured.

    【mock 契约】bind_tools 只记录工具名和首个参数名；首次无 tool result 时发出
    一次工具调用，收到 tool result 后输出 canned response。_stream 与 _generate
    使用同一判断，因此离线模式也能走 HITL 的 action_requests/resume 路径。

    It produces context-aware canned responses and supports streaming so the
    full workflow runs without API keys. When tools are bound (e.g. by
    ``create_agent``), it drives one tool call and then a final answer, so the
    agent engine — including human-in-the-loop tool approval — also works
    offline.
    """

    # Populated by ``bind_tools`` as a list of [tool_name, first_arg_name].
    # 这是 mock 的最小工具协议，不是对真实 provider tool schema 的替代。
    tool_specs: list = []

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "MockChatModel":
        specs: list = []
        for t in tools:
            name = getattr(t, "name", None) or (
                t.get("name") if isinstance(t, dict) else None
            )
            arg_names = list(getattr(t, "args", {}) or {})
            if name:
                specs.append([name, arg_names[0] if arg_names else "query"])
        return self.model_copy(update={"tool_specs": specs})

    def _tool_call_message(self, messages: List[BaseMessage]) -> Optional[dict]:
        """Return a single tool call to make, or None to answer directly."""
        if self.tool_specs and not _has_tool_result(messages):
            name, arg = self.tool_specs[0]
            return {"name": name, "args": {arg: _last_human(messages)}, "id": "mock_call_1"}
        return None

    @staticmethod
    def _canned_response(messages: List[BaseMessage]) -> str:
        human = ""
        for msg in reversed(messages):
            if msg.type in ("human", "user"):
                human = str(msg.content)
                break

        text = human.lower()
        if "synthesize" in text or "analyst" in text:
            return (
                "Synthesis: the findings converge on a clear picture. Key "
                "patterns, trade-offs, and a few caveats are highlighted below "
                "so you can act on them with confidence."
            )
        if "research query" in text or "research the current question" in text:
            return (
                "Finding A: a strong, well-supported result.\n"
                "Finding B: a useful nuance that refines the headline answer.\n"
                "Finding C: a practical implication worth keeping in mind."
            )
        return (
            "Here is a clear, structured answer to your question. (This is the "
            "built-in mock model — set LLM_MODEL and a provider API key to use a "
            "real LLM.)"
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_call = self._tool_call_message(messages)
        if tool_call:
            message = AIMessage(content="", tool_calls=[tool_call])
        else:
            message = AIMessage(content=self._canned_response(messages))
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, **kwargs)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        import json

        tool_call = self._tool_call_message(messages)
        if tool_call:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": tool_call["name"],
                            "args": json.dumps(tool_call["args"]),
                            "id": tool_call["id"],
                            "index": 0,
                        }
                    ],
                )
            )
            return
        for token in self._canned_response(messages).split(" "):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token + " "))
            if run_manager:
                run_manager.on_llm_new_token(token + " ", chunk=chunk)
            yield chunk


def using_mock_llm() -> bool:
    """判断 provider 选择边界：显式 USE_MOCK_LLM 优先，否则无模型且无已知 key 才 mock。

    只要配置了 LLM_MODEL 或任一已知凭据，就尝试真实 provider；真实构造失败仍由
    get_llm 的防御性 fallback 保证应用可启动。
    """
    if _truthy(os.getenv("USE_MOCK_LLM")):
        return True
    has_model = bool(os.getenv("LLM_MODEL"))
    has_key = any(os.getenv(key) for key in _KNOWN_PROVIDER_KEYS)
    return not has_model and not has_key


def get_llm(**overrides: Any) -> BaseChatModel:
    """按环境配置构造统一的 BaseChatModel。

    【配置契约】LLM_MODEL 指模型 ID，LLM_PROVIDER 可显式指定 provider，
    LLM_TEMPERATURE 提供默认 temperature，overrides 最后覆盖环境默认值。
    凭据由各 provider 自己读取；本工厂只负责选择/构造，不执行请求。初始化异常
    会记录 warning 并回到 MockChatModel，形成“真实 provider → mock”开关边界。

    Falls back to :class:`MockChatModel` when nothing is configured or when
    initialisation fails, so the template always runs.
    """
    if using_mock_llm():
        logger.info(
            "Using built-in MockChatModel (no LLM_MODEL/provider key configured). "
            "Set LLM_MODEL and a provider API key for real responses."
        )
        return MockChatModel()

    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    provider = os.getenv("LLM_PROVIDER") or None
    params: dict[str, Any] = {"temperature": float(os.getenv("LLM_TEMPERATURE", "0.7"))}
    params.update(overrides)

    try:
        return init_chat_model(model, model_provider=provider, **params)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning(
            "Failed to initialise model '%s' (%s); falling back to MockChatModel.",
            model,
            exc,
        )
        return MockChatModel()
