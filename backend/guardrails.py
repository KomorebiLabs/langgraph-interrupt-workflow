"""Guardrail middleware — safety controls that stack alongside HITL.

The agent engine (``agent.py``) composes middleware. ``HumanInTheLoopMiddleware``
gates *tool calls*; this module adds a *content* guardrail that runs on every
model call:

- **PII redaction** — emails, phone numbers, credit-card-like and SSN-like
  numbers in the conversation are masked *before the model sees them*, without
  mutating the persisted state. This demonstrates ``wrap_model_call``, which
  wraps the request/response around the model.
- **Input blocklist** — if a configured phrase appears in the latest user
  message, the agent short-circuits with a safe refusal instead of calling the
  model at all.

Everything is configurable via environment variables and is a no-op unless
enabled, so the zero-config demo is unaffected:

    GUARDRAILS_ENABLED=true          # turn the middleware on (default: on)
    GUARDRAILS_REDACT_PII=true       # mask PII before the model sees it
    GUARDRAILS_BLOCKLIST=hack,exploit  # comma-separated blocked phrases

This is intentionally dependency-free (stdlib ``re`` only) so it always runs.
Swap the regexes / blocklist for your own policy, or plug in a dedicated
PII/detoxification service inside ``_redact``.
"""

# 【阅读地图】
#   层级：模型调用外围的内容安全 middleware。上游是 agent.py 的装配；下游是 model handler。
#   顺序：先检查最新 HumanMessage 的 blocklist，再对副本做 PII 脱敏，最后调用 handler。
#   关键不变量：blocked 直接返回 AIMessage，不触发模型；redacted 只替换 request，
#   持久化的原始 messages 不被修改；同步/异步入口必须保持同一策略。
#
# 【对象清单】
#   _PII_PATTERNS：示例正则与占位符；_truthy/_redact/_emit：配置、脱敏、事件辅助。
#   GuardrailMiddleware：from_env 配置工厂、_blocked_phrase 策略、wrap/awrap 模型边界。

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.config import get_stream_writer

from langchain.agents.middleware import AgentMiddleware

logger = logging.getLogger(__name__)


# --- PII patterns (illustrative — extend for your domain) -------------------
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("[EMAIL]", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("[CARD]", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("[SSN]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("[PHONE]", re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")),
]


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _redact(text: str) -> tuple[str, int]:
    """返回脱敏文本及替换次数；占位符是发送给模型的可见值。

    这是纯函数式的字符串变换：调用方用返回文本构造 request 副本，
    因而 checkpoint/state 中的原始消息仍保留给审计或恢复流程。
    """
    count = 0
    for placeholder, pattern in _PII_PATTERNS:
        text, n = pattern.subn(placeholder, text)
        count += n
    return text, count


def _emit(payload: dict) -> None:
    """Best-effort custom stream event (no-op outside a streaming run)."""
    try:
        get_stream_writer()(payload)
    except Exception:  # pragma: no cover - not in a streaming context
        pass


class GuardrailMiddleware(AgentMiddleware):
    """在每次模型调用边界执行 blocklist 与 PII 策略。

    ``redact_pii`` 控制是否替换邮箱/卡号/SSN/电话；``blocklist`` 已在构造时
    归一化为小写短语。两项策略都通过 custom stream event 暴露给前端，但
    _emit 失败不会影响模型调用。
    """

    def __init__(
        self,
        *,
        redact_pii: bool = True,
        blocklist: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.redact_pii = redact_pii
        self.blocklist = [b.lower() for b in (blocklist or []) if b.strip()]

    @classmethod
    def from_env(cls) -> "GuardrailMiddleware | None":
        """Build from environment, or ``None`` when guardrails are disabled."""
        if not _truthy(os.getenv("GUARDRAILS_ENABLED"), default=True):
            return None
        blocklist = [
            b.strip() for b in os.getenv("GUARDRAILS_BLOCKLIST", "").split(",") if b.strip()
        ]
        return cls(
            redact_pii=_truthy(os.getenv("GUARDRAILS_REDACT_PII"), default=True),
            blocklist=blocklist,
        )

    def _blocked_phrase(self, messages: list[BaseMessage]) -> str | None:
        """只检查倒序遍历中遇到的最新 HumanMessage，并返回首个命中短语。"""
        if not self.blocklist:
            return None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                text = str(msg.content).lower()
                return next((b for b in self.blocklist if b in text), None)
        return None

    def wrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """同步模型边界：block → redact request 副本 → handler。

        blocked 分支的 AIMessage 是短路结果；未 blocked 时 handler 的返回值原样透传。
        """
        messages: list[BaseMessage] = list(request.messages)

        # 1) Blocklist — refuse without calling the model.
        blocked = self._blocked_phrase(messages)
        if blocked:
            logger.warning("Guardrail blocked input containing %r", blocked)
            _emit({"type": "guardrail", "action": "blocked", "phrase": blocked})
            return AIMessage(
                content=(
                    "I can't help with that request. It was blocked by a safety "
                    "guardrail. Please rephrase or ask something else."
                )
            )

        # 2) PII redaction — mask before the model sees it (state is untouched).
        if self.redact_pii:
            redacted_count = 0
            new_messages: list[BaseMessage] = []
            for msg in messages:
                if isinstance(msg.content, str):
                    cleaned, n = _redact(msg.content)
                    if n:
                        redacted_count += n
                        msg = msg.model_copy(update={"content": cleaned})
                new_messages.append(msg)
            if redacted_count:
                logger.info("Guardrail redacted %d PII span(s)", redacted_count)
                _emit(
                    {
                        "type": "guardrail",
                        "action": "redacted",
                        "count": redacted_count,
                    }
                )
                request = request.override(messages=new_messages)

        return handler(request)

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """异步版本，保持同步入口的 block → redact 顺序与返回契约。"""
        messages: list[BaseMessage] = list(request.messages)

        blocked = self._blocked_phrase(messages)
        if blocked:
            logger.warning("Guardrail blocked input containing %r", blocked)
            _emit({"type": "guardrail", "action": "blocked", "phrase": blocked})
            return AIMessage(
                content=(
                    "I can't help with that request. It was blocked by a safety "
                    "guardrail. Please rephrase or ask something else."
                )
            )

        if self.redact_pii:
            redacted_count = 0
            new_messages: list[BaseMessage] = []
            for msg in messages:
                if isinstance(msg.content, str):
                    cleaned, n = _redact(msg.content)
                    if n:
                        redacted_count += n
                        msg = msg.model_copy(update={"content": cleaned})
                new_messages.append(msg)
            if redacted_count:
                logger.info("Guardrail redacted %d PII span(s)", redacted_count)
                _emit(
                    {"type": "guardrail", "action": "redacted", "count": redacted_count}
                )
                request = request.override(messages=new_messages)

        return await handler(request)
