"""Prebuilt middleware "power-pack" for the agent engine.

LangChain v1 ships a catalog of production-ready middleware. This module wires a
curated, env-configurable subset onto the agent (composed *alongside* the custom
guardrail and human-in-the-loop middleware in ``agent.py``):

- **SummarizationMiddleware** — compresses old messages as the conversation
  grows, so long-running threads never overflow the context window.
- **ModelCallLimitMiddleware** — caps model calls per run (a runaway / cost
  guardrail).
- **ToolCallLimitMiddleware** — caps tool calls per run (opt-in).
- **ModelRetryMiddleware** — retries transient model/endpoint errors with
  exponential backoff.
- **TodoListMiddleware** — gives the agent a ``write_todos`` planning tool so it
  can decompose and track multi-step work (opt-in — adds a tool).
- **ModelFallbackMiddleware** — falls back to another model on failure (opt-in;
  needs a second model configured).

Everything has a sensible default and is controlled by environment variables, so
the zero-config demo is unaffected. Defaults are conservative enough that they
never trigger during a short conversation.
"""

# 【阅读地图】
#   层级：agent.py 使用的可选 LangChain middleware 装配器；本文件不执行模型请求。
#   输入：一个 chat model（仅供摘要 middleware）；环境变量决定各组件是否加入。
#   输出：(middleware 实例列表, active_names 报告列表)，列表顺序就是运行时包顺序。
#
# 【对象清单】
#   _truthy/_int_env：宽松解析开关与正整数限制；build_middleware_pack：唯一装配入口；
#   active_middleware_names：仅返回能力报告。组件按摘要 → 模型限制 → 工具限制 → 重试
#   → todo 规划 → fallback 排列；agent.py 再把 HITL 放在整个包之后。

from __future__ import annotations

import logging
import os
from typing import Any

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)

logger = logging.getLogger(__name__)


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _int_env(name: str, default: int | None) -> int | None:
    """读取正整数限制；空值走默认值，非正数表示关闭该限制。"""
    raw = os.getenv(name, "").strip()
    if raw == "":
        return default
    try:
        value = int(raw)
        return value if value > 0 else None
    except ValueError:
        return default


def build_middleware_pack(model: Any) -> tuple[list, list[str]]:
    """Return ``(middleware, active_names)`` for the configured power-pack.

    【顺序契约】返回列表会被 agent.py 原样插入 guardrail 与 HITL 之间；
    因而这里的先后不仅用于展示，也决定请求/响应包装层级。默认开启摘要、
    模型调用上限与重试；工具上限、TodoList、fallback 均是显式 opt-in。

    ``model`` is the chat model the summarization middleware uses to condense
    history. The returned middleware are meant to sit between the guardrail and
    the human-in-the-loop middleware in ``build_agent``.
    """
    middleware: list = []
    active: list[str] = []

    # 1) Summarization — protect against context overflow on long threads.
    #    先压缩历史，再让后续限制/重试层处理更小的上下文。
    if _truthy(os.getenv("AGENT_SUMMARIZATION"), default=True):
        trigger_messages = _int_env("AGENT_SUMMARIZATION_TRIGGER_MESSAGES", 40) or 40
        keep_messages = _int_env("AGENT_SUMMARIZATION_KEEP_MESSAGES", 20) or 20
        middleware.append(
            SummarizationMiddleware(
                model=model,
                trigger=("messages", trigger_messages),
                keep=("messages", keep_messages),
            )
        )
        active.append(f"summarization(>{trigger_messages} msgs)")

    # 2) Model-call limit — runaway / cost guardrail.
    model_call_limit = _int_env("AGENT_MODEL_CALL_LIMIT", 25)
    if model_call_limit:
        middleware.append(
            ModelCallLimitMiddleware(run_limit=model_call_limit, exit_behavior="end")
        )
        active.append(f"model_call_limit({model_call_limit})")

    # 3) Tool-call limit — opt-in cap on tool invocations per run.
    tool_call_limit = _int_env("AGENT_TOOL_CALL_LIMIT", None)
    if tool_call_limit:
        middleware.append(
            ToolCallLimitMiddleware(run_limit=tool_call_limit, exit_behavior="continue")
        )
        active.append(f"tool_call_limit({tool_call_limit})")

    # 4) Model retry — recover from transient endpoint errors.
    #    仅包裹模型调用；它不是 HITL 决策重试，也不会重放已执行工具。
    if _truthy(os.getenv("AGENT_MODEL_RETRY"), default=True):
        retries = _int_env("AGENT_MODEL_RETRIES", 2) or 2
        middleware.append(ModelRetryMiddleware(max_retries=retries))
        active.append(f"model_retry({retries})")

    # 5) TodoList planning tool — opt-in (adds a `write_todos` tool).
    if _truthy(os.getenv("AGENT_TODO_LIST"), default=False):
        from langchain.agents.middleware import TodoListMiddleware

        middleware.append(TodoListMiddleware())
        active.append("todo_list")

    # 6) Model fallback — opt-in; needs a second model id.
    #    这里只按模型 ID 构造备用层；构造失败时保持主模型路径不变。
    fallback_model = os.getenv("AGENT_FALLBACK_MODEL", "").strip()
    if fallback_model:
        try:
            from langchain.agents.middleware import ModelFallbackMiddleware

            middleware.append(ModelFallbackMiddleware(fallback_model))
            active.append(f"model_fallback({fallback_model})")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not enable model fallback: %s", exc)

    return middleware, active


def active_middleware_names(model: Any) -> list[str]:
    """仅生成 /capabilities 的名称快照，不重复暴露 middleware 内部参数。"""
    return build_middleware_pack(model)[1]
