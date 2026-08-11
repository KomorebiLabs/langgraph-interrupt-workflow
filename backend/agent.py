"""Agent engine — the modern, model-driven counterpart to ``graph.py``.

This is the second selectable backend "engine". Where ``graph.py`` is an
explicit, deterministic pipeline (fixed interrupt points + ``Send`` fan-out),
this is a genuinely *agentic* research assistant built with ``create_agent``
(LangChain v1, which replaces the deprecated ``create_react_agent``):

- the **model drives the loop** — it decides when to call ``web_search``;
- ``HumanInTheLoopMiddleware`` pauses for **approve / edit / reject / respond**
  before a tool runs (the same interrupt/resume mechanism, with no boilerplate);
- it shares the same provider-agnostic LLM, ``web_search`` tool, and ``Store``
  long-term memory used by the workflow engine.

Toggle between the two engines live in the UI to compare the paradigms.

CLI demo: ``python agent.py "What are the latest advances in battery tech?"``

================================================================================
backend/agent.py - 模型驱动的工具 Agent 与 HITL 流式适配器
================================================================================

【阅读地图】
    上游：main.py 在 lifespan 中调用 build_agent()，前端通过 /agent/start 与 /agent/decide 驱动。
    下游：create_agent 维护模型循环；middleware 负责安全与审批；stream_agent_response 将
          updates/messages/custom 转成前端的 progress/content/state/done/error 事件。
    核心契约：模型决定何时请求工具，但所有已注册工具都必须先经过 HumanInTheLoopMiddleware；
          Command(resume={"decisions": [...]}) 继续同一 checkpointer thread。

【控制流】
    messages → model loop → tool action request → HITL interrupt
             → approve/edit/reject/respond → tool/result → model loop → final state

【阅读提示】
    先看 build_agent 的 middleware 顺序和工具审批映射，再看 _pending_interrupt、
    stream_agent_response 的事件投影，最后看结构化输出与 CLI demo。
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from pydantic import BaseModel, Field

from guardrails import GuardrailMiddleware
from llm import get_llm, text_of
from middleware_pack import build_middleware_pack
from tools import web_search

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful research assistant. Use the web_search tool to gather "
    "current information before answering. Cite what you find and be concise. "
    "If the user has remembered context, take it into account."
)

# Tools the agent may call. web_search is gated by human approval below.
AGENT_TOOLS = [web_search]


class ResearchSummary(BaseModel):
    """Structured result the agent can return when structured output is enabled.

    Opt in with ``AGENT_STRUCTURED_OUTPUT=true`` and a real (non-mock) model.
    ``create_agent`` then places a validated instance in
    ``state["structured_response"]``.

    字段契约：summary 是短答案；key_findings 是要点列表；sources 是引用；confidence
    是 low/medium/high 的粗粒度信号。该对象只在启用 response_format 时出现在 state，
    普通文本模式仍以 messages 的最后一条消息作为最终答案。
    """

    summary: str = Field(description="A concise 2-3 sentence answer to the question.")
    key_findings: List[str] = Field(
        default_factory=list, description="The most important findings, as bullet points."
    )
    sources: List[str] = Field(
        default_factory=list, description="URLs or titles the answer draws on, if any."
    )
    confidence: str = Field(
        default="medium", description="Rough confidence: low | medium | high."
    )


def structured_output_enabled() -> bool:
    """Whether AGENT_STRUCTURED_OUTPUT opts the agent into structured output."""
    return os.getenv("AGENT_STRUCTURED_OUTPUT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# Backwards-compatible private alias.
_structured_output_enabled = structured_output_enabled


def agent_middleware_summary() -> list[str]:
    """Names of the middleware active on the agent, for /capabilities reporting."""
    names: list[str] = []
    if GuardrailMiddleware.from_env() is not None:
        names.append("guardrails")
    names.extend(build_middleware_pack(get_llm())[1])
    names.append("human_in_the_loop")
    return names


def build_agent(
    checkpointer: Any | None = None,
    store: Any | None = None,
    extra_tools: Optional[list] = None,
    structured: Optional[bool] = None,
):
    """Build a tool-using agent that requires approval before sensitive tools.

    Middleware stack (order matters — outermost first):

    1. :class:`~guardrails.GuardrailMiddleware` — redacts PII / blocks input
       around every model call (skipped when guardrails are disabled).
    2. ``HumanInTheLoopMiddleware`` — interrupts before ``web_search`` (and any
       MCP tools) run. Resume with
       ``Command(resume={"decisions": [{"type": "approve"}]})`` (or ``edit`` /
       ``reject`` / ``respond``).

    这不是固定节点 workflow：model 拥有循环控制权，但 tools 列表构成安全边界；
    extra_tools 会加入同一审批映射，因此 MCP 工具不会绕过现有 HITL。

    Args:
        checkpointer: durable, resumable per-thread state.
        store: cross-thread long-term memory (see ``memory.py``).
        extra_tools: additional tools (e.g. MCP tools) exposed to the agent and
            gated by the same human approval as ``web_search``.
        structured: force structured output on/off. When ``None``, follows the
            ``AGENT_STRUCTURED_OUTPUT`` environment variable.
    """
    tools = [*AGENT_TOOLS, *(extra_tools or [])]
    model = get_llm()

    # Gate every tool (built-in + MCP) behind human approval.
    hitl = HumanInTheLoopMiddleware(
        interrupt_on={t.name: True for t in tools},
        description_prefix="The agent wants to run a tool and needs your approval",
    )

    # Middleware order (outermost first): guardrail → prebuilt power-pack → HITL.
    middleware: list = []
    guardrail = GuardrailMiddleware.from_env()
    if guardrail is not None:
        middleware.append(guardrail)
    pack, _active = build_middleware_pack(model)
    middleware.extend(pack)
    middleware.append(hitl)

    use_structured = _structured_output_enabled() if structured is None else structured
    response_format = ResearchSummary if use_structured else None

    return create_agent(
        model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
        response_format=response_format,
        checkpointer=checkpointer,
        store=store,
    )


# Module-level instance for `langgraph dev` / LangGraph Studio.
agent = build_agent()


def _pending_interrupt(value: Any) -> dict:
    """Normalize a HITLMiddleware interrupt request for the client."""
    requests = value.get("action_requests", []) if isinstance(value, dict) else []
    configs = value.get("review_configs", [{}]) if isinstance(value, dict) else [{}]
    allowed = configs[0].get("allowed_decisions", ["approve", "reject"]) if configs else [
        "approve",
        "reject",
    ]
    return {"tool_requests": requests, "allowed": allowed}


async def stream_agent_response(graph, thread_id: str, command_input, config: Optional[dict] = None):
    """Run/resume the agent and stream progress, tokens, and a closing state.

    ``command_input`` is the initial ``{"messages": [...]}`` (to start) or a
    ``Command(resume=...)`` (to resume after an approval decision).

    【事件契约】custom → progress；messages → content token；updates 中的 interrupt
    → tool_requests/allowed；收尾 state 表示 requires_input/final_response/current_step；
    done/error 是客户端的终止哨兵。state.next 是是否仍等待人工输入的权威来源。
    """
    config = config or {"configurable": {"thread_id": thread_id}}
    interrupt_value = None
    try:
        async for mode, data in graph.astream(
            command_input, config=config, stream_mode=["updates", "messages", "custom"]
        ):
            if mode == "custom":
                # Guardrail (PII redaction / blocklist) and other progress events.
                event = data if isinstance(data, dict) else {"message": str(data)}
                event.setdefault("type", "progress")
                yield event
            elif mode == "updates" and isinstance(data, dict):
                if "__interrupt__" in data:
                    interrupt_value = data["__interrupt__"][0].value
                elif "tools" in data:
                    yield {"type": "progress", "message": "🔧 Tool executed — synthesizing…"}
            elif mode == "messages":
                chunk, _meta = data
                token = text_of(getattr(chunk, "content", None))
                if token:
                    yield {"type": "content", "content": token, "done": False}

        state = await graph.aget_state(config)
        values = state.values or {}
        messages = values.get("messages", [])
        final = ""
        if messages and not state.next:
            final = text_of(getattr(messages[-1], "content", "")) or ""

        event = {
            "type": "state",
            "requires_input": bool(state.next),
            "final_response": final,
            "current_step": "awaiting_approval" if state.next else "completed",
        }
        # Surface structured output when the agent was built with a response_format.
        structured = values.get("structured_response")
        if structured is not None and not state.next:
            event["structured_response"] = (
                structured.model_dump()
                if hasattr(structured, "model_dump")
                else structured
            )
        if interrupt_value is not None:
            event.update(_pending_interrupt(interrupt_value))
        yield event
        yield {"type": "done", "content": "", "done": True}
    except Exception as exc:  # pragma: no cover - surfaced to the client
        logger.exception("Agent streaming error")
        yield {"type": "error", "content": f"Error: {exc}", "done": True}


def _demo(question: str) -> None:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    demo_agent = build_agent(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "agent-demo"}}

    result = demo_agent.invoke(
        {"messages": [{"role": "user", "content": question}]}, config
    )

    if "__interrupt__" in result:
        request = result["__interrupt__"][0].value
        print("\n⏸  Human-in-the-loop pause — approval requested:")
        print(request)
        print("\n▶  Auto-approving for this demo...\n")
        result = demo_agent.invoke(
            Command(resume={"decisions": [{"type": "approve"}]}), config
        )

    final = result["messages"][-1]
    print("Agent:", getattr(final, "content", final))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    query = " ".join(sys.argv[1:]) or "What is LangGraph and why is it useful?"
    _demo(query)
