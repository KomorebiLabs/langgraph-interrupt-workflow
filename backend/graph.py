"""Research-assistant workflow built with LangGraph's ``interrupt`` primitive.

This is the heart of the template: a multi-step graph that pauses at several
points to collect human decisions (approach, research direction, output
format), preserving and resuming state via a checkpointer. It demonstrates the
human-in-the-loop pattern end to end and is intentionally easy to adapt to your
own domain.

The graph is LLM- and provider-agnostic (see ``llm.py``) and runs with zero
configuration via a built-in mock model.

================================================================================
backend/graph.py - 可中断、可恢复的多阶段研究工作流
================================================================================

【阅读地图】
    层级：状态定义 + LangGraph 节点 + 工作流编排 + 流式适配器
    上游：main.py 在 lifespan 中调用 build_research_graph()；/start、/resume、/stream
          分别创建、恢复、流式驱动同一 thread_id 的 graph checkpoint。
    下游：节点通过 ResearchState 传递研究计划、并行结果、分析和最终答案；
          stream_research_response() 把 progress/content/state/done/error 事件交给 main.py 的 SSE。
    核心契约：interrupt() 暂停当前 thread，Command(resume=...) 从暂停点继续；
          current_step 与 state.next 是 API/UI 判断阶段和是否需要输入的信号；
          research_results 使用 reset_or_append，空列表表示清空，非空列表表示并行追加。

【系统位置】
    main.py /start → recall_memory → research_planner_interrupt ──interrupt──┐
                                                                                │ Command(resume)
    query_planner ── Command(goto=[Send(...)]) ─► sub_researcher × N ───────────┘
        → research_direction_interrupt → deep_analyzer
        → format_selection_interrupt → response_generator → persist_memory → END

【模块职责】
    节点只负责读取当前状态并返回状态更新；StateGraph 负责连接节点和恢复边界。
    LLM 节点在 build_research_graph() 统一套用重试、超时和补偿策略；API 层不应绕过
    stream_research_response() 自己解释 graph 的 stream mode。

【阅读提示】
    先看 ResearchState 和 reset_or_append，再看 query_planner 的 Send fan-out；
    然后看三个 interrupt 节点及其 resume 值，最后看 build_research_graph() 和 SSE 事件 shape。
"""

from __future__ import annotations

import logging
import os
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, RetryPolicy, Send, interrupt

from llm import get_llm, text_of
from memory import get_active_store, load_user_memory, save_user_memory
from tools import web_search

logger = logging.getLogger(__name__)

# 【对象清单】
# 配置/辅助：_retry_max_attempts、_node_timeout、resilience_config、SUBQUERY_COUNT、
# _max_subqueries、reset_or_append；它们把环境变量和并发/可靠性边界集中在编排层。
# 状态：ResearchState（主 graph 的跨节点状态）、SubResearchState（Send 分支的最小输入）。
# 节点：recall_memory、research_planner_interrupt、query_planner、sub_researcher、
# handle_cancel、research_direction_interrupt、deep_analyzer、format_selection_interrupt、
# response_generator、persist_memory。
# 可靠性/边界：_analysis_fallback、_response_fallback、build_research_graph、
# stream_research_response。阅读重点是 user_choice/research_direction/format_choice 的
# 中断决策语义，以及 state.next 才是 LangGraph 等待输入的运行时事实。


def _retry_max_attempts() -> int:
    try:
        return max(1, int(os.getenv("RETRY_MAX_ATTEMPTS", "3")))
    except ValueError:
        return 3


def _node_timeout() -> Optional[float]:
    """Per-node wall-clock timeout in seconds (None disables)."""
    raw = os.getenv("NODE_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
        return value if value > 0 else None
    except ValueError:
        return None


def resilience_config() -> dict:
    """Report the active resilience settings (for /capabilities)."""
    return {
        "retry_max_attempts": _retry_max_attempts(),
        "node_timeout_seconds": _node_timeout(),
        "compensation": True,  # error_handler fallbacks on the critical LLM nodes
    }

# How many parallel sub-questions to research for each approach.
SUBQUERY_COUNT = {"simplified": 2, "focused": 2, "continue_context": 3, "proceed": 4}


def _max_subqueries() -> Optional[int]:
    """Optional cap on parallel sub-researchers (``RESEARCH_MAX_SUBQUERIES``).

    The workflow fans out one concurrent LLM call per sub-question. On a
    rate-limited key (e.g. a free tier), set this to 1-2 to avoid bursting past
    the provider's requests-per-minute / concurrency limits. Unset = no cap.
    """
    raw = os.getenv("RESEARCH_MAX_SUBQUERIES", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
        return value if value >= 1 else None
    except ValueError:
        return None


# `research_results` 的 reducer 是并行汇聚边界：空写入代表开始新一轮，非空写入代表
# 当前 Send 分支的单条 finding。没有这个约定，并行 worker 可能互相覆盖结果。
def reset_or_append(existing: List[str], new: List[str]) -> List[str]:
    """Reducer: an empty write resets the list; otherwise append.

    This lets parallel sub-researchers accumulate findings within a single run
    (each appends its result) while a follow-up question can start clean by
    writing ``[]``.
    """
    if not new:
        return []
    return (existing or []) + new


class ResearchState(TypedDict):
    """State threaded through the research workflow."""

    # 【状态字段契约】
    # messages：对话历史；_previous_exchange 和各 LLM prompt 都会读取。
    # user_query：本轮问题的主输入；planner/analyzer/generator 都依赖它。
    # research_plan：规划阶段的人类可读摘要，主要供 API/UI 展示。
    # research_results：并行 findings；空列表表示 reset，非空列表由 reducer 追加。
    # sub_queries：query_planner 产生的并行问题列表，SSE state 用于展示/调试。
    # analysis：deep_analyzer 的综合结果；失败时由 fallback 填充降级文本。
    # final_response：最终或降级答案；/get_state 与 SSE state 对外暴露。
    # current_step：展示和导航字段；是否仍需输入的权威来源是 LangGraph 的 state.next。
    # requires_user_input / interrupt_data：兼容 UI 的状态形状，不替代 __interrupt__/state.next。
    # user_choice / research_direction / format_choice：三个决策点的 resume 值。
    # user_id / user_memory：Store 的跨 thread 身份与召回上下文，不等同于 checkpoint。
    messages: Annotated[List[AnyMessage], add_messages]
    user_query: str
    research_plan: str
    # Reducer so parallel sub-researchers (fanned out via Send) can append
    # findings concurrently; a follow-up question resets it by writing [].
    research_results: Annotated[List[str], reset_or_append]
    sub_queries: List[str]
    analysis: str
    final_response: str
    current_step: str
    requires_user_input: bool
    interrupt_data: Optional[Dict[str, Any]]
    user_choice: Optional[str]
    format_choice: Optional[str]
    research_direction: Optional[str]
    # Cross-thread long-term memory (loaded from / saved to the Store).
    user_id: Optional[str]
    user_memory: Optional[str]


class SubResearchState(TypedDict):
    """Input state for a single parallel sub-researcher (via Send).

    每个 Send 只携带一个 sub_query 及其展示序号，不复制完整 ResearchState；
    sub_researcher 返回单元素 research_results，由主状态 reducer 汇总。
    sub_index/sub_total 只用于 progress 事件，不参与研究内容判断。
    """

    user_query: str
    sub_query: str
    sub_index: int
    sub_total: int
    user_choice: str


def _previous_exchange(messages: List[AnyMessage]) -> tuple[str, str]:
    """Return the (previous_query, previous_response) pair, if any."""
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    previous_query = user_messages[-2].content if len(user_messages) > 1 else ""
    previous_response = ai_messages[-1].content if ai_messages else ""
    return text_of(previous_query), text_of(previous_response)


# --- Node 1: Research planning (interrupt) ---------------------------------
async def research_planner_interrupt(state: ResearchState) -> Dict[str, Any]:
    """Pause to let the user choose how the research should proceed.

    【输入/输出契约】
        Reads: messages、user_query；用最近一次问答判断是否 follow-up。
        Calls: interrupt(message)；恢复值写入 user_choice。
        Writes: HumanMessage、research_plan、current_step=information_gathering。
        Contract: cancel 由 query_planner 路由到 handle_cancel；其他选择决定并行规模。
    """
    logger.info("Planning research strategy")
    messages = state.get("messages", [])
    previous_query, previous_response = _previous_exchange(
        messages + [HumanMessage(content=state["user_query"])]
    )
    is_followup = bool(previous_query and previous_response)

    if is_followup:
        interrupt_msg = f"""## Follow-up Question Analysis

**Previous Question**: {previous_query}

**Current Question**: {state['user_query']}

This looks like a follow-up to our previous conversation. How would you like me to proceed?

- **proceed**: Full comprehensive research with detailed analysis
- **simplified**: Quick overview with key points
- **focused**: Targeted research on specific aspects
- **continue_context**: Build upon our previous conversation
- **cancel**: Stop the research process"""
    else:
        interrupt_msg = f"""## Research Query Analysis

I've analyzed your question: **"{state['user_query']}"**

How would you like me to approach this research?

- **proceed**: Full comprehensive research with detailed analysis
- **simplified**: Quick overview with key points
- **focused**: Targeted research on specific aspects
- **cancel**: Stop the research process"""

    user_choice = interrupt(interrupt_msg)
    logger.info("research_planner_interrupt resumed with choice=%s", user_choice)

    return {
        "messages": [HumanMessage(content=state["user_query"])],
        "research_plan": "Comprehensive research and analysis",
        "user_choice": user_choice,
        "current_step": "information_gathering",
    }


def _emit(payload: dict) -> None:
    """Emit a custom progress event to any streaming consumer (no-op otherwise)."""
    try:
        get_stream_writer()(payload)
    except Exception:  # pragma: no cover - not in a streaming context
        pass


# --- Memory: recall (start) -------------------------------------------------
async def recall_memory(state: ResearchState) -> Dict[str, Any]:
    """Load cross-thread memory for this user before planning.

    The current question is passed as the search ``query`` so a semantic store
    recalls the *most relevant* memories (and a plain store recalls recent ones).
    """
    memory = await load_user_memory(
        get_active_store(), state.get("user_id"), query=state.get("user_query")
    )
    if memory:
        logger.info("Recalled %d chars of long-term memory", len(memory))
    return {"user_memory": memory}


# --- Node 2a: Query planner (Send fan-out via Command) ----------------------
async def query_planner(
    state: ResearchState,
) -> Command[Literal["sub_researcher", "handle_cancel"]]:
    """把用户选择转换为并行子问题，并通过 Command/Send 扇出执行。

    Reads: user_query、user_choice、user_memory；cancel 不调用 LLM，直接 goto handle_cancel。
    Calls: LLM 生成子问题，并由 _max_subqueries 限制并发数量。
    Writes: sub_queries、research_plan、current_step。
    Returns: Command(goto=[Send("sub_researcher", payload), ...], update=...)。
    Invariant: Send payload 只包含 SubResearchState 所需字段，结果依赖 reducer 聚合。
    """
    user_choice = state.get("user_choice", "proceed")
    if user_choice == "cancel":
        return Command(goto="handle_cancel")

    n = SUBQUERY_COUNT.get(user_choice, 4)
    cap = _max_subqueries()
    if cap is not None:
        n = min(n, cap)
    memory = state.get("user_memory") or ""
    mem_section = f"\n\nWhat we already know about this user:\n{memory}" if memory else ""

    llm = get_llm()
    system = (
        "You are a research planner. Break the user's question into "
        f"{n} distinct, focused sub-questions that together cover it thoroughly. "
        "Return each sub-question on its own line with no numbering."
    )
    response = await llm.ainvoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=f"Question: {state['user_query']}{mem_section}"),
        ]
    )
    lines = [ln.strip(" -•\t") for ln in text_of(response.content).split("\n") if ln.strip()]
    sub_queries = lines[:n] or [state["user_query"]]
    while len(sub_queries) < min(n, 2):  # ensure at least a couple of workers
        sub_queries.append(f"{state['user_query']} (aspect {len(sub_queries) + 1})")

    _emit(
        {
            "type": "progress",
            "phase": "planning",
            "message": f"Planned {len(sub_queries)} parallel research threads",
            "total": len(sub_queries),
        }
    )

    sends = [
        Send(
            "sub_researcher",
            {
                "user_query": state["user_query"],
                "sub_query": q,
                "sub_index": i,
                "sub_total": len(sub_queries),
                "user_choice": user_choice,
            },
        )
        for i, q in enumerate(sub_queries)
    ]
    return Command(
        goto=sends,
        update={
            "sub_queries": sub_queries,
            "research_plan": f"Parallel research across {len(sub_queries)} sub-questions",
            "current_step": "information_gathering",
        },
    )


# --- Node 2b: Parallel sub-researcher (one per Send) ------------------------
async def sub_researcher(state: SubResearchState) -> Dict[str, Any]:
    """完成一个子问题的检索与摘要，并返回一个可聚合 finding。

    先 best-effort 调用 web_search；搜索失败只记录 warning，LLM 仍可基于子问题生成结果。
    返回的 research_results 必须是单元素列表，交由 reset_or_append 追加；sub_index/sub_total
    只进入 progress 事件，供 SSE 客户端展示并发进度。
    """
    sub = state["sub_query"]
    logger.info("Sub-researching: %s", sub)

    search_context = ""
    try:
        search_context = web_search.invoke({"query": sub})
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("web_search failed: %s", exc)

    llm = get_llm()
    response = await llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "You are a focused researcher. Given a sub-question and reference "
                    "material, produce one concise, substantive finding (2-3 sentences)."
                )
            ),
            HumanMessage(content=f"Sub-question: {sub}\nReference:\n{search_context}"),
        ]
    )
    finding = f"[{sub}] {text_of(response.content).strip()}"

    _emit(
        {
            "type": "progress",
            "phase": "researching",
            "message": f"Researched: {sub}",
            "index": state.get("sub_index", 0) + 1,
            "total": state.get("sub_total", 1),
        }
    )
    return {"research_results": [finding]}


# --- Cancel branch ----------------------------------------------------------
async def handle_cancel(state: ResearchState) -> Dict[str, Any]:
    """Short-circuit when the user cancels at the planning interrupt."""
    message = "Research was cancelled at your request."
    return {
        "messages": [AIMessage(content=message)],
        "research_results": ["Research cancelled by user request"],
        "final_response": message,
        "current_step": "completed",
        "requires_user_input": False,
    }


# --- Node 3: Research direction (conditional interrupt) ---------------------
async def research_direction_interrupt(state: ResearchState) -> Dict[str, Any]:
    """为 comprehensive 研究补充方向选择；其他模式跳过人工中断。

    proceed/comprehensive 才调用 interrupt；恢复值写入 research_direction。
    simplified/focused 等模式根据是否已有上下文自动选择 continue 或 continue_context。
    所有路径都把 current_step 推进到 analysis，下游只依赖 direction，不需判断是否真的暂停过。
    """
    logger.info("Checking research direction")
    user_choice = state.get("user_choice", "proceed")
    messages = state.get("messages", [])
    has_context = len(messages) > 1

    if user_choice in ("proceed", "comprehensive"):
        direction_msg = """## Research Direction Refinement

I've gathered substantial information. Would you like me to explore a specific angle further?

- **technical**: Deep dive into technical aspects and implementation details
- **practical**: Focus on real-world applications and use cases
- **recent**: Emphasize latest developments and current trends
- **comparative**: Compare different approaches or solutions
- **continue**: Proceed with general comprehensive analysis"""
        if has_context:
            direction_msg += "\n- **continue_context**: Build specifically on our previous conversation"

        direction_choice = interrupt(direction_msg)
        return {"research_direction": direction_choice, "current_step": "analysis"}

    # Non-comprehensive runs skip the interrupt.
    direction = "continue_context" if has_context else "continue"
    return {"research_direction": direction, "current_step": "analysis"}


# --- Node 4: Deep analysis --------------------------------------------------
async def deep_analyzer(state: ResearchState) -> Dict[str, Any]:
    """把并行 findings 综合成可供最终写作的 analysis。

    Reads: research_results、user_query、research_direction 及历史问答。
    Calls: LLM；direction=continue_context 且有历史时要求显式承接前文。
    Writes: analysis、current_step=format_selection。
    Failure contract: 重试耗尽由 _analysis_fallback 提供原始 findings 摘要，避免 API 直接 500。
    """
    logger.info("Analyzing information")
    llm = get_llm()
    research_summary = "\n".join(state.get("research_results", []))
    research_direction = state.get("research_direction", "continue")
    previous_query, previous_response = _previous_exchange(state.get("messages", []))
    has_context = bool(previous_query and previous_response)

    if research_direction == "continue_context" and has_context:
        system_prompt = (
            "You are an expert analyst building on a previous conversation. "
            "Connect the current analysis to the earlier discussion, identify "
            "relationships, and synthesize insights that show progression.\n"
            f"Previous question: {previous_query}\n"
            f"Previous response: {previous_response[:400]}..."
        )
        content = (
            f"Current query: {state['user_query']}\n\n"
            f"Current research findings:\n{research_summary}"
        )
    else:
        system_prompt = (
            "You are an expert analyst. Synthesize the findings into coherent "
            "insights, identify patterns and implications, and prepare actionable "
            "conclusions while noting any limitations."
        )
        content = (
            f"User query: {state['user_query']}\n\n"
            f"Research findings to analyze:\n{research_summary}"
        )

    response = await llm.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=content)]
    )
    return {"analysis": text_of(response.content), "current_step": "format_selection"}


# --- Node 5: Format selection (interrupt) -----------------------------------
async def format_selection_interrupt(state: ResearchState) -> Dict[str, Any]:
    """在最终生成前确定输出风格，或复用已提供的合法格式。

    user_choice 若已是 format_choices 中的值（例如 streaming 快捷路径），则不再次 interrupt；
    否则展示 analysis 预览并等待选择。所有路径最终写入 format_choice 和 response_formatting 阶段。
    """
    logger.info("Selecting response format")
    format_choices = (
        "comprehensive",
        "executive",
        "structured",
        "conversational",
        "bullet_points",
    )

    # If a format was already supplied (e.g. via the streaming endpoint), skip.
    user_choice = state.get("user_choice", "")
    if user_choice in format_choices:
        return {"format_choice": user_choice, "current_step": "response_formatting"}

    analysis = state.get("analysis", "")
    preview = analysis[:300] + ("..." if len(analysis) > 300 else "")
    format_msg = f"""## Research Complete — Choose Response Format

Here's a preview of the analysis:

**{preview}**

How would you like the final response presented?

- **comprehensive**: Thorough, detailed response with examples
- **executive**: Concise executive summary with key recommendations
- **structured**: Clear headings and bullet points
- **conversational**: Natural, professional tone
- **bullet_points**: Quick-reference lists and takeaways"""

    format_choice = interrupt(format_msg)
    return {"format_choice": format_choice, "current_step": "response_formatting"}


# --- Node 6: Response generation --------------------------------------------
async def response_generator(state: ResearchState) -> Dict[str, Any]:
    """Produce the final formatted response (streamed by the API)."""
    logger.info("Crafting final response")
    llm = get_llm()
    format_choice = state.get("format_choice", "comprehensive")
    research_direction = state.get("research_direction", "continue")
    previous_query, previous_response = _previous_exchange(state.get("messages", []))
    has_context = bool(previous_query and previous_response)

    format_instructions = {
        "comprehensive": "Create a thorough, detailed response with examples and clear headings.",
        "executive": "Create a concise executive summary with the most critical insights and recommendations.",
        "structured": "Format with clear sections, headings, and organized bullet points.",
        "conversational": "Write in a natural, conversational tone while staying professional.",
        "bullet_points": "Organize primarily as bullet points, lists, and key takeaways.",
    }
    style = format_instructions.get(format_choice, format_instructions["comprehensive"])

    if has_context:
        system_prompt = (
            "You are writing a follow-up response that builds on a previous "
            "conversation.\n"
            f"Previous question: {previous_query}\n"
            f"Previous response: {previous_response[:300]}...\n\n"
            f"Formatting style: {style}\n"
            "Reference the prior exchange naturally and maintain continuity."
        )
    else:
        system_prompt = (
            "You are writing the final response for the user.\n"
            f"Formatting style: {style}\n"
            "Be accurate, actionable, and directly address the question."
        )

    memory = state.get("user_memory") or ""
    if memory:
        system_prompt += f"\n\nRemembered context about this user:\n{memory}"

    context = (
        f"Original question: {state['user_query']}\n"
        f"Research findings: {'; '.join(state.get('research_results', []))}\n"
        f"Analysis insights: {state.get('analysis', 'N/A')}"
    )

    response = await llm.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=context)]
    )
    final_response = text_of(response.content)

    return {
        "messages": [AIMessage(content=final_response)],
        "final_response": final_response,
        "current_step": "completed",
        "requires_user_input": False,
    }


# --- Memory: persist (end) --------------------------------------------------
async def persist_memory(state: ResearchState) -> Dict[str, Any]:
    """把本轮问题及用户偏好的输出格式写入跨 thread Store。

    这是结束阶段的副作用节点；正常答案和 fallback 答案都会经过这里。它不修改 graph 结果，
    返回空 update，使 checkpoint 能沿 response_generator 后继续到 END。
    """
    note = f"Asked about: {state.get('user_query', '')[:120]}"
    if state.get("format_choice"):
        note += f" (preferred format: {state['format_choice']})"
    await save_user_memory(get_active_store(), state.get("user_id"), note)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 阶段：重试耗尽后的补偿（compensation）
# retry 只处理瞬态失败；达到上限后由 error_handler 把 graph 导向可交付的降级结果，
# 使分析失败仍能进入格式选择、最终生成失败仍能持久化并结束，而不是把内部异常暴露给 API。
# ─────────────────────────────────────────────────────────────────────────────
# --- Compensation handlers (run after a node's retries are exhausted) -------
async def _analysis_fallback(state: ResearchState) -> Command:
    """Saga-style compensation: if analysis fails, degrade to the raw findings."""
    logger.warning("deep_analyzer failed after retries — using fallback analysis")
    summary = "\n".join(state.get("research_results", [])) or "No findings were gathered."
    return Command(
        goto="format_selection_interrupt",
        update={
            "analysis": (
                "(Automatic fallback) The analysis step could not be completed, so "
                "here is a summary of the raw findings:\n" + summary
            ),
            "current_step": "format_selection",
        },
    )


async def _response_fallback(state: ResearchState) -> Command:
    """Compensation: if final generation fails, return a graceful degraded answer."""
    logger.warning("response_generator failed after retries — using fallback response")
    text = (
        state.get("analysis")
        or "; ".join(state.get("research_results", []))
        or "I wasn't able to complete the research this time. Please try again."
    )
    return Command(
        goto="persist_memory",
        update={
            "messages": [AIMessage(content=text)],
            "final_response": text,
            "current_step": "completed",
            "requires_user_input": False,
        },
    )


def build_research_graph(checkpointer: Any | None = None, store: Any | None = None):
    """Build and compile the research workflow.

    LLM-backed nodes are hardened with LangGraph 1.2 resilience primitives:
    a ``retry_policy`` (retries transient errors with backoff), an optional
    per-node ``timeout``, and ``error_handler`` compensation on the critical
    analysis / response nodes so a failure degrades gracefully instead of 500ing.

    Args:
        checkpointer: A LangGraph checkpointer for durable, resumable state.
            When ``None`` (e.g. LangGraph Studio / langgraph dev), the runtime
            supplies its own persistence layer.
        store: A LangGraph ``BaseStore`` for cross-thread long-term memory.
            When ``None``, the memory nodes degrade to no-ops.
    """
    retry = RetryPolicy(max_attempts=_retry_max_attempts())
    timeout = _node_timeout()
    # Common kwargs for LLM-backed nodes.
    llm_node = {"retry_policy": retry}
    if timeout is not None:
        llm_node["timeout"] = timeout

    builder = StateGraph(ResearchState)
    builder.add_node("recall_memory", recall_memory)
    builder.add_node("research_planner_interrupt", research_planner_interrupt)
    builder.add_node("query_planner", query_planner, **llm_node)
    builder.add_node("sub_researcher", sub_researcher, **llm_node)
    builder.add_node("handle_cancel", handle_cancel)
    builder.add_node("research_direction_interrupt", research_direction_interrupt)
    builder.add_node(
        "deep_analyzer",
        deep_analyzer,
        error_handler=_analysis_fallback,
        destinations=("format_selection_interrupt",),
        **llm_node,
    )
    builder.add_node("format_selection_interrupt", format_selection_interrupt)
    builder.add_node(
        "response_generator",
        response_generator,
        error_handler=_response_fallback,
        destinations=("persist_memory",),
        **llm_node,
    )
    builder.add_node("persist_memory", persist_memory)

    builder.add_edge(START, "recall_memory")
    builder.add_edge("recall_memory", "research_planner_interrupt")
    builder.add_edge("research_planner_interrupt", "query_planner")
    # query_planner fans out to parallel sub_researchers (or cancels) via Command.
    builder.add_edge("sub_researcher", "research_direction_interrupt")
    builder.add_edge("research_direction_interrupt", "deep_analyzer")
    builder.add_edge("deep_analyzer", "format_selection_interrupt")
    builder.add_edge("format_selection_interrupt", "response_generator")
    builder.add_edge("response_generator", "persist_memory")
    builder.add_edge("persist_memory", END)
    builder.add_edge("handle_cancel", END)

    return builder.compile(checkpointer=checkpointer, store=store)


# Module-level graph for `langgraph dev` / LangGraph Studio. The FastAPI app
# builds its own instance with a durable checkpointer (see main.py).
research_graph = build_research_graph(checkpointer=MemorySaver())


# Nodes whose LLM output should be streamed to the client as the final answer.
STREAMING_NODES = {"response_generator"}


async def stream_research_response(
    graph, thread_id: str, user_choice: str, config: Optional[dict] = None
):
    """恢复一个 thread，并把 graph 执行转换为客户端可消费的事件流。

    【事件契约】
        progress：并行子研究进度（custom stream）；content：response_generator 的 token；
        state：本次运行后的 requires_input、interrupt_message、current_step、结果摘要和 next；
        done：正常结束哨兵；error：异常的终止事件，done=True。

    state.next 是是否仍需用户输入的权威来源；config 可带 checkpoint_id，用于历史 checkpoint 的 fork。
    任意未处理异常在此包装为 error 事件，避免生成器直接中断 SSE。
    """
    logger.info("Streaming research for thread=%s choice=%s", thread_id, user_choice)
    config = config or {"configurable": {"thread_id": thread_id}}

    try:
        interrupt_message = None
        async for mode, data in graph.astream(
            Command(resume=user_choice),
            config=config,
            stream_mode=["custom", "messages", "updates"],
        ):
            if mode == "custom":
                event = data if isinstance(data, dict) else {"message": str(data)}
                event.setdefault("type", "progress")
                yield event
            elif mode == "messages":
                chunk, meta = data
                token = text_of(getattr(chunk, "content", None))
                if meta.get("langgraph_node") in STREAMING_NODES and token:
                    yield {
                        "type": "content",
                        "content": token,
                        "done": False,
                        "node": meta.get("langgraph_node"),
                    }
            elif mode == "updates":
                if isinstance(data, dict) and "__interrupt__" in data:
                    interrupt_message = data["__interrupt__"][0].value

        state = await graph.aget_state(config)
        values = state.values or {}
        yield {
            "type": "state",
            "requires_input": bool(state.next),
            "interrupt_message": interrupt_message,
            "current_step": values.get("current_step", "unknown"),
            "final_response": values.get("final_response", ""),
            "research_results": values.get("research_results", []),
            "sub_queries": values.get("sub_queries", []),
            "next": list(state.next),
        }
        yield {"type": "done", "content": "", "done": True}
    except Exception as exc:  # pragma: no cover - surfaced to the client
        logger.exception("Streaming error")
        yield {"type": "error", "content": f"Error in streaming: {exc}", "done": True}
