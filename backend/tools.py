"""Tools available to the agent.

The example ``web_search`` tool uses Tavily when ``TAVILY_API_KEY`` is set and
``langchain-tavily`` is installed; otherwise it returns deterministic mock
results so the template still runs offline. Add your own tools here.

================================================================================
backend/tools.py - 工具执行与进度事件边界
================================================================================

【阅读地图】
    上游：agent.py 的 AGENT_TOOLS、graph.py 的 sub_researcher 注册/调用 web_search。
    下游：工具结果返回给 LLM；_emit 通过 LangGraph custom stream 进入 agent/workflow 的 SSE progress。
    核心契约：query 是自然语言输入；web_search 始终返回非空字符串；没有凭据或网络失败时
    必须降级到稳定 mock 文本，避免工具可选依赖让主流程无法运行。

【阅读提示】
    先看 _emit 的 streaming 边界，再看 web_search 的真实搜索/离线 fallback 两条分支，
    最后看 TOOLS：新增工具若需要审批，必须同时纳入 agent.py 的 HITL 映射。
"""

from __future__ import annotations

import logging
import os

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _emit(message: str) -> None:
    """把工具进度写入 LangGraph 的 custom stream。

    Reads: message；Side effect: 发送 {type: progress, message} 给 streaming consumer。
    非 streaming 调用没有 writer 时静默 no-op，因此它不是业务结果或错误信号。
    """
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"type": "progress", "message": message})
    except Exception:  # pragma: no cover - not inside a streaming run
        pass


@tool
def web_search(query: str) -> str:
    """Search the web for current information about a topic.

    输入：聚焦的自然语言 query。输出：供 LLM 消费的格式化字符串，每条结果包含标题、摘要和 URL。
    流程：先发 Searching progress；有 Tavily 凭据时尝试真实搜索；网络/依赖/空结果统一降级，
    再发 offline progress 并返回确定性样例。离线分支保证零配置测试和模板演示仍可运行。

    Args:
        query: A focused natural-language search query.

    Returns:
        A formatted string of search results (title, snippet, url).
    """
    preview = query if len(query) <= 60 else query[:57] + "…"
    _emit(f"🔎 Searching the web for “{preview}”…")

    # 真实搜索分支：外部 I/O 仅在凭据存在时启用；异常和空结果继续走确定性 fallback。
    if os.getenv("TAVILY_API_KEY"):
        try:
            from langchain_tavily import TavilySearch

            response = TavilySearch(max_results=5).invoke({"query": query})
            results = response.get("results", []) if isinstance(response, dict) else []
            if results:
                _emit(f"📄 Found {len(results)} source{'s' if len(results) != 1 else ''}")
                return "\n\n".join(
                    f"- {item.get('title', 'Result')}\n  {item.get('content', '')}\n  {item.get('url', '')}"
                    for item in results
                )
        except Exception as exc:  # pragma: no cover - network/credential issues
            logger.warning("Tavily search failed (%s); using mock results.", exc)

    # 离线契约：返回非空、稳定、可供后续模型继续综合的文本；它不代表真实检索结果。
    _emit("📄 Using offline sample results (set TAVILY_API_KEY for live search)")
    return (
        f"[mock search results for '{query}']\n"
        "- Overview: a concise, relevant summary of the topic.\n"
        "- Key points: the most important facts a reader should know.\n"
        "- Note: install 'langchain-tavily' and set TAVILY_API_KEY for live web search."
    )


# 这是 agent/graph 的工具注册边界；工具名称会出现在 HITL action_requests 中，变更需同步审批与评测契约。
TOOLS = [web_search]
