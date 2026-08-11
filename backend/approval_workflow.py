"""Approval workflow — a second human-in-the-loop example.

A common real-world pattern: an AI drafts something (an email, a reply, a
policy snippet), then a human **approves**, **edits**, or **rejects with
feedback** before it's "sent". On reject, the draft is regenerated using the
feedback; on edit, the human's version is used verbatim.

This complements the multi-step research workflow in ``graph.py`` by showing
the three canonical HITL actions (approve / edit / reject) on a single
interrupt, mirroring what ``HumanInTheLoopMiddleware`` offers for tool calls.

================================================================================
backend/approval_workflow.py - 单次草稿审批状态机
================================================================================

【阅读地图】
    上游：main.py 编译 approval_graph；/approval/start 创建 thread，/approval/decide
          用 Command(resume=...) 注入人工决定。
    下游：human_review 的决定进入 route_after_review；finalize 产出 final_output；
          main.py 的 _approval_payload 将状态投影为 API JSON。
    核心契约：approve/edit 终结并标记 sent；reject 写 feedback、递增 revision_count，
          在 MAX_REVISIONS 内回到 drafter；超过上限仍 finalize，但状态明确未完全解决反馈。

【状态机】
    START → drafter → human_review ─approve/edit→ finalize → END
                              │
                              └─reject + 未达上限→ drafter
                                reject + 达上限 → finalize（受限成功）

【阅读提示】
    先看 ApprovalState，再看 human_review 的 resume payload，最后看
    route_after_review 与 finalize；draft 已生成不等于 draft 已发送。
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from llm import get_llm, text_of

logger = logging.getLogger(__name__)

# Safety valve so a reject → redraft loop can't run forever.
# 它既是成本保护，也是状态机的硬终止条件；达到上限后 finalize 会明确标记受限成功。
MAX_REVISIONS = 3


# 【ApprovalState 状态契约】
# messages：最终发送内容的消息历史；task：不变的起草目标。
# draft：当前待审版本，edit 时替换为用户内容；feedback：reject 后供下一次 drafter 读取。
# revision_count：reject 次数，route_after_review 用它决定回环还是终止。
# decision：approve/edit/reject 路由哨兵；status：UI 展示的 drafting/awaiting_review/sent 状态。
# final_output：只有 finalize 写入才代表对外发送结果；draft 本身不等于已发送。


class ApprovalState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    task: str
    draft: str
    feedback: str
    revision_count: int
    decision: str
    status: str
    final_output: str


async def drafter(state: ApprovalState) -> Dict[str, Any]:
    """生成首稿或根据 reviewer feedback 重写当前 draft。

    Reads: task、draft、feedback、revision_count；有 feedback 时要求 LLM 完整回应反馈。
    Writes: draft、status=awaiting_review。它不会把内容标记为 sent，发送语义只属于 finalize。
    """
    logger.info("Drafting content (revision %s)", state.get("revision_count", 0))
    llm = get_llm()
    feedback = state.get("feedback", "")

    if feedback:
        system = (
            "You are revising a draft based on reviewer feedback. Produce an "
            "improved version that fully addresses the feedback."
        )
        human = (
            f"Task: {state['task']}\n\n"
            f"Previous draft:\n{state.get('draft', '')}\n\n"
            f"Reviewer feedback:\n{feedback}\n\n"
            "Rewrite the draft to address the feedback."
        )
    else:
        system = (
            "You are a helpful assistant that drafts clear, professional content "
            "for the given task. Return only the draft."
        )
        human = f"Task: {state['task']}\n\nWrite a complete draft."

    response = await llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=human)]
    )
    return {"draft": text_of(response.content), "status": "awaiting_review"}


async def human_review(state: ApprovalState) -> Dict[str, Any]:
    """暂停并接受 approve、edit、reject 三种人工决定。

    interrupt payload 是 main.py/UI 的输入契约：type=approval、task、draft、revision_count、actions、message。
    resume 可为裸字符串，也可为 dict；edit 取 content，reject 取 feedback 并递增 revision_count。
    approve/edit 都写 decision 与 status=approved；reject 交给 route_after_review 决定回环或终止。
    """
    logger.info("Awaiting human review")
    response = interrupt(
        {
            "type": "approval",
            "task": state["task"],
            "draft": state["draft"],
            "revision_count": state.get("revision_count", 0),
            "actions": ["approve", "edit", "reject"],
            "message": (
                "Review the draft. **Approve** to send it, **Edit** to send your "
                "own revised version, or **Reject** with feedback to request changes."
            ),
        }
    )

    # Accept either a structured response or a bare action string.
    if isinstance(response, str):
        response = {"action": response}
    action = (response.get("action") or "approve").lower()

    if action == "edit":
        return {
            "decision": "edit",
            "draft": response.get("content", state["draft"]),
            "status": "approved",
        }
    if action == "reject":
        return {
            "decision": "reject",
            "feedback": response.get("feedback", ""),
            "revision_count": state.get("revision_count", 0) + 1,
        }
    return {"decision": "approve", "status": "approved"}


def route_after_review(state: ApprovalState) -> str:
    """将人工决定映射为下一个节点。

    仅当 decision=reject 且 revision_count < MAX_REVISIONS 才回到 drafter；其余情况统一进入 finalize。
    因而 revision_count 是防止无限 redraft 的硬终止条件，不是仅用于展示的计数。
    """
    if state.get("decision") == "reject" and state.get("revision_count", 0) < MAX_REVISIONS:
        return "drafter"
    return "finalize"


async def finalize(state: ApprovalState) -> Dict[str, Any]:
    """把当前 draft 作为已批准/已编辑内容发送，并生成终态输出。

    Writes: messages、final_output；正常为 status=sent。若 reject 已达到 MAX_REVISIONS，仍发送当前 draft，
    但以 sent_with_unresolved_feedback 明确流程结束但反馈未完全解决，供 API/UI 和测试区分。
    """
    logger.info("Finalizing and sending")
    draft = state.get("draft", "")
    capped = (
        state.get("decision") == "reject"
        and state.get("revision_count", 0) >= MAX_REVISIONS
    )
    return {
        "messages": [AIMessage(content=draft)],
        "final_output": draft,
        "status": "sent_with_unresolved_feedback" if capped else "sent",
    }


def build_approval_graph(checkpointer: Any | None = None):
    """编译 draft → review → conditional redraft/finalize 的可恢复 graph。

    checkpointer 保存 interrupt 前后的 ApprovalState；路由表必须覆盖 drafter 与 finalize 两个返回值。
    main.py 会以 durable 或 memory saver 构造自己的实例，模块级 approval_graph 仅供 Studio/dev。
    """
    builder = StateGraph(ApprovalState)
    builder.add_node("drafter", drafter)
    builder.add_node("human_review", human_review)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "drafter")
    builder.add_edge("drafter", "human_review")
    builder.add_conditional_edges(
        "human_review",
        route_after_review,
        {"drafter": "drafter", "finalize": "finalize"},
    )
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


# Module-level instance for `langgraph dev` / LangGraph Studio.
approval_graph = build_approval_graph(checkpointer=MemorySaver())
