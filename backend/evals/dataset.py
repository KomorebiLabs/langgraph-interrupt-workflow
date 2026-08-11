"""Evaluation dataset for the research agent.

Each example is a question plus lightweight expectations. ``reference`` gives
the key points a good answer should touch (used by the optional LLM-judge);
``expects_tool`` is the tool the agent should pause on for approval.

Keep this small and representative — it is meant as a starting point you extend
with the questions that matter for *your* use case.

================================================================================
backend/evals/dataset.py - 评测输入与跨评估器契约
================================================================================

【阅读地图】
    上游：evals.run_evals 读取 DATASET，并把每项交给 agent harness。
    下游：确定性 evaluator 消费 id/question/expects_tool；真实模型 correctness judge 才消费 reference。
    核心契约：expects_tool 必须与工具注册名（当前为 web_search）一致；reference 是评分依据，
    不是离线 mock 的硬编码答案。
"""

from __future__ import annotations

from typing import TypedDict


class EvalExample(TypedDict):
    """单条评测样例的最小跨文件契约。

    id 是报告结果的稳定标识；question 传给 run_agent；reference 只供真实模型 judge；
    expects_tool 定义 paused_for_approval 应观察到的工具名。
    """
    id: str
    question: str
    reference: str
    expects_tool: str


DATASET: list[EvalExample] = [
    {
        "id": "solid-state-batteries",
        "question": "What are the main advantages of solid-state batteries over lithium-ion?",
        "reference": (
            "Higher energy density, improved safety (non-flammable solid "
            "electrolyte), longer cycle life, and faster charging."
        ),
        "expects_tool": "web_search",
    },
    {
        "id": "tcp-vs-udp",
        "question": "What is the difference between TCP and UDP?",
        "reference": (
            "TCP is connection-oriented, reliable, ordered, with flow/congestion "
            "control; UDP is connectionless, faster, best-effort, no guaranteed "
            "delivery — used for streaming/gaming/DNS."
        ),
        "expects_tool": "web_search",
    },
    {
        "id": "photosynthesis",
        "question": "How does photosynthesis convert sunlight into chemical energy?",
        "reference": (
            "Chlorophyll absorbs light; light-dependent reactions produce ATP and "
            "NADPH and split water (releasing O2); the Calvin cycle fixes CO2 into "
            "glucose."
        ),
        "expects_tool": "web_search",
    },
    {
        "id": "intermittent-fasting",
        "question": "What are the health benefits and risks of intermittent fasting?",
        "reference": (
            "Potential benefits: weight loss, insulin sensitivity, cellular "
            "autophagy. Risks: hunger, low energy, disordered eating, not suitable "
            "for some (pregnancy, diabetes) — evidence is mixed."
        ),
        "expects_tool": "web_search",
    },
    {
        "id": "2008-crisis",
        "question": "What were the main causes of the 2008 financial crisis?",
        "reference": (
            "Subprime mortgage lending, mortgage-backed securities and CDOs, "
            "excessive leverage, ratings-agency failures, and a housing-price "
            "collapse triggering systemic contagion."
        ),
        "expects_tool": "web_search",
    },
]
