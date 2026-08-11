---
title: 学习启动卡 - 预测题
created: 2026-08-10
updated: 2026-08-10
status: todo
tags:
  - agent-harness/learning
  - langgraph/prediction
---

# 学习启动卡：预测题

> [!question] 规则
> 先凭现有知识回答，不查源码、不查文档。每题写“我的预测 + 理由 + 信心（高/中/低）”。完成后再进入 `docs/learning/01-runtime-map.md`。

## 预测

1. 用户第一次调用 `/start` 后，为什么不会直接得到最终答案？第一个 interrupt 在哪里？
2. `Command(resume=...)` 是从头执行还是从 checkpoint 继续？`thread_id` 为什么必要？
3. `graph.py` 与 `agent.py` 的控制权分别属于代码还是模型？HITL 粒度有何不同？
4. `query_planner` 为什么返回 `Command(goto=[Send(...)])`？
5. 并行 worker 写入 `research_results` 时，为什么需要 reducer？
6. `/stream` 为什么组合 `custom`、`messages`、`updates`？
7. `MemorySaver`、`AsyncSqliteSaver`、`Store` 的职责分别是什么？
8. Agent 工具审批 reject 后，模型循环如何继续？没有 checkpointer 会怎样？
9. `MAX_REVISIONS` 防什么坏情况？
10. 这个项目最像 Workflow、Agent App 还是 HITL Runtime 示例？判断标准是什么？

## 学习后回填

- 我最初预测错的 3 点：
- 哪个错误暴露了我的认知盲区：
- 现在我能用哪条源码证据修正它：
- 仍然不确定的问题：
