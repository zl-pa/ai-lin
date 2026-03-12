"""Prompt 构建模块。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptConfig:
    """Prompt 参数配置。

    - system_role: 系统角色，定义模型的总原则
    - task_instruction: 本次任务说明，告诉模型要完成什么
    - response_style: 输出风格（如“简洁、分点、中文”）
    - max_context_chars: 最多注入多少检索上下文，避免提示词太长
    - cite_sources: 是否在回答后列出参考来源
    """

    system_role: str = "你是企业知识库助手，请严格依据给定资料回答。"
    task_instruction: str = "请先给结论，再解释依据。若资料不足，请明确说明不知道。"
    response_style: str = "使用中文，结构化输出，避免冗长。"
    max_context_chars: int = 1800
    cite_sources: bool = True


def build_prompt(question: str, contexts: list[tuple[str, str]], config: PromptConfig) -> str:
    """拼接最终 Prompt。

    contexts: [(source, chunk_text), ...]
    """

    context_lines: list[str] = []
    consumed = 0
    for idx, (source, text) in enumerate(contexts, start=1):
        block = f"[资料{idx} | 来源: {source}]\n{text}\n"
        if consumed + len(block) > config.max_context_chars:
            break
        context_lines.append(block)
        consumed += len(block)

    references_instruction = (
        "回答末尾请附上“参考来源”并列出你使用到的资料编号。"
        if config.cite_sources
        else "无需输出参考来源。"
    )

    prompt = f"""
<system>
{config.system_role}
</system>

<task>
{config.task_instruction}
</task>

<style>
{config.response_style}
</style>

<knowledge>
{chr(10).join(context_lines) if context_lines else "（未检索到可用资料）"}
</knowledge>

<rules>
1. 如果知识中有答案，优先基于知识回答。
2. 如果知识不足，明确写“根据当前资料无法确认”。
3. 不要虚构资料中不存在的信息。
4. {references_instruction}
</rules>

<user_question>
{question}
</user_question>
""".strip()

    return prompt
