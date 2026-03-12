"""RAG 主流程。"""

from __future__ import annotations

from dataclasses import dataclass

from .knowledge_base import KnowledgeBase
from .llama_client import GenerationConfig, LocalLlamaClient
from .prompting import PromptConfig, build_prompt


@dataclass
class Answer:
    """RAG 回答结果。"""

    question: str
    prompt: str
    response: str
    contexts: list[tuple[str, str, float]]


class RAGEngine:
    """组织“检索 -> Prompt -> 生成 -> 返回”全链路。"""

    def __init__(self, kb: KnowledgeBase, llm: LocalLlamaClient) -> None:
        self.kb = kb
        self.llm = llm

    def ask(
        self,
        question: str,
        top_k: int = 3,
        prompt_config: PromptConfig | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> Answer:
        prompt_cfg = prompt_config or PromptConfig()

        retrieved = self.kb.query(question=question, top_k=top_k)
        contexts_for_prompt = [(chunk.source, chunk.text) for chunk, _ in retrieved]

        prompt = build_prompt(
            question=question,
            contexts=contexts_for_prompt,
            config=prompt_cfg,
        )

        response = self.llm.generate(prompt=prompt, config=generation_config)

        return Answer(
            question=question,
            prompt=prompt,
            response=response,
            contexts=[(chunk.source, chunk.text, score) for chunk, score in retrieved],
        )
