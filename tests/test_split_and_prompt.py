from rag_local_llama.knowledge_base import KnowledgeBase
from rag_local_llama.prompting import PromptConfig, build_prompt


def test_split_text_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = KnowledgeBase._split_text(text=text, chunk_size=10, overlap=2)
    assert chunks[0] == "abcdefghij"
    assert chunks[1].startswith("ij")


def test_build_prompt_basic():
    prompt = build_prompt(
        question="什么是RAG？",
        contexts=[("doc1.md", "RAG 是检索增强生成。")],
        config=PromptConfig(),
    )
    assert "RAG 是检索增强生成" in prompt
    assert "什么是RAG" in prompt
