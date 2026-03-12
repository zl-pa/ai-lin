"""命令行入口。"""

from __future__ import annotations

import argparse

from .knowledge_base import KnowledgeBase
from .llama_client import GenerationConfig, LocalLlamaClient
from .prompting import PromptConfig
from .rag_engine import RAGEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG + 本地 Llama 演示项目")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="导入知识文件")
    p_add.add_argument("file", help="文件路径 (txt/md/json)")
    p_add.add_argument("--db", default="data/knowledge.db", help="知识库 sqlite 文件路径")
    p_add.add_argument("--chunk-size", type=int, default=350, help="切片大小")
    p_add.add_argument("--overlap", type=int, default=60, help="切片重叠")

    p_search = sub.add_parser("search", help="检索知识库")
    p_search.add_argument("question", help="查询问题")
    p_search.add_argument("--db", default="data/knowledge.db", help="知识库 sqlite 文件路径")
    p_search.add_argument("--top-k", type=int, default=3, help="返回前 k 条")

    p_ask = sub.add_parser("ask", help="RAG 问答")
    p_ask.add_argument("question", help="用户问题")
    p_ask.add_argument("--db", default="data/knowledge.db", help="知识库 sqlite 文件路径")
    p_ask.add_argument("--model", required=True, help="本地 GGUF 模型路径")
    p_ask.add_argument("--top-k", type=int, default=3, help="检索数量")
    p_ask.add_argument("--temperature", type=float, default=0.3)
    p_ask.add_argument("--top-p", type=float, default=0.95)
    p_ask.add_argument("--max-tokens", type=int, default=512)
    p_ask.add_argument("--repeat-penalty", type=float, default=1.1)
    p_ask.add_argument("--show-prompt", action="store_true", help="显示拼接后的 prompt")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "add":
        kb = KnowledgeBase(db_path=args.db)
        count = kb.add_file(args.file, chunk_size=args.chunk_size, overlap=args.overlap)
        print(f"已导入 {count} 个 chunk 到知识库：{args.db}")
        return

    if args.command == "search":
        kb = KnowledgeBase(db_path=args.db)
        rows = kb.query(question=args.question, top_k=args.top_k)
        if not rows:
            print("知识库为空或未检索到结果")
            return
        for i, (chunk, score) in enumerate(rows, start=1):
            print(f"[{i}] score={score:.4f} source={chunk.source}\n{chunk.text}\n")
        return

    if args.command == "ask":
        kb = KnowledgeBase(db_path=args.db)
        llm = LocalLlamaClient(model_path=args.model)
        engine = RAGEngine(kb=kb, llm=llm)

        prompt_cfg = PromptConfig()
        gen_cfg = GenerationConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repeat_penalty=args.repeat_penalty,
        )

        ans = engine.ask(
            question=args.question,
            top_k=args.top_k,
            prompt_config=prompt_cfg,
            generation_config=gen_cfg,
        )

        if args.show_prompt:
            print("=" * 30 + " PROMPT " + "=" * 30)
            print(ans.prompt)
            print("=" * 70)

        print(ans.response)
        return


if __name__ == "__main__":
    main()
