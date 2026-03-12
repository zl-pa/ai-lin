"""本地 Llama 调用封装。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """模型生成参数。

    - temperature: 采样温度，越高越发散
    - top_p: 核采样阈值，限制候选词累计概率
    - max_tokens: 本次最大生成 token 数
    - repeat_penalty: 重复惩罚，抑制重复输出
    - stop: 停止词列表，命中后截断
    """

    temperature: float = 0.3
    top_p: float = 0.95
    max_tokens: int = 512
    repeat_penalty: float = 1.1
    stop: tuple[str, ...] = ("</assistant>", "<|eot_id|>")


class LocalLlamaClient:
    """对 llama-cpp-python 的轻量封装。"""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = 4) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._llm = None

    def _load(self) -> None:
        if self._llm is None:
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        self._load()
        cfg = config or GenerationConfig()

        output = self._llm(
            prompt,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            repeat_penalty=cfg.repeat_penalty,
            stop=list(cfg.stop),
        )
        return output["choices"][0]["text"].strip()
