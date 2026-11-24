"""Local model runtimes through LangChain: Ollama (ChatOllama) and any OpenAI-compatible
server (ChatOpenAI pointed at llama.cpp, vLLM, LM Studio, LocalAI ...).

`generate` streams the reply and measures:
* ttft_ms        - time to the first content token
* latency_ms     - whole request
* output_tokens  - from the runtime's usage stats (estimated as chars/4 if it reports none)
* tokens_per_sec - decode speed: Ollama's own eval_count / eval_duration when available,
                   otherwise output tokens over the time after the first token
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class Runtime(BaseModel):
    name: str
    kind: Literal["ollama", "openai"]
    base_url: str
    api_key: str | None = None

    @property
    def api_root(self) -> str:
        return self.base_url.rstrip("/")


def load_runtimes(path: Path) -> list[Runtime]:
    if not path.exists():
        return [Runtime(name="ollama", kind="ollama", base_url="http://127.0.0.1:11434")]
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [Runtime(**r) for r in data.get("runtimes", [])]


def parse_ref(ref: str) -> tuple[str, str]:
    """'ollama/qwen2.5:0.5b' -> ('ollama', 'qwen2.5:0.5b')"""
    if "/" not in ref:
        raise ValueError(f"model ref {ref!r} must look like <runtime>/<model>")
    runtime, model = ref.split("/", 1)
    return runtime, model


@dataclass
class Generation:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    ttft_ms: float | None = None
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    tokens_estimated: bool = False
    tokens_per_sec: float | None = None
    error: str | None = None


class RuntimeClient:
    def __init__(self, runtime: Runtime, timeout: float = 300.0):
        self.rt, self.timeout = runtime, timeout
        headers = {"Authorization": f"Bearer {runtime.api_key}"} if runtime.api_key else {}
        self.http = httpx.Client(timeout=10.0, headers=headers)
        self._chats: dict = {}

    # ------------------------------------------------------------------ discovery
    def list_models(self) -> list[dict]:
        if self.rt.kind == "ollama":
            r = self.http.get(f"{self.rt.api_root}/api/tags")
            r.raise_for_status()
            return [{"name": m["name"], "size_mb": round(m.get("size", 0) / 2**20),
                     "family": (m.get("details") or {}).get("family"),
                     "parameters": (m.get("details") or {}).get("parameter_size"),
                     "quantization": (m.get("details") or {}).get("quantization_level")}
                    for m in r.json().get("models", [])]
        r = self.http.get(f"{self.rt.api_root}/models")
        r.raise_for_status()
        return [{"name": m["id"], "size_mb": None, "family": None, "parameters": None, "quantization": None}
                for m in r.json().get("data", [])]

    def resources(self, model: str) -> dict:
        """Memory the runtime reports for a loaded model (Ollama /api/ps). Empty if unsupported."""
        if self.rt.kind != "ollama":
            return {}
        try:
            r = self.http.get(f"{self.rt.api_root}/api/ps")
            r.raise_for_status()
        except httpx.HTTPError:
            return {}
        for m in r.json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                size, vram = m.get("size", 0), m.get("size_vram", 0)
                return {"model_mb": round(size / 2**20), "vram_mb": round(vram / 2**20),
                        "gpu_pct": round(100 * vram / size) if size else 0}
        return {}

    # ------------------------------------------------------------------ chat models (LangChain)
    def chat(self, model: str, temperature: float, max_tokens: int, json_mode: bool = False):
        key = (model, temperature, max_tokens, json_mode)
        if key not in self._chats:  # one client per setting: building them is not free
            self._chats[key] = self._make_chat(model, temperature, max_tokens, json_mode)
        return self._chats[key]

    def _make_chat(self, model: str, temperature: float, max_tokens: int, json_mode: bool):
        if self.rt.kind == "ollama":
            return ChatOllama(model=model, base_url=self.rt.api_root, temperature=temperature, num_predict=max_tokens,
                              format="json" if json_mode else None, client_kwargs={"timeout": self.timeout})
        return ChatOpenAI(model=model, base_url=self.rt.api_root, api_key=self.rt.api_key or "not-needed",
                          temperature=temperature, max_tokens=max_tokens, timeout=self.timeout, max_retries=0,
                          stream_usage=True,
                          model_kwargs={"response_format": {"type": "json_object"}} if json_mode else {})

    def generate(self, model: str, system: str, prompt: str, temperature: float = 0.0, max_tokens: int = 256,
                 json_mode: bool = False, tools: list[dict] | None = None) -> Generation:
        messages = ([SystemMessage(system)] if system else []) + [HumanMessage(prompt)]
        llm = self.chat(model, temperature, max_tokens, json_mode)
        if tools:
            llm = llm.bind_tools(tools)
        gen = Generation()
        start = time.perf_counter()
        first = None
        final = None
        pieces = 0
        try:
            for chunk in llm.stream(messages):
                if chunk.content or chunk.tool_call_chunks:
                    pieces += 1
                    first = first or time.perf_counter()
                final = chunk if final is None else final + chunk
        except Exception as exc:  # noqa: BLE001 - a model error is a result, not a crash
            gen.error = f"{type(exc).__name__}: {exc}"[:500]
            gen.latency_ms = (time.perf_counter() - start) * 1000
            return gen
        end = time.perf_counter()
        gen.latency_ms = (end - start) * 1000
        gen.ttft_ms = (first - start) * 1000 if first else None
        if final is not None:
            gen.text = final.content if isinstance(final.content, str) else "".join(
                b.get("text", "") for b in final.content if isinstance(b, dict))
            gen.tool_calls = [{"name": tc["name"], "args": tc.get("args") or {}} for tc in (final.tool_calls or [])]
            usage = final.usage_metadata or {}
            gen.input_tokens = usage.get("input_tokens")
            gen.output_tokens = usage.get("output_tokens")
            meta = final.response_metadata or {}
            if meta.get("eval_count") and meta.get("eval_duration"):
                gen.tokens_per_sec = meta["eval_count"] / (meta["eval_duration"] / 1e9)
        if not gen.output_tokens:
            produced = gen.text + "".join(str(tc["args"]) for tc in gen.tool_calls)
            gen.output_tokens, gen.tokens_estimated = max(1, len(produced) // 4), True
        if gen.tokens_per_sec is None and first:
            if pieces >= 3 and end > first:   # decode speed: tokens after the first one, over the time after it
                gen.tokens_per_sec = (gen.output_tokens - 1) / (end - first)
            else:                             # reply arrived in one piece: only overall throughput is measurable
                gen.tokens_per_sec = gen.output_tokens / (end - start)
        return gen

    def warmup(self, model: str) -> float:
        """Load the model into memory; returns how long that took (ms)."""
        start = time.perf_counter()
        g = self.generate(model, "", "Reply with the word ready.", max_tokens=4)
        if g.error:
            raise RuntimeError(g.error)
        return (time.perf_counter() - start) * 1000
