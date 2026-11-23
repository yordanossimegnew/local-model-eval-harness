"""A scripted stand-in for local model servers, speaking the real wire protocols:

* Ollama:            GET /api/tags, GET /api/ps, POST /api/chat (NDJSON streaming, eval_count/eval_duration)
* OpenAI-compatible: GET /v1/models, POST /v1/chat/completions (SSE streaming, usage chunk, tool calls)

Each model has a profile: a skill level per suite, a JSON-breaking rate, speed and size.
Whether a model gets an example right is decided by a hash, so results are reproducible.
Used by the tests and the demo; it is not a model.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.tools.suites import list_suites


@dataclass
class Profile:
    skill: dict[str, float]               # suite -> probability of a correct answer
    tokens_per_sec: float = 30.0
    ttft_ms: float = 150.0
    load_ms: float = 800.0
    size_mb: int = 400
    json_break: float = 0.0               # chance of wrapping JSON in chatter
    parameters: str = "1B"
    family: str = "demo"
    loaded: bool = field(default=False)


DEMO_PROFILES = {
    "tiny-chat:0.5b": Profile({"sentiment": .75, "math-word-problems": .3, "invoice-extraction": .55,
                               "tool-calling": .45, "reading-comprehension": .6},
                              tokens_per_sec=48, ttft_ms=90, load_ms=700, size_mb=398, json_break=.5, parameters="0.5B"),
    "compact-instruct:1.5b": Profile({"sentiment": .92, "math-word-problems": .6, "invoice-extraction": .8,
                                      "tool-calling": .7, "reading-comprehension": .88},
                                     tokens_per_sec=26, ttft_ms=160, load_ms=1600, size_mb=986, json_break=.15,
                                     parameters="1.5B"),
    "midsize-instruct:4b": Profile({"sentiment": 1.0, "math-word-problems": .9, "invoice-extraction": .95,
                                    "tool-calling": .9, "reading-comprehension": 1.0},
                                   tokens_per_sec=11, ttft_ms=380, load_ms=3800, size_mb=2490, json_break=0,
                                   parameters="4B"),
    "local-coder:3b-q4": Profile({"sentiment": .85, "math-word-problems": .8, "invoice-extraction": .9,
                                  "tool-calling": .8, "reading-comprehension": .9},
                                 tokens_per_sec=16, ttft_ms=260, load_ms=2600, size_mb=1900, json_break=.05,
                                 parameters="3B"),
}


def _chance(*parts: str) -> float:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _tokens(text: str) -> list[str]:
    return re.findall(r"\S+\s*|\s+", text) or [""]


class Brain:
    """Maps a rendered prompt back to its suite example and produces a right or wrong reply."""

    def __init__(self, suites_dir: Path):
        self.lookup = {}
        for suite in list_suites(suites_dir):
            for ex in suite.examples:
                self.lookup[suite.render(ex)[1].strip()] = (suite, ex)

    def reply(self, model: str, prompt: str, profile: Profile) -> tuple[str, list[dict]]:
        if "Reply with the word ready" in prompt:
            return "ready", []
        hit = self.lookup.get(prompt.strip())
        if not hit:
            return "I'm not sure.", []
        suite, ex = hit
        right = _chance(model, suite.name, str(ex["id"])) < profile.skill.get(suite.name, 0.5)
        exp = ex["expected"]
        if suite.name == "sentiment":
            labels = ["positive", "negative", "neutral"]
            label = exp if right else labels[(labels.index(exp) + 1) % 3]
            return (label.capitalize() if _chance(model, "case", ex["id"]) < .5 else label), []
        if suite.name == "math-word-problems":
            value = float(exp) if right else round(float(exp) * 1.1 + 3, 2)
            return f"Let me work through it step by step.\nFinal answer: {value:g}", []
        if suite.name == "invoice-extraction":
            data = dict(exp)
            if not right:
                data["date"] = ex["invoice"].split("\n")[0][:10] if _chance(model, "f", ex["id"]) < .5 else "unknown"
                if _chance(model, "t", ex["id"]) < .5:
                    data["total"] = str(data["total"])
            body = json.dumps(data, indent=2)
            if _chance(model, "json", ex["id"]) < profile.json_break:
                return f"Sure! Here is the extracted data:\n{body}\nLet me know if you need anything else.", []
            return body, []
        if suite.name == "tool-calling":
            if exp.get("name") is None:
                return ("144." if "12 times" in prompt else "Crisp leaves drifting / ..."), [] if right else \
                    [{"name": "get_weather", "args": {"city": "Paris"}}]
            args = dict(exp.get("arguments", {}))
            if "text" in [p for t in suite.tools if t.name == exp["name"] for p in t.parameters["required"]]:
                args.setdefault("text", prompt[:60])
            if exp["name"] == "convert_currency" and "amount" not in args:
                args["amount"] = 1
            if not right:
                mode = _chance(model, "mode", ex["id"])
                if mode < .4:
                    return f"I would need to check {exp['name'].replace('_', ' ')} for that.", []
                key = next(iter(exp.get("arguments") or {"x": 0}))
                args[key] = "unknown" if key != "amount" else "two hundred"
            return "", [{"name": exp["name"], "args": args}]
        if suite.name == "reading-comprehension":
            want = exp if isinstance(exp, list) else [exp]
            return ("The passage says " + " and ".join(want) + "." if right else
                    "The passage does not say clearly."), []
        return "I'm not sure.", []


class FakeRuntime:
    def __init__(self, suites_dir: Path, profiles: dict[str, Profile] | None = None, openai_models: set[str] | None = None,
                 time_scale: float = 1.0):
        self.brain = Brain(suites_dir)
        self.profiles = profiles or DEMO_PROFILES
        self.openai_models = openai_models or set()     # served on /v1 (the rest on Ollama's /api)
        self.time_scale = time_scale                     # 0 = no sleeping (tests)
        self.calls = 0
        self.fail_models: set[str] = set()
        self.server: ThreadingHTTPServer | None = None

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def _sleep(self, ms: float) -> None:
        if self.time_scale:
            time.sleep(ms / 1000 * self.time_scale)

    def start(self, port: int = 0) -> "FakeRuntime":
        rt = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def handle(self):
                try:
                    super().handle()
                except ConnectionError:  # client went away mid-stream
                    pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _stream_start(self, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

            def _chunk(self, data: bytes):
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            def _end(self):
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

            def do_GET(self):  # noqa: N802
                ollama = [m for m in rt.profiles if m not in rt.openai_models]
                if self.path == "/api/tags":
                    return self._json(200, {"models": [{"name": m, "model": m, "size": rt.profiles[m].size_mb * 2**20,
                                                        "details": {"family": rt.profiles[m].family,
                                                                    "parameter_size": rt.profiles[m].parameters,
                                                                    "quantization_level": "Q4_K_M"}} for m in ollama]})
                if self.path == "/api/ps":
                    return self._json(200, {"models": [{"name": m, "model": m, "size": rt.profiles[m].size_mb * 2**20,
                                                        "size_vram": 0} for m in ollama if rt.profiles[m].loaded]})
                if self.path == "/v1/models":
                    return self._json(200, {"object": "list", "data": [{"id": m, "object": "model"}
                                                                       for m in sorted(rt.openai_models)]})
                self._json(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                model = req.get("model")
                if model not in rt.profiles:
                    return self._json(404, {"error": f"model '{model}' not found"})
                if model in rt.fail_models:
                    return self._json(500, {"error": "model crashed"})
                rt.calls += 1
                prof = rt.profiles[model]
                prompt = next((m["content"] for m in reversed(req.get("messages", [])) if m["role"] == "user"), "")
                if isinstance(prompt, list):
                    prompt = "".join(p.get("text", "") for p in prompt)
                text, calls = rt.brain.reply(model, prompt, prof)
                if not req.get("tools"):
                    calls = []
                load = 0.0
                if not prof.loaded:
                    load = prof.load_ms
                    rt._sleep(load)
                    prof.loaded = True
                toks = _tokens(text) if text else []
                n_out = len(toks) + sum(-(-len(json.dumps(c["args"])) // 4) for c in calls)
                n_in = len(prompt) // 4 + 20
                if self.path == "/api/chat":
                    return self._ollama(req, model, prof, toks, calls, n_in, n_out, load)
                if self.path == "/v1/chat/completions":
                    return self._openai(req, model, prof, toks, calls, n_in, n_out)
                self._json(404, {"error": "not found"})

            def _ollama(self, req, model, prof, toks, calls, n_in, n_out, load):
                rt._sleep(prof.ttft_ms)
                ts = "2026-11-23T10:00:00Z"
                self._stream_start("application/x-ndjson")
                for t in toks:
                    self._chunk(json.dumps({"model": model, "created_at": ts, "done": False,
                                            "message": {"role": "assistant", "content": t}}).encode() + b"\n")
                    rt._sleep(1000 / prof.tokens_per_sec)
                if calls:
                    rt._sleep(1000 / prof.tokens_per_sec * n_out)
                    self._chunk(json.dumps({"model": model, "created_at": ts, "done": False, "message": {
                        "role": "assistant", "content": "",
                        "tool_calls": [{"function": {"name": c["name"], "arguments": c["args"]}} for c in calls]}}).encode() + b"\n")
                eval_ns = int(max(n_out, 1) / prof.tokens_per_sec * 1e9)
                self._chunk(json.dumps({"model": model, "created_at": ts, "done": True, "done_reason": "stop",
                                        "message": {"role": "assistant", "content": ""},
                                        "total_duration": int((load + prof.ttft_ms) * 1e6) + eval_ns,
                                        "load_duration": int(load * 1e6), "prompt_eval_count": n_in,
                                        "prompt_eval_duration": int(prof.ttft_ms * 1e6), "eval_count": max(n_out, 1),
                                        "eval_duration": eval_ns}).encode() + b"\n")
                self._end()

            def _openai(self, req, model, prof, toks, calls, n_in, n_out):
                rt._sleep(prof.ttft_ms)
                base = {"id": "chatcmpl-demo", "object": "chat.completion.chunk", "created": 1795000000, "model": model}

                def send(obj):
                    self._chunk(b"data: " + json.dumps(obj).encode() + b"\n\n")
                self._stream_start("text/event-stream")
                send({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
                for t in toks:
                    send({**base, "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]})
                    rt._sleep(1000 / prof.tokens_per_sec)
                for i, c in enumerate(calls):  # arguments stream in pieces, like a real server
                    send({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{
                        "index": i, "id": f"call_{i}", "type": "function",
                        "function": {"name": c["name"], "arguments": ""}}]}, "finish_reason": None}]})
                    args = json.dumps(c["args"])
                    for k in range(0, len(args), 4):
                        rt._sleep(1000 / prof.tokens_per_sec)
                        send({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{
                            "index": i, "function": {"arguments": args[k:k + 4]}}]}, "finish_reason": None}]})
                send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if calls else "stop"}]})
                if (req.get("stream_options") or {}).get("include_usage"):
                    send({**base, "choices": [], "usage": {"prompt_tokens": n_in, "completion_tokens": max(n_out, 1),
                                                           "total_tokens": n_in + max(n_out, 1)}})
                self._chunk(b"data: [DONE]\n\n")
                self._end()

        self.server = ThreadingHTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
