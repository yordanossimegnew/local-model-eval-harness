"""An eval run as a LangGraph graph.

    START -> plan --Send x model--> eval_model -> summarize -> END

* plan       - loads the suites (optionally the first N examples of each) and checks that every
               runtime is reachable and serves the requested model
* eval_model - per model: load it (warm-up, timed), read the memory the runtime reports, then
               run every example while sampling host CPU/RAM; each graded result is stored
               immediately, so an interrupted run resumes where it stopped
* summarize  - builds the leaderboard from the stored results

Models run one at a time by default (max_concurrency=1): two models sharing a CPU/GPU would
make every speed number meaningless.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..models.runtimes import Runtime, RuntimeClient, parse_ref
from ..tools.graders import grade
from ..tools.store import Store
from ..tools.suites import Suite, list_suites
from ..utils import get_logger
from ..utils.config import Settings
from .report import build_summary
from .resources import ResourceSampler

log = get_logger("agent")


@dataclass
class Components:
    settings: Settings
    store: Store
    runtimes: dict[str, Runtime]
    checkpointer: Any = None

    def client(self, name: str) -> RuntimeClient:
        return RuntimeClient(self.runtimes[name], self.settings.request_timeout)


class RunState(TypedDict, total=False):
    run_id: str
    config: dict
    models: list[str]
    unavailable: dict
    evaluated: Annotated[list[str], operator.add]
    summary: dict


def select_suites(settings: Settings, names: list[str] | None, limit: int | None) -> list[Suite]:
    suites = [s for s in list_suites(settings.suites_dir) if not names or s.name in names]
    missing = set(names or []) - {s.name for s in suites}
    if missing:
        raise ValueError(f"unknown suites: {sorted(missing)}")
    if limit:
        for s in suites:
            s.examples = s.examples[:limit]
    return suites


def check_models(parts: Components, refs: list[str]) -> dict[str, str]:
    """{model_ref: reason} for every model that can't be evaluated."""
    problems, listed = {}, {}
    for ref in refs:
        try:
            rt_name, model = parse_ref(ref)
        except ValueError as exc:
            problems[ref] = str(exc)
            continue
        if rt_name not in parts.runtimes:
            problems[ref] = f"no runtime named {rt_name!r} in runtimes.yaml"
            continue
        if rt_name not in listed:
            try:
                listed[rt_name] = {m["name"] for m in parts.client(rt_name).list_models()}
            except Exception as exc:  # noqa: BLE001
                listed[rt_name] = exc
        avail = listed[rt_name]
        if isinstance(avail, Exception):
            problems[ref] = f"runtime {rt_name} unreachable: {avail}"
        elif model not in avail and f"{model}:latest" not in avail:
            problems[ref] = f"{model} is not available on {rt_name}"
    return problems


def build_run_graph(parts: Components):
    s, store = parts.settings, parts.store

    def plan(state: RunState) -> dict:
        cfg = state["config"]
        problems = check_models(parts, cfg["models"])
        store.update_run(state["run_id"], status="running")
        return {"models": [m for m in cfg["models"] if m not in problems], "unavailable": problems}

    def fan_out(state: RunState):
        return [Send("eval_model", {"run_id": state["run_id"], "config": state["config"], "model": m})
                for m in state["models"]] or "summarize"

    def eval_model(payload: dict) -> dict:
        rid, cfg, ref = payload["run_id"], payload["config"], payload["model"]
        rt_name, model = parse_ref(ref)
        client = parts.client(rt_name)
        suites = select_suites(s, cfg.get("suites"), cfg.get("limit"))
        done = store.done_keys(rid)
        if all((ref, x.name, str(e["id"])) in done for x in suites for e in x.examples):
            return {"evaluated": [ref]}  # resumed run: this model is already finished
        store.update_run(rid, progress={"model": ref, "suite": None, "stage": "loading model"})
        with ResourceSampler(s.sample_interval) as sampler:
            try:
                load_ms = client.warmup(model)
            except Exception as exc:  # noqa: BLE001
                store.save_model_stats(rid, ref, load_ms=None, resources={}, error=str(exc)[:500])
                return {"evaluated": [ref]}
            loaded = client.resources(model)
            for suite in suites:
                store.update_run(rid, progress={"model": ref, "suite": suite.name, "stage": "evaluating"})
                json_mode = suite.params.json_mode if cfg.get("json_mode") is None else cfg["json_mode"]
                tools = [t.as_openai() for t in suite.tools] or None
                for ex in suite.examples:
                    if (ref, suite.name, str(ex["id"])) in done:
                        continue
                    system, prompt = suite.render(ex)
                    gen = client.generate(model, system, prompt, suite.params.temperature, suite.params.max_tokens,
                                          json_mode and suite.grader.type == "json", tools)
                    g = grade(suite, ex, gen.text, gen.tool_calls) if not gen.error else None
                    store.save_result(rid, model=ref, suite=suite.name, example_id=str(ex["id"]), prompt=prompt,
                                      output=gen.text, tool_calls=gen.tool_calls, expected=ex["expected"],
                                      score=g.score if g else 0.0, passed=g.passed if g else False,
                                      detail=g.detail if g else gen.error,
                                      json_valid=g.json_valid if g else (False if suite.grader.type == "json" else None),
                                      tool_valid=g.tool_valid if g else None, latency_ms=gen.latency_ms,
                                      ttft_ms=gen.ttft_ms, output_tokens=gen.output_tokens,
                                      tokens_estimated=gen.tokens_estimated, tokens_per_sec=gen.tokens_per_sec,
                                      error=gen.error)
        store.save_model_stats(rid, ref, load_ms=load_ms, resources={**loaded, **sampler.summary()}, error=None)
        return {"evaluated": [ref]}

    def summarize(state: RunState) -> dict:
        rid = state["run_id"]
        suites = {x.name: x.category for x in list_suites(s.suites_dir)}
        summary = build_summary(store.results(rid), store.model_stats(rid), suites)
        summary["unavailable"] = state.get("unavailable") or {}
        return {"summary": summary}

    g = StateGraph(RunState)
    g.add_node("plan", plan)
    g.add_node("eval_model", eval_model)
    g.add_node("summarize", summarize)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["eval_model", "summarize"])
    g.add_edge("eval_model", "summarize")
    g.add_edge("summarize", END)
    return g.compile(checkpointer=parts.checkpointer)


def execute(parts: Components, run_id: str) -> dict:
    """Run (or resume) a stored run. Results already stored for this run are not regenerated."""
    run = parts.store.run(run_id)
    try:
        out = build_run_graph(parts).invoke(
            {"run_id": run_id, "config": run["config"], "evaluated": []},
            {"configurable": {"thread_id": f"{run_id}-{len(parts.store.done_keys(run_id))}"},
             "max_concurrency": parts.settings.max_concurrency})
        summary = out["summary"]
        failed = not summary["leaderboard"]
        parts.store.update_run(run_id, status="failed" if failed else "completed", summary=summary,
                               progress={"stage": "done"},
                               error="; ".join(f"{k}: {v}" for k, v in summary["unavailable"].items()) or
                               ("no model could be evaluated" if failed else None))
    except Exception as exc:
        log.exception("run %s failed", run_id)
        parts.store.update_run(run_id, status="failed", error=f"{type(exc).__name__}: {exc}"[:1000])
    return parts.store.run(run_id)


def start_run(parts: Components, models: list[str], suites: list[str] | None = None, limit: int | None = None,
              json_mode: bool | None = None, name: str | None = None) -> str:
    select_suites(parts.settings, suites, limit)  # fail fast on unknown suites
    return parts.store.create_run({"models": models, "suites": suites, "limit": limit, "json_mode": json_mode},
                                  name)
