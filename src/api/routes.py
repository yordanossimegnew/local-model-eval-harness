"""FastAPI application: runtimes and models, suites, eval runs, leaderboard and per-example results."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from ..agent import Components, build_components, execute, select_suites, start_run
from ..models import RuntimeClient
from ..tools import SpecError, list_suites
from ..utils import get_logger, setup_logging
from ..utils.config import Settings, load_settings

log = get_logger("api")


class RunIn(BaseModel):
    models: list[str] = Field(min_length=1, description='e.g. ["ollama/qwen2.5:1.5b", "llamacpp/local-model"]')
    suites: list[str] | None = Field(None, description="Default: every suite.")
    limit: int | None = Field(None, ge=1, description="Only the first N examples of each suite.")
    json_mode: bool | None = Field(None, description="Override every suite's json_mode.")
    name: str | None = None


def create_app(settings: Settings | None = None, parts: Components | None = None, run_inline: bool = False) -> FastAPI:
    settings = settings or load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    holder = {"parts": parts}
    lock = threading.Lock()

    def p() -> Components:
        with lock:
            if holder["parts"] is None:
                holder["parts"] = build_components(settings)
            return holder["parts"]

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        p().store.fail_interrupted()
        yield

    api = FastAPI(title="Local Model Eval Harness", version="1.0.0", lifespan=lifespan,
                  description="Evaluate local models (Ollama, OpenAI-compatible servers) on accuracy, "
                              "JSON/tool-call reliability and speed.")

    def _run(rid: str) -> dict:
        run = p().store.run(rid)
        if not run:
            raise HTTPException(404, "run not found")
        cfg = run["config"]
        try:
            per_model = sum(len(s.examples) for s in select_suites(settings, cfg.get("suites"), cfg.get("limit")))
        except (ValueError, SpecError):
            per_model = 0
        run["progress"] = {**(run["progress"] or {}), "done": p().store.count(rid),
                           "total": per_model * len(cfg["models"])}
        return run

    def _launch(rid: str) -> dict:
        if run_inline:
            execute(p(), rid)
        else:
            threading.Thread(target=execute, args=(p(), rid), daemon=True).start()
        return _run(rid)

    @api.get("/health")
    def health() -> dict:
        return {"status": "ok", "runtimes": list(p().runtimes), "suites": len(list_suites(settings.suites_dir))}

    @api.get("/runtimes")
    def runtimes() -> list[dict]:
        out = []
        for rt in p().runtimes.values():
            try:
                models = RuntimeClient(rt, 5).list_models()
                out.append({"name": rt.name, "kind": rt.kind, "base_url": rt.base_url, "online": True,
                            "models": [{**m, "ref": f"{rt.name}/{m['name']}"} for m in models]})
            except Exception as exc:  # noqa: BLE001 - an offline runtime is normal
                out.append({"name": rt.name, "kind": rt.kind, "base_url": rt.base_url, "online": False,
                            "error": type(exc).__name__, "models": []})
        return out

    @api.get("/suites")
    def suites() -> list[dict]:
        try:
            return [{"name": s.name, "description": s.description, "category": s.category, "grader": s.grader.type,
                     "examples": len(s.examples), "json_mode": s.params.json_mode, "tools": [t.name for t in s.tools]}
                    for s in list_suites(settings.suites_dir)]
        except SpecError as exc:
            raise HTTPException(422, str(exc)) from exc

    @api.get("/suites/{name}")
    def suite(name: str) -> dict:
        match = [s for s in list_suites(settings.suites_dir) if s.name == name]
        if not match:
            raise HTTPException(404, "suite not found")
        return match[0].model_dump()

    @api.post("/runs", status_code=202)
    def create_run(body: RunIn) -> dict:
        try:
            rid = start_run(p(), body.models, body.suites, body.limit, body.json_mode, body.name)
        except (ValueError, SpecError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return _launch(rid)

    @api.get("/runs")
    def list_runs(limit: int = Query(30, ge=1, le=200)) -> list[dict]:
        return [{k: v for k, v in r.items() if k != "summary"} | {"models": r["config"]["models"]}
                for r in p().store.runs(limit)]

    @api.get("/runs/{rid}")
    def get_run(rid: str) -> dict:
        return _run(rid)

    @api.post("/runs/{rid}/resume", status_code=202)
    def resume(rid: str) -> dict:
        run = _run(rid)
        if run["status"] in ("running", "queued"):
            raise HTTPException(409, "run is still in progress")
        p().store.update_run(rid, status="queued", error=None)
        return _launch(rid)

    @api.delete("/runs/{rid}", status_code=204)
    def delete_run(rid: str):
        if not p().store.delete_run(rid):
            raise HTTPException(404, "run not found")

    @api.get("/runs/{rid}/results")
    def results(rid: str, model: str | None = None, suite: str | None = None) -> list[dict]:
        _run(rid)
        return p().store.results(rid, model, suite)

    @api.get("/leaderboard")
    def leaderboard() -> dict:
        done = [r for r in p().store.runs(200) if r["status"] == "completed" and r["summary"]]
        if not done:
            raise HTTPException(404, "no completed runs yet")
        return {"run_id": done[0]["id"], **done[0]["summary"]}

    return api
