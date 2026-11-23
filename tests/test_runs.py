"""Full eval runs through LangGraph against scripted Ollama and OpenAI-compatible servers, on Postgres."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.agent import execute, start_run
from src.api import create_app

MODELS = ["ollama/tiny-chat:0.5b", "ollama/midsize-instruct:4b", "llamacpp/local-coder:3b-q4"]


def run(parts, models=MODELS, **kw):
    return execute(parts, start_run(parts, models, **kw))


def board(result):
    return {b["model"]: b for b in result["summary"]["leaderboard"]}


def test_full_run_builds_a_leaderboard(parts):
    result = run(parts)
    assert result["status"] == "completed"
    s = result["summary"]
    assert len(s["suites"]) == 5 and len(parts.store.results(result["id"])) == 3 * 48
    b = board(result)
    assert s["leaderboard"][0]["model"] == "ollama/midsize-instruct:4b"           # most accurate first
    assert b["ollama/tiny-chat:0.5b"]["accuracy"] < b["ollama/midsize-instruct:4b"]["accuracy"]
    assert b["ollama/tiny-chat:0.5b"]["json_valid_rate"] < b["ollama/midsize-instruct:4b"]["json_valid_rate"]
    assert b["ollama/tiny-chat:0.5b"]["tool_valid_rate"] < 1.0
    # Ollama's own eval stats give exact decode speed (time_scale=0 in tests, so wall clock can't)
    assert round(b["ollama/tiny-chat:0.5b"]["tokens_per_sec"]) == 48
    assert round(b["ollama/midsize-instruct:4b"]["tokens_per_sec"]) == 11
    assert b["ollama/midsize-instruct:4b"]["resources"]["model_mb"] == 2490      # from /api/ps
    assert "host_ram_peak_mb" in b["llamacpp/local-coder:3b-q4"]["resources"]
    assert b["llamacpp/local-coder:3b-q4"]["load_ms"] is not None
    cell = s["cells"]["ollama/midsize-instruct:4b|sentiment"]
    assert cell["n"] == 12 and 0 <= cell["pass_ci"][0] <= cell["pass_rate"] <= cell["pass_ci"][1] <= 1


def test_results_keep_outputs_and_reasons(parts):
    result = run(parts, ["ollama/tiny-chat:0.5b"], suites=["invoice-extraction", "tool-calling"])
    rows = parts.store.results(result["id"])
    chatty = [r for r in rows if r["suite"] == "invoice-extraction" and r["json_valid"] is False]
    assert chatty and any("extra text" in r["detail"] for r in chatty)
    calls = [r for r in rows if r["suite"] == "tool-calling" and r["tool_calls"]]
    assert calls and all("name" in r["tool_calls"][0] for r in calls)
    assert all(r["latency_ms"] is not None and r["output_tokens"] for r in rows)


def test_limit_and_suite_selection(parts):
    result = run(parts, ["ollama/tiny-chat:0.5b"], suites=["sentiment"], limit=3)
    assert len(parts.store.results(result["id"])) == 3
    assert result["summary"]["suites"] == ["sentiment"]


def test_unavailable_models_are_reported_not_fatal(parts):
    result = run(parts, ["ollama/tiny-chat:0.5b", "ollama/not-pulled:7b", "offline/whatever", "nope/x"], limit=1)
    assert result["status"] == "completed"
    un = result["summary"]["unavailable"]
    assert "not available" in un["ollama/not-pulled:7b"]
    assert "unreachable" in un["offline/whatever"] and "no runtime" in un["nope/x"]
    assert [b["model"] for b in result["summary"]["leaderboard"]] == ["ollama/tiny-chat:0.5b"]


def test_all_models_unavailable_fails_the_run(parts):
    result = run(parts, ["offline/whatever"])
    assert result["status"] == "failed" and "unreachable" in result["error"]


def test_model_errors_are_recorded_and_resume_only_redoes_them(parts, fake):
    rid = start_run(parts, ["ollama/tiny-chat:0.5b", "ollama/midsize-instruct:4b"], suites=["sentiment"])
    fake.profiles["midsize-instruct:4b"].loaded = True       # skip warm-up failure: fail mid-run instead
    fake.fail_models.add("midsize-instruct:4b")
    result = execute(parts, rid)
    stats = parts.store.model_stats(rid)
    assert stats["ollama/midsize-instruct:4b"]["error"]      # warm-up failed -> recorded, run continues
    assert board(result)["ollama/tiny-chat:0.5b"]["n"] == 12

    fake.fail_models.clear()
    calls = fake.calls
    result = execute(parts, rid)                              # resume: tiny's 12 results are reused
    assert fake.calls - calls == 1 + 12                       # warm-up + midsize's 12 examples only
    assert board(result)["ollama/midsize-instruct:4b"]["n"] == 12
    assert len(parts.store.results(rid)) == 24


def test_json_mode_is_passed_to_the_runtime(parts, fake, monkeypatch):
    seen = []
    from src.models import runtimes
    real = runtimes.RuntimeClient.chat

    def spy(self, model, temperature, max_tokens, json_mode=False):
        seen.append(json_mode)
        return real(self, model, temperature, max_tokens, json_mode)
    monkeypatch.setattr(runtimes.RuntimeClient, "chat", spy)
    run(parts, ["ollama/tiny-chat:0.5b"], suites=["invoice-extraction", "sentiment"], limit=1, json_mode=True)
    assert seen == [False, True, False]  # warm-up, invoice (json grader) on, sentiment off


def test_api_flow(settings, parts):
    with TestClient(create_app(settings, parts=parts, run_inline=True)) as c:
        rts = {r["name"]: r for r in c.get("/runtimes").json()}
        assert rts["ollama"]["online"] and not rts["offline"]["online"]
        assert "ollama/tiny-chat:0.5b" in [m["ref"] for m in rts["ollama"]["models"]]
        assert "llamacpp/local-coder:3b-q4" in [m["ref"] for m in rts["llamacpp"]["models"]]
        assert len(c.get("/suites").json()) == 5
        assert c.get("/suites/tool-calling").json()["tools"][0]["name"] == "get_weather"

        r = c.post("/runs", json={"models": MODELS[:2], "limit": 2})
        assert r.status_code == 202, r.text
        run_ = r.json()
        assert run_["status"] == "completed" and run_["progress"]["done"] == run_["progress"]["total"] == 20
        rows = c.get(f"/runs/{run_['id']}/results", params={"model": MODELS[0], "suite": "sentiment"}).json()
        assert len(rows) == 2 and rows[0]["prompt"]
        lb = c.get("/leaderboard").json()
        assert lb["run_id"] == run_["id"] and len(lb["leaderboard"]) == 2
        assert c.get("/runs").json()[0]["models"] == MODELS[:2]

        assert c.post("/runs", json={"models": MODELS[:1], "suites": ["nope"]}).status_code == 422
        assert c.post("/runs", json={"models": []}).status_code == 422
        assert c.post(f"/runs/{run_['id']}/resume").json()["status"] == "completed"
        assert c.delete(f"/runs/{run_['id']}").status_code == 204
        assert c.get(f"/runs/{run_['id']}").status_code == 404


def test_cli_run_and_report(parts, monkeypatch, capsys):
    import main
    monkeypatch.setattr("src.agent.build_components", lambda s: parts)
    assert main.main(["run", "ollama/midsize-instruct:4b", "--suite", "sentiment", "--limit", "4"]) == 0
    out = capsys.readouterr().out
    assert "ollama/midsize-instruct:4b" in out and "100%" in out
    assert main.main(["run", "offline/x"]) == 1
    rid = parts.store.runs(5)[1]["id"]
    assert main.main(["report", rid]) == 0
