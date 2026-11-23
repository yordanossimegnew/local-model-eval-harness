"""Local Model Eval Harness: FastAPI app (`uvicorn main:app`) and CLI.

    python main.py models                                   # what each runtime serves
    python main.py suites                                   # validate and list suites
    python main.py run ollama/qwen2.5:0.5b ollama/llama3.2:1b --limit 5
    python main.py report <run_id>                          # print a finished run's leaderboard
    python main.py serve
"""

from __future__ import annotations

import argparse
import json
import sys

from src.utils.config import load_settings


def __getattr__(name):  # `uvicorn main:app` builds the app lazily
    if name == "app":
        from src.api import create_app
        return create_app()
    raise AttributeError(name)


def _fmt(v, pct=False, unit=""):
    if v is None:
        return "–"
    return f"{v:.0%}" if pct else f"{v:,.1f}{unit}"


def print_board(summary: dict) -> None:
    head = (f"{'model':34} {'acc':>5} {'pass':>5} {'json':>5} {'tools':>5} {'tok/s':>7} {'TTFT':>9} "
            f"{'p50':>9} {'load':>9}")
    print(head + "\n" + "-" * len(head))
    for b in summary["leaderboard"]:
        print(f"{b['model'][:34]:34} {_fmt(b['accuracy'], True):>5} {_fmt(b['pass_rate'], True):>5} "
              f"{_fmt(b['json_valid_rate'], True):>5} {_fmt(b['tool_valid_rate'], True):>5} "
              f"{_fmt(b['tokens_per_sec']):>7} {_fmt(b['ttft_p50_ms'], unit='ms'):>9} "
              f"{_fmt(b['latency_p50_ms'], unit='ms'):>9} {_fmt(b['load_ms'], unit='ms'):>9}")
    for m, why in (summary.get("unavailable") or {}).items():
        print(f"skipped {m}: {why}")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(prog="local-eval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    sub.add_parser("models", help="list the models each runtime serves")
    sub.add_parser("suites", help="validate and list suites")
    run = sub.add_parser("run", help="evaluate models")
    run.add_argument("models", nargs="+", help="<runtime>/<model>")
    run.add_argument("--suite", action="append", dest="suites")
    run.add_argument("--limit", type=int)
    run.add_argument("--json-mode", choices=["on", "off"])
    run.add_argument("--json", action="store_true", help="print the summary as JSON")
    rep = sub.add_parser("report")
    rep.add_argument("run_id")
    args = parser.parse_args(argv)
    settings = load_settings()

    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("main:app", host=args.host, port=args.port)
        return 0
    if args.cmd == "suites":
        from src.tools import SpecError, list_suites
        try:
            for s in list_suites(settings.suites_dir):
                print(f"✓ {s.name:24} {s.category:9} {s.grader.type:10} {len(s.examples)} examples")
        except SpecError as exc:
            print(f"✗ {exc}", file=sys.stderr)
            return 2
        return 0

    from src.agent import build_components, execute, start_run
    from src.models import RuntimeClient
    from src.utils import setup_logging
    setup_logging(settings.log_level, settings.log_dir)
    parts = build_components(settings)

    if args.cmd == "models":
        for rt in parts.runtimes.values():
            try:
                models = RuntimeClient(rt, 5).list_models()
                print(f"{rt.name} ({rt.kind}, {rt.base_url}): {len(models)} models")
                for m in models:
                    extra = f"  {m['parameters']} {m['quantization']}" if m["parameters"] else ""
                    print(f"  {rt.name}/{m['name']}{extra}")
            except Exception as exc:  # noqa: BLE001
                print(f"{rt.name} ({rt.base_url}): offline ({type(exc).__name__})")
        return 0
    if args.cmd == "run":
        json_mode = None if args.json_mode is None else args.json_mode == "on"
        try:
            rid = start_run(parts, args.models, args.suites, args.limit, json_mode)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        print(f"run {rid}: evaluating {len(args.models)} model(s)…", file=sys.stderr)
        result = execute(parts, rid)
        if result["status"] != "completed":
            print(f"run failed: {result['error']}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result["summary"], indent=2))
        else:
            print_board(result["summary"])
        return 0
    if args.cmd == "report":
        run = parts.store.run(args.run_id)
        if not run or not run["summary"]:
            print("no finished run with that id", file=sys.stderr)
            return 2
        print_board(run["summary"])
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
