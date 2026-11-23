"""Turn raw results into the leaderboard: model x suite scores, reliability, speed and resources."""

from __future__ import annotations

from collections import defaultdict

from ..tools.stats import mean, percentile, wilson


def _cell(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    passes = sum(r["passed"] for r in rows)
    lo, hi = wilson(passes, len(rows))
    jv = [r["json_valid"] for r in rows if r["json_valid"] is not None]
    tv = [r["tool_valid"] for r in rows if r["tool_valid"] is not None]
    return {
        "n": len(rows), "errors": len(rows) - len(ok),
        "accuracy": mean([r["score"] for r in rows]) or 0.0,
        "pass_rate": passes / len(rows) if rows else 0.0, "pass_ci": [lo, hi],
        "json_valid_rate": sum(jv) / len(jv) if jv else None,
        "tool_valid_rate": sum(tv) / len(tv) if tv else None,
        "latency_p50_ms": percentile([r["latency_ms"] for r in ok], 0.5),
        "latency_p95_ms": percentile([r["latency_ms"] for r in ok], 0.95),
        "ttft_p50_ms": percentile([r["ttft_ms"] for r in ok], 0.5),
        "tokens_per_sec": percentile([r["tokens_per_sec"] for r in ok], 0.5),
        "output_tokens_mean": mean([r["output_tokens"] for r in ok]),
    }


def build_summary(results: list[dict], stats: dict[str, dict], suite_categories: dict[str, str]) -> dict:
    by = defaultdict(list)
    for r in results:
        by[(r["model"], r["suite"])].append(r)
    models = sorted({r["model"] for r in results})
    suites = sorted({r["suite"] for r in results})
    cells = {f"{m}|{s}": _cell(by[(m, s)]) for m in models for s in suites if by[(m, s)]}
    board = []
    for m in models:
        rows = [r for r in results if r["model"] == m]
        per_suite = [cells[f"{m}|{s}"] for s in suites if f"{m}|{s}" in cells]
        overall = _cell(rows)
        st = stats.get(m, {})
        board.append({
            "model": m,
            "accuracy": mean([c["accuracy"] for c in per_suite]) or 0.0,  # macro average: every suite counts equally
            "pass_rate": overall["pass_rate"], "pass_ci": overall["pass_ci"], "n": overall["n"],
            "errors": overall["errors"],
            "json_valid_rate": overall["json_valid_rate"], "tool_valid_rate": overall["tool_valid_rate"],
            "tokens_per_sec": overall["tokens_per_sec"], "ttft_p50_ms": overall["ttft_p50_ms"],
            "latency_p50_ms": overall["latency_p50_ms"], "latency_p95_ms": overall["latency_p95_ms"],
            "load_ms": st.get("load_ms"), "resources": st.get("resources") or {}, "load_error": st.get("error"),
        })
    board.sort(key=lambda b: (-b["accuracy"], -(b["tokens_per_sec"] or 0)))
    return {"models": models, "suites": suites, "categories": {s: suite_categories.get(s, "accuracy") for s in suites},
            "cells": cells, "leaderboard": board}
