"""Deterministic graders. Each returns a score in [0, 1], pass/fail and a short explanation.

JSON and tool-call reliability are measured separately from correctness:
* json_valid  - the whole reply (minus a ``` fence) parses and matches the schema
* tool_valid  - the model called a tool that exists, with arguments that match its schema
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from .suites import Grader, Suite

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)


@dataclass
class Grade:
    score: float
    passed: bool
    detail: str
    json_valid: bool | None = None
    tool_valid: bool | None = None
    parsed: Any = field(default=None)


def _norm(s: Any, case_sensitive: bool = False) -> str:
    s = " ".join(str(s).split()).strip(" .\"'")
    return s if case_sensitive else s.lower()


def parse_json(text: str) -> tuple[Any, bool]:
    """(value, strictly_valid). Strict = the reply is only JSON (a ``` fence is tolerated).
    Otherwise the first {...} block is recovered so field accuracy can still be scored."""
    body = FENCE.sub(r"\1", text.strip())
    try:
        return json.loads(body), True
    except json.JSONDecodeError:
        pass
    start = body.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(body)):
            depth += {"{": 1, "}": -1}.get(body[i], 0)
            if depth == 0:
                try:
                    return json.loads(body[start:i + 1]), False
                except json.JSONDecodeError:
                    break
        start = body.find("{", start + 1)
    return None, False


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) < 1e-6
    if isinstance(a, str) or isinstance(b, str):
        return _norm(a) == _norm(b)
    return a == b


def grade_text(g: Grader, output: str, expected: Any) -> Grade:
    if g.type == "exact":
        ok = _norm(output, g.case_sensitive) == _norm(expected, g.case_sensitive)
        return Grade(float(ok), ok, "exact match" if ok else f"expected {expected!r}")

    if g.type == "choice":
        found = [c for c in g.choices if re.search(rf"\b{re.escape(c)}\b", output, 0 if g.case_sensitive else re.I)]
        if len(found) != 1:
            return Grade(0.0, False, "no label found" if not found else f"ambiguous: {', '.join(found)}")
        ok = _norm(found[0]) == _norm(expected)
        return Grade(float(ok), ok, f"answered {found[0]!r}" + ("" if ok else f", expected {expected!r}"))

    if g.type == "contains":
        wanted = expected if isinstance(expected, list) else [expected]
        hits = [w for w in wanted if _norm(w) in _norm(output)]
        score = len(hits) / len(wanted)
        missing = [w for w in wanted if w not in hits]
        return Grade(score, score == 1.0, "all found" if not missing else f"missing {missing}")

    if g.type == "regex":
        ok = re.search(expected, output, 0 if g.case_sensitive else re.I) is not None
        return Grade(float(ok), ok, f"/{expected}/ " + ("matched" if ok else "not matched"))

    if g.type == "number":
        nums = NUM.findall(output.replace("$", ""))
        if not nums:
            return Grade(0.0, False, "no number in the answer")
        got = float(nums[-1].replace(",", ""))  # the final number is the answer in a worked solution
        ok = abs(got - float(expected)) <= g.tolerance + 1e-9
        return Grade(float(ok), ok, f"answered {got:g}" + ("" if ok else f", expected {float(expected):g}"))

    if g.type == "json":
        value, strict = parse_json(output)
        if value is None:
            return Grade(0.0, False, "not JSON", json_valid=False)
        schema_ok, why = True, ""
        if g.json_schema:
            try:
                jsonschema.validate(value, g.json_schema)
            except jsonschema.ValidationError as exc:
                schema_ok, why = False, f"schema: {exc.message}"
        fields = g.fields or (list(expected) if isinstance(expected, dict) else [])
        right = [f for f in fields if isinstance(value, dict) and _same(value.get(f), expected.get(f))]
        wrong = [f for f in fields if f not in right]
        score = len(right) / len(fields) if fields else float(schema_ok)
        valid = strict and schema_ok
        notes = [] if strict else ["extra text around the JSON"]
        notes += [why] if why else []
        notes += [f"wrong {', '.join(wrong)}"] if wrong else []
        return Grade(score, valid and score == 1.0, "; ".join(notes) or "valid, all fields correct",
                     json_valid=valid, parsed=value)
    raise ValueError(f"unknown grader {g.type}")


def grade_tool_call(suite: Suite, tool_calls: list[dict], expected: dict) -> Grade:
    want_name = expected.get("name")
    if want_name is None:  # the right answer is not to call a tool
        ok = not tool_calls
        return Grade(float(ok), ok, "correctly answered without a tool" if ok else
                     f"called {tool_calls[0]['name']} unnecessarily", tool_valid=None)
    if not tool_calls:
        return Grade(0.0, False, "no tool call", tool_valid=False)
    call = tool_calls[0]
    tools = {t.name: t for t in suite.tools}
    valid, why = call["name"] in tools, "" if call["name"] in tools else f"unknown tool {call['name']}"
    if valid:
        try:
            jsonschema.validate(call["args"], tools[call["name"]].parameters)
        except jsonschema.ValidationError as exc:
            valid, why = False, f"invalid args: {exc.message}"
    if call["name"] != want_name:
        return Grade(0.0, False, f"called {call['name']}, expected {want_name}", tool_valid=valid)
    want_args = expected.get("arguments", {})
    right = [k for k, v in want_args.items() if _same(call["args"].get(k), v)]
    wrong = [f"{k}={call['args'].get(k)!r}" for k in want_args if k not in right]
    score = 0.5 + 0.5 * (len(right) / len(want_args) if want_args else 1.0)
    notes = ([why] if why else []) + ([f"wrong {', '.join(wrong)}"] if wrong else [])
    return Grade(score, valid and score == 1.0, "; ".join(notes) or f"{want_name} with correct arguments",
                 tool_valid=valid, parsed=call)


def grade(suite: Suite, example: dict, output: str, tool_calls: list[dict] | None = None) -> Grade:
    if suite.grader.type == "tool_call":
        return grade_tool_call(suite, tool_calls or [], example["expected"])
    return grade_text(suite.grader, output, example["expected"])
