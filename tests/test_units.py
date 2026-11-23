from __future__ import annotations

import json

import pytest

from src.tools import load_suite, parse_json, wilson
from src.tools.graders import grade_text, grade_tool_call
from src.tools.suites import Grader, SpecError, Suite


def test_choice_grader_handles_case_and_ambiguity():
    g = Grader(type="choice", choices=["positive", "negative", "neutral"])
    assert grade_text(g, "Positive.", "positive").passed
    assert not grade_text(g, "negative", "positive").passed
    r = grade_text(g, "It is positive or maybe negative", "positive")
    assert not r.passed and "ambiguous" in r.detail
    assert "no label" in grade_text(g, "dunno", "positive").detail


def test_number_grader_takes_the_last_number():
    g = Grader(type="number", tolerance=0.01)
    assert grade_text(g, "7 x 3 = 21, 50 - 21 = 29\nFinal answer: $29", 29).passed
    assert grade_text(g, "about 85.714", 85.71).passed
    assert not grade_text(g, "1,200 / 40 = 31", 30).passed
    assert grade_text(g, "The total is 1,240", 1240).passed


def test_contains_and_regex():
    g = Grader(type="contains")
    r = grade_text(g, "The southern trail, 14 km.", ["southern", "14"])
    assert r.passed and r.score == 1
    r = grade_text(g, "The southern trail.", ["southern", "14"])
    assert r.score == 0.5 and "14" in r.detail
    assert grade_text(Grader(type="regex"), "Ticket 2026-114", r"\d{4}-\d{3}").passed


def test_json_grader_separates_validity_from_field_accuracy():
    schema = {"type": "object", "required": ["a", "b"], "properties": {"a": {"type": "string"}, "b": {"type": "number"}}}
    g = Grader(type="json", fields=["a", "b"], json_schema=schema)
    exp = {"a": "X", "b": 2}
    ok = grade_text(g, '```json\n{"a": "x", "b": 2.0}\n```', exp)
    assert ok.passed and ok.json_valid  # fenced JSON is fine, string compare is case-insensitive
    chatty = grade_text(g, 'Sure! {"a": "X", "b": 2} Hope that helps.', exp)
    assert chatty.score == 1 and not chatty.json_valid and not chatty.passed
    wrong_type = grade_text(g, '{"a": "X", "b": "2"}', exp)
    assert not wrong_type.json_valid and "schema" in wrong_type.detail
    assert grade_text(g, "no json here", exp).json_valid is False
    assert parse_json('x {"a": {"b": 1}} y') == ({"a": {"b": 1}}, False)


def _tool_suite():
    return Suite(name="t", template="{{q}}", grader=Grader(type="tool_call"), tools=[
        {"name": "get_weather", "description": "w", "parameters": {
            "type": "object", "required": ["city"], "properties": {"city": {"type": "string"}}}},
        {"name": "convert_currency", "description": "c", "parameters": {
            "type": "object", "required": ["amount"], "properties": {"amount": {"type": "number"}}}}])


def test_tool_call_grader():
    s = _tool_suite()
    exp = {"name": "get_weather", "arguments": {"city": "Lisbon"}}
    assert grade_tool_call(s, [{"name": "get_weather", "args": {"city": "lisbon"}}], exp).passed
    r = grade_tool_call(s, [{"name": "get_weather", "args": {"city": "Porto"}}], exp)
    assert r.score == 0.5 and r.tool_valid and not r.passed
    r = grade_tool_call(s, [{"name": "convert_currency", "args": {"amount": "two"}}],
                        {"name": "convert_currency", "arguments": {"amount": 2}})
    assert r.tool_valid is False and "invalid args" in r.detail
    assert grade_tool_call(s, [], exp).tool_valid is False
    assert grade_tool_call(s, [{"name": "delete_db", "args": {}}], exp).tool_valid is False
    assert grade_tool_call(s, [], {"name": None}).passed
    assert not grade_tool_call(s, [{"name": "get_weather", "args": {"city": "x"}}], {"name": None}).passed


def test_suite_validation(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "suite.yaml").write_text("template: '{{q}}'\ngrader: {type: exact}\n", encoding="utf-8")
    (d / "examples.jsonl").write_text(json.dumps({"id": 1, "q": "a", "expected": "b"}) + "\n" +
                                      json.dumps({"id": 1, "q": "c", "expected": "d"}) + "\n", encoding="utf-8")
    with pytest.raises(SpecError, match="unique id"):
        load_suite(d)
    (d / "examples.jsonl").write_text(json.dumps({"id": 1, "expected": "b"}) + "\n", encoding="utf-8")
    with pytest.raises(SpecError, match="missing"):
        load_suite(d)
    (d / "examples.jsonl").write_text("{not json\n", encoding="utf-8")
    with pytest.raises(SpecError, match="line 1"):
        load_suite(d)


def test_wilson_interval():
    lo, hi = wilson(8, 10)
    assert round(lo, 2) == 0.49 and round(hi, 2) == 0.94
    assert wilson(0, 0) == (0.0, 0.0)
