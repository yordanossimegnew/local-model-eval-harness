"""Eval suites: a YAML header (prompt, grader, params) + JSONL examples, one folder per suite.

    suites/<name>/suite.yaml
    suites/<name>/examples.jsonl     one {"id": ..., <template vars>..., "expected": ...} per line
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

VAR = re.compile(r"\{\{\s*(\w+)\s*\}\}")


class SpecError(ValueError):
    pass


class Tool(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema

    def as_openai(self) -> dict:
        return {"type": "function", "function": self.model_dump()}


class Grader(BaseModel):
    type: Literal["exact", "choice", "contains", "regex", "number", "json", "tool_call"]
    choices: list[str] = Field(default_factory=list, description="choice: allowed labels")
    tolerance: float = Field(0.0, description="number: absolute tolerance")
    json_schema: dict | None = Field(None, description="json: schema the output must satisfy")
    fields: list[str] = Field(default_factory=list, description="json: fields compared to expected (partial credit)")
    case_sensitive: bool = False


class Params(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 256
    json_mode: bool = Field(False, description="Ask the runtime for constrained JSON (Ollama format=json / "
                                                "OpenAI response_format). Off = measure raw JSON reliability.")


class Suite(BaseModel):
    name: str
    description: str = ""
    category: str = "accuracy"           # accuracy | json | tools
    system: str = ""
    template: str
    grader: Grader
    params: Params = Field(default_factory=Params)
    tools: list[Tool] = Field(default_factory=list)
    examples: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def check(self):
        if self.grader.type == "tool_call" and not self.tools:
            raise ValueError("a tool_call suite needs tools")
        if self.grader.type == "choice" and not self.grader.choices:
            raise ValueError("a choice grader needs choices")
        ids = [e.get("id") for e in self.examples]
        if any(i is None for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("every example needs a unique id")
        for e in self.examples:
            if "expected" not in e:
                raise ValueError(f"example {e['id']} has no 'expected'")
            missing = set(self.variables) - set(e)
            if missing:
                raise ValueError(f"example {e['id']} is missing {sorted(missing)}")
        return self

    @property
    def variables(self) -> list[str]:
        return sorted(set(VAR.findall(self.template)) | set(VAR.findall(self.system)))

    def render(self, example: dict) -> tuple[str, str]:
        sub = lambda t: VAR.sub(lambda m: str(example[m.group(1)]), t)  # noqa: E731
        return sub(self.system), sub(self.template)

    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json(exclude={"description"}).encode()).hexdigest()[:12]


def load_suite(path: Path) -> Suite:
    folder = path if path.is_dir() else path.parent
    try:
        header = yaml.safe_load((folder / "suite.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SpecError(f"{folder.name}/suite.yaml: {exc}") from exc
    examples = []
    ex_file = folder / "examples.jsonl"
    if ex_file.exists():
        for n, line in enumerate(ex_file.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SpecError(f"{folder.name}/examples.jsonl line {n}: {exc}") from exc
    try:
        return Suite(**{"name": folder.name, **header, "examples": examples})
    except ValidationError as exc:
        raise SpecError(f"{folder.name}: {exc}") from exc


def list_suites(suites_dir: Path) -> list[Suite]:
    return [load_suite(p) for p in sorted(suites_dir.iterdir()) if (p / "suite.yaml").exists()]
