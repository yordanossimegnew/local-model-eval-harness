from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.agent import build_components  # noqa: E402
from src.models import Runtime  # noqa: E402
from src.utils.config import Settings  # noqa: E402
from tests.fake_runtime import FakeRuntime  # noqa: E402

TEST_DB = os.getenv("TEST_DATABASE_URL")
OPENAI_MODELS = {"local-coder:3b-q4"}


@pytest.fixture
def fake():
    f = FakeRuntime(ROOT / "suites", openai_models=set(OPENAI_MODELS), time_scale=0).start()
    yield f
    f.stop()


@pytest.fixture
def settings(tmp_path):
    if not TEST_DB:
        pytest.skip("TEST_DATABASE_URL not set")
    eng = create_engine(TEST_DB.replace("postgresql://", "postgresql+psycopg://", 1))
    with eng.begin() as c:
        if c.execute(text("SELECT to_regclass('runs')")).scalar():
            c.execute(text("TRUNCATE runs, results, model_stats CASCADE"))
    eng.dispose()
    return Settings(database_url=TEST_DB, suites_dir=ROOT / "suites", log_dir=tmp_path, sample_interval=0.05)


@pytest.fixture
def parts(settings, fake):
    runtimes = [Runtime(name="ollama", kind="ollama", base_url=fake.base),
                Runtime(name="llamacpp", kind="openai", base_url=fake.base + "/v1"),
                Runtime(name="offline", kind="openai", base_url="http://127.0.0.1:9/v1")]
    return build_components(settings, runtimes=runtimes, checkpointer=None)
