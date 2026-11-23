"""Wire up Postgres, the runtimes from runtimes.yaml and the checkpointer."""

from __future__ import annotations

from typing import Any

from ..models.runtimes import Runtime, load_runtimes
from ..tools.store import Store
from ..utils.config import Settings
from .agent import Components
from .memory import open_checkpointer


def build_components(settings: Settings, runtimes: list[Runtime] | None = None, checkpointer: Any = "postgres") -> Components:
    store = Store(settings.database_url)
    store.init()
    rts = runtimes if runtimes is not None else load_runtimes(settings.runtimes_file)
    if checkpointer == "postgres":
        checkpointer = open_checkpointer(settings.database_url)
    return Components(settings, store, {r.name: r for r in rts}, checkpointer)
