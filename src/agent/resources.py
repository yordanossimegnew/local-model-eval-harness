"""Samples host CPU and RAM while a model is being evaluated (meaningful when the runtime is local)."""

from __future__ import annotations

import threading

import psutil


class ResourceSampler:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self._stop = threading.Event()
        self.cpu: list[float] = []
        self.ram_used: list[float] = []
        self.baseline_mb = psutil.virtual_memory().used / 2**20
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        psutil.cpu_percent(None)
        while not self._stop.wait(self.interval):
            self.cpu.append(psutil.cpu_percent(None))
            self.ram_used.append(psutil.virtual_memory().used / 2**20)

    def __enter__(self) -> "ResourceSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def summary(self) -> dict:
        if not self.ram_used:
            return {}
        peak = max(self.ram_used)
        return {"host_cpu_avg_pct": round(sum(self.cpu) / len(self.cpu), 1), "host_cpu_peak_pct": round(max(self.cpu), 1),
                "host_ram_peak_mb": round(peak), "host_ram_delta_mb": round(max(0.0, peak - self.baseline_mb)),
                "host_ram_total_mb": round(psutil.virtual_memory().total / 2**20)}
