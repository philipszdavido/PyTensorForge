import time
from collections import defaultdict


class Profiler:

    def __init__(self):
        self.totals = defaultdict(float)
        self.window = defaultdict(float)

    def add(self, name, seconds):
        self.totals[name] += seconds
        self.window[name] += seconds

    def section(self, name):
        return _Section(self, name)

    def take_window(self):
        snapshot = dict(self.window)
        self.window.clear()
        return snapshot

    def summary(self):
        total = sum(self.totals.values()) or 1.0
        return {k: {"seconds": v, "share": v / total} for k, v in sorted(self.totals.items())}


class _Section:

    def __init__(self, profiler, name):
        self.profiler = profiler
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *exc):
        self.profiler.add(self.name, time.perf_counter() - self.start)
