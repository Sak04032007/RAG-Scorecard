import time
from contextlib import contextmanager

class LatencyCostTracker:
    def __init__(self):
        self.timings = {}
        self.input_cost_per_token = 0.150 / 1_000_000
        self.output_cost_per_token = 0.600 / 1_000_000
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @contextmanager
    def track_stage(self, stage_name: str):
        start_time = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - start_time) * 1000
        self.timings[stage_name] = self.timings.get(stage_name, 0.0) + elapsed

    def add_tokens(self, input_tokens: int = 0, output_tokens: int = 0):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def get_estimated_cost(self) -> float:
        return (self.total_input_tokens * self.input_cost_per_token) + (self.total_output_tokens * self.output_cost_per_token)
