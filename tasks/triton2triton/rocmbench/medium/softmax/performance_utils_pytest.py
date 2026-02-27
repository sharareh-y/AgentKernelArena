import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass
class BenchConfig:
    warm_up: int = 10
    repetition: int = 100


def do_bench_config(warm_up: int = 10, repetition: int = 100) -> BenchConfig:
    """Create a benchmark configuration object compatible with existing task code."""
    return BenchConfig(warm_up=max(0, int(warm_up)), repetition=max(1, int(repetition)))


_BENCHMARK_RESULTS: list[dict[str, Any]] = []


def _sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_times(callable_fn: Callable[[], Any], config: BenchConfig) -> list[float]:
    """Run warmup + measured iterations and return list of times in ms."""
    for _ in range(config.warm_up):
        callable_fn()
    _sync_if_needed()

    times_ms: list[float] = []
    for _ in range(config.repetition):
        _sync_if_needed()
        start = time.perf_counter()
        callable_fn()
        _sync_if_needed()
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)
    return times_ms


def _compute_timing_stats(times_ms: list[float], config: BenchConfig) -> dict[str, Any]:
    """Compute mean, median, p90, min, max from a list of times."""
    times_sorted = sorted(times_ms)
    n = len(times_sorted)
    return {
        "mean": sum(times_sorted) / n,
        "median": times_sorted[n // 2],
        "p90": times_sorted[min(n - 1, int(round(0.9 * (n - 1))))],
        "min": times_sorted[0],
        "max": times_sorted[-1],
        "repetition": config.repetition,
        "warm_up": config.warm_up,
    }


class PytestBenchmarker:
    """Simple benchmark helper used by rocmbench pytest performance tests."""

    def __init__(self, op_callable: Callable[[], Any], op_name: str, config: BenchConfig) -> None:
        self.op_callable = op_callable
        self.op_name = op_name
        self.config = config

    def run_benchmark(
        self,
        current_params_dict: dict[str, Any],
        gbps_calculator: Callable[[dict[str, Any], float], float] | None = None,
        tflops_calculator: Callable[[dict[str, Any], float], float] | None = None,
        baseline_callable: Callable[[], Any] | None = None,
    ) -> dict[str, Any]:
        # Measure the main (optimized/triton) operation.
        times_ms = _measure_times(self.op_callable, self.config)
        timing_stats = _compute_timing_stats(times_ms, self.config)
        mean_ms = timing_stats["mean"]

        result: dict[str, Any] = {
            "op_name": self.op_name,
            "params": current_params_dict,
            "timing_ms": timing_stats,
        }

        if gbps_calculator is not None:
            try:
                result["gbps"] = float(gbps_calculator(current_params_dict, mean_ms))
            except Exception as exc:
                result["gbps_error"] = str(exc)
        if tflops_calculator is not None:
            try:
                result["tflops"] = float(tflops_calculator(current_params_dict, mean_ms))
            except Exception as exc:
                result["tflops_error"] = str(exc)

        # Measure baseline (e.g. PyTorch reference) if provided.
        if baseline_callable is not None:
            baseline_times = _measure_times(baseline_callable, self.config)
            baseline_stats = _compute_timing_stats(baseline_times, self.config)
            result["baseline_timing_ms"] = baseline_stats
            baseline_mean = baseline_stats["mean"]
            if mean_ms > 0:
                result["speedup_ratio"] = baseline_mean / mean_ms
            else:
                result["speedup_ratio"] = 1.0

        _BENCHMARK_RESULTS.append(result)
        return result


def save_all_benchmark_results(output_directory: str) -> None:
    """Persist collected benchmark entries to a single JSON file."""
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_results.json"
    out_path.write_text(json.dumps(_BENCHMARK_RESULTS, indent=2, sort_keys=True), encoding="utf-8")
