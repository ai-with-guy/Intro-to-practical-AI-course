"""Model-only OpenVINO latency and throughput benchmarks."""

import gc
from time import perf_counter

import numpy as np
import openvino as ov


def benchmark_model(core, model_path, device, tensor, iterations=200, warmup=10):
    """Compare batch-one latency and concurrent throughput on an explicit device.

    Inputs must already be preprocessed. Timing includes OpenVINO submission and
    transfers, but excludes compilation, image preprocessing and video I/O.
    Separate LATENCY and THROUGHPUT compilations use each device's recommended
    settings.
    """
    if iterations < 1 or warmup < 1:
        raise ValueError('iterations and warmup must be positive')
    if tensor.shape[0] != 1:
        raise ValueError('Use batch size 1 for this comparison')

    gc.collect()
    model = core.read_model(model_path)
    model.reshape({0: list(tensor.shape)})
    inputs = {0: np.ascontiguousarray(tensor)}

    compiled = core.compile_model(model, device, {'PERFORMANCE_HINT': 'LATENCY'})
    request = compiled.create_infer_request()
    for _ in range(warmup):
        request.infer(inputs)
    latencies = []
    for _ in range(iterations):
        start = perf_counter()
        request.infer(inputs)
        latencies.append((perf_counter() - start) * 1000)
    del request, compiled
    gc.collect()

    # Keep batch size one: GPU automatic batching can otherwise wait for a
    # partially filled batch and distort short benchmark runs.
    compiled = core.compile_model(model, device, {
        'PERFORMANCE_HINT': 'THROUGHPUT', 'ALLOW_AUTO_BATCHING': False,
    })
    queue = ov.AsyncInferQueue(compiled, jobs=0)
    requests = len(queue)
    for _ in range(max(warmup, requests)):
        queue.start_async(inputs)
    queue.wait_all()
    start = perf_counter()
    for _ in range(iterations):
        queue.start_async(inputs)
    queue.wait_all()  # Include completion, not just submission time.
    elapsed = perf_counter() - start
    del queue, compiled, model

    return {
        'latency_ms': float(np.median(latencies)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'throughput_ips': iterations / elapsed,
        'requests': requests,
    }
