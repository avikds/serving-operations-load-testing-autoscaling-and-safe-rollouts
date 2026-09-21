"""
Serving Operations: Load Testing, Autoscaling and Safe Rollouts

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - request_metrics
import math
import numpy as np

def request_metrics(t_arrive, token_times):
    """Compute serving metrics for a single request."""
    n = len(token_times)

    if n == 0:
        raise ValueError("token_times must contain at least one token time.")

    first_token = token_times[0]
    last_token = token_times[-1]

    # Time to first token and end-to-end latency.
    ttft = first_token - t_arrive
    e2e = last_token - t_arrive

    # A single token has no inter-token gap or generation interval.
    if n == 1:
        itl_mean = 0.0
        tps = 0.0
    else:
        gaps = np.diff(token_times)
        itl_mean = float(np.mean(gaps))

        generation_time = last_token - first_token
        tps = float((n - 1) / generation_time)

    return {
        "ttft": float(ttft),
        "e2e": float(e2e),
        "itl_mean": itl_mean,
        "tps": tps,
    }


def percentile(values, p):
    """Return the percentile using the book's one-based ceil rule."""
    if not 0 < p <= 1:
        raise ValueError("p must satisfy 0 < p <= 1.")

    if len(values) == 0:
        raise ValueError("values must contain at least one element.")

    sorted_values = sorted(values)

    # One-based index = ceil(p * n), converted to zero-based indexing.
    index = math.ceil(p * len(sorted_values)) - 1

    return sorted_values[index]


def latency_summary(values):
    """Return the requested percentile and mean latency summary."""
    if len(values) == 0:
        raise ValueError("values must contain at least one element.")

    return {
        "p50": float(percentile(values, 0.50)),
        "p90": float(percentile(values, 0.90)),
        "p95": float(percentile(values, 0.95)),
        "p99": float(percentile(values, 0.99)),
        "mean": float(np.mean(values)),
    }

