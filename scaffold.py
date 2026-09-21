"""
Serving Operations: Load Testing, Autoscaling and Safe Rollouts scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Serving Operations: Load Testing, Autoscaling and Safe Rollouts (Inference Engineering, chapters 4.5 and 7).

Story: measure one replica honestly under Poisson and bursty load and find its knee;
price a cold start stage by stage; tune an autoscaler on a spike and a daily wave;
gate a canary rollout; bill the fleet; watch a rolling window raise alerts; and
harden the client with jittered backoff and a streaming parser.
"""
import numpy as np


def main() -> None:
    slo = {"ttft": 0.5, "itl": 0.05}
    make_sim = lambda: ReplicaSim(max_batch=64, prefill_tokens_per_s=20000, decode_ms_base=15, decode_ms_per_seq=0.25)

    # ---- 1. Measure ----
    print("one replica, 60 s of traffic at 6 req/s, objective TTFT <= 500 ms and ITL <= 50 ms")
    for pattern in ("constant", "poisson", "bursty"):
        rng = np.random.default_rng(0)
        reqs = request_mix(arrival_times(6.0, 60, pattern, rng), rng)
        print(f"  {pattern:9s} " + format_benchmark(run_benchmark(make_sim(), reqs, slo)))
    curve = throughput_latency_curve(make_sim, [2, 4, 6, 8, 10, 12, 16, 24], 60, slo)
    print("throughput-latency curve (poisson arrivals):")
    for line in format_curve(curve):
        print("  " + line)
    knee = capacity_knee(curve)
    print(f"  capacity knee: {knee} req/s per replica at 95% goodput")

    # ---- 2. Autoscale ----
    b = cold_start_breakdown(10, 1.0, 140, 2.0, 15, 5)
    total = sum(b.values())
    print(f"\ncold start {total:.0f} s: " + ", ".join(f"{k} {v:.0f}s" for k, v in b.items()) + f"; largest stage {largest_stage(b)}")
    fast = cold_start_breakdown(10, 1.0, 140, 10.0, 15, 5)
    print(f"  with weights from a local cache: {sum(fast.values()):.0f} s; headroom for +30 req/s per minute: {scale_up_headroom(30, total, knee)} -> {scale_up_headroom(30, sum(fast.values()), knee)} replicas")
    sustainable = knee * 2.0  # concurrency one replica can hold at the knee, by Little's law with 2 s of latency
    cands = [dict(min_replicas=m, max_replicas=20, target_concurrency=tc, window_s=30, scale_down_delay_s=120)
             for m in (3, 6, 9) for tc in (int(0.6 * sustainable), int(0.9 * sustainable))]
    for kind in ("spike", "diurnal"):
        load = traffic_profile(kind, 3600, 1.0)
        best, results = tune_autoscaler(cands, load, 1.0, cold_start_s=total, replica_capacity=knee, gpu_hourly=2.5, violation_budget=0.03)
        print(f"  {kind} traffic 20 to 60 req/s over one hour, replica capacity {knee} req/s, cold start {total:.0f} s, budget 3% of ticks queueing over 1 s:")
        for line in format_tuning(results, best):
            print("    " + line)

    # ---- 3. Ship ----
    base = {"p99_ms": 800, "error_rate": 0.01, "quality": 0.90}
    th = {"latency_rel": 0.10, "error_abs": 0.005, "quality_abs": 0.02}
    rollout = canary_rollout([5, 25, 50, 100], lambda s: {"p99_ms": 800 + 4 * s, "error_rate": 0.01, "quality": 0.90}, base, th)
    print(f"\ncanary rollout of a version whose p99 grows with traffic share: {rollout}")
    print(f"cost: ${cost_per_million_tokens(2.5, 1200):.3f} per million tokens busy, ${cost_per_million_tokens(2.5, 1200, 0.4):.3f} at 40% utilization; "
          f"break-even vs a $2/M API at $14600/month: {break_even_monthly_tokens(14600, 2.0):.0f} M tokens/month")
    m = RollingSLO(60, {"p99_ms": 1000, "error_rate": 0.02, "min_rps": 0.5})
    rng = np.random.default_rng(0)
    for t in range(180):
        m.ingest(t, 300 + 700 * (t > 120) + rng.integers(0, 50), t < 150 or t % 3 != 0)
    print(f"observability: t=100 {m.snapshot(100)['p99_ms']:.0f} ms p99, alerts {m.alerts(100)}; t=179 {m.snapshot(179)['p99_ms']:.0f} ms p99, alerts {m.alerts(179)}; t=400 alerts {m.alerts(400)}")

    # ---- 4. Client ----
    responses = iter([(503, None), (503, None), (429, None), (200, "ok")])
    r = retry_with_backoff(lambda: next(responses), 6, 0.5, 8.0, np.random.default_rng(0))
    print(f"\nclient: {r['attempts']} attempts, delays {[round(d, 2) for d in r['delays']]} s, final status {r['status']}")
    stream = 'data: {"token": "Inference "}\n\ndata: {"token": "engineering "}\n\n: keep-alive\n\ndata: {"token": "ships."}\n\ndata: [DONE]\n\n'
    chunks = [stream[i:i + 5] for i in range(0, len(stream), 5)]
    print(f"  SSE stream in {len(chunks)} five-byte chunks -> {collect_stream(chunks)!r}")


if __name__ == "__main__":
    main()

