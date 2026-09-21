# Serving Operations: Load Testing, Autoscaling and Safe Rollouts

Chapters 4.5 and 7 of Inference Engineering as working code. Build the measurement layer first: per-request time to first token, inter-token latency and tokens per second, the percentile rule, and a load generator with constant, Poisson and bursty arrivals, because a benchmark under smooth traffic lies. Simulate a continuous-batching replica and find its knee, the offered load beyond which p99 time to first token leaves the objective. Then run the book's autoscaling story: a cold-start timeline you can attack stage by stage, an autoscaler with the five knobs the book names, a fleet simulation that pays the cold-start penalty on every scale-up, and a tuner that picks the cheapest configuration that still meets the objective. Finish by shipping: a canary rollout with a promote-or-rollback gate, the cost-per-million-tokens and break-even arithmetic, a rolling observability window that raises alerts, and the client side, jittered exponential backoff and a streaming server-sent-events parser that survives chunks split mid-line.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** request_metrics
- [x] **2.** arrival_times
- [x] **3.** ReplicaSim
- [x] **4.** run_benchmark
- [x] **5.** throughput_latency_curve
- [x] **6.** cold_start_seconds
- [x] **7.** Autoscaler
- [x] **8.** traffic_profile
- [x] **9.** simulate_fleet

---

Built on Deep-ML.
