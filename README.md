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
- [x] **10.** tune_autoscaler
- [x] **11.** canary_decision
- [x] **12.** cost_per_million_tokens
- [x] **13.** RollingSLO
- [x] **14.** retry_with_backoff
- [x] **15.** SSEParser

## Results

```
one replica, 60 s of traffic at 6 req/s, objective TTFT <= 500 ms and ITL <= 50 ms
  constant    5.60 req/s    947.0 tok/s TTFT p50     53 ms p99     80 ms ITL p99  26.0 ms goodput 100.0%
  poisson     4.93 req/s    790.0 tok/s TTFT p50     55 ms p99    107 ms ITL p99  26.5 ms goodput 100.0%
  bursty      4.82 req/s    778.1 tok/s TTFT p50     59 ms p99    143 ms ITL p99  31.9 ms goodput 100.0%
throughput-latency curve (poisson arrivals):
  offered   2.0 req/s ->    266.1 tok/s  TTFT p99      95 ms  goodput 100.0%
  offered   4.0 req/s ->    574.7 tok/s  TTFT p99      91 ms  goodput 100.0%
  offered   6.0 req/s ->    790.0 tok/s  TTFT p99     107 ms  goodput 100.0%
  offered   8.0 req/s ->   1054.0 tok/s  TTFT p99     120 ms  goodput 100.0%
  offered  10.0 req/s ->   1346.0 tok/s  TTFT p99     858 ms  goodput 96.4%
  offered  12.0 req/s ->   1500.0 tok/s  TTFT p99    6215 ms  goodput 29.7%
  offered  16.0 req/s ->   1534.8 tok/s  TTFT p99   28542 ms  goodput 10.0%
  offered  24.0 req/s ->   1577.0 tok/s  TTFT p99   71014 ms  goodput  5.1%
  capacity knee: 10 req/s per replica at 95% goodput

cold start 100 s: provision 0s, image_pull 10s, weight_load 70s, init 15s, warmup 5s; largest stage weight_load
  with weights from a local cache: 44 s; headroom for +30 req/s per minute: 5 -> 3 replicas
  spike traffic 20 to 60 req/s over one hour, replica capacity 10 req/s, cold start 100 s, budget 3% of ticks queueing over 1 s:
      min  3 max 20 target  12 window   30s delay  120s -> violations  3.4% cost $   13.51
      min  3 max 20 target  18 window   30s delay  120s -> violations  3.8% cost $   11.02
      min  6 max 20 target  12 window   30s delay  120s -> violations  0.0% cost $   16.13
    * min  6 max 20 target  18 window   30s delay  120s -> violations  0.0% cost $   15.28
      min  9 max 20 target  12 window   30s delay  120s -> violations  0.0% cost $   22.78
      min  9 max 20 target  18 window   30s delay  120s -> violations  0.0% cost $   22.50
  diurnal traffic 20 to 60 req/s over one hour, replica capacity 10 req/s, cold start 100 s, budget 3% of ticks queueing over 1 s:
      min  3 max 20 target  12 window   30s delay  120s -> violations  0.0% cost $   18.29
    * min  3 max 20 target  18 window   30s delay  120s -> violations  0.0% cost $   12.74
      min  6 max 20 target  12 window   30s delay  120s -> violations  0.0% cost $   19.47
      min  6 max 20 target  18 window   30s delay  120s -> violations  0.0% cost $   15.72
      min  9 max 20 target  12 window   30s delay  120s -> violations  0.0% cost $   23.22
      min  9 max 20 target  18 window   30s delay  120s -> violations  0.0% cost $   22.50

canary rollout of a version whose p99 grows with traffic share: {'status': 'rolled_back', 'reached': 25, 'reasons': ['latency']}
cost: $0.579 per million tokens busy, $1.447 at 40% utilization; break-even vs a $2/M API at $14600/month: 7300 M tokens/month
observability: t=100 349 ms p99, alerts []; t=179 1049 ms p99, alerts ['latency', 'errors']; t=400 alerts ['traffic']

client: 4 attempts, delays [0.32, 0.27, 0.08] s, final status 200
  SSE stream in 24 five-byte chunks -> 'Inference engineering ships.'
```
