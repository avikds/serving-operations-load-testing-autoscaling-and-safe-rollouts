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

# Step 2 - arrival_times
def arrival_times(rate, duration, pattern, rng):
    """Generate sorted request arrival times in [0, duration)."""
    if rate <= 0:
        raise ValueError("rate must be positive.")
    if duration < 0:
        raise ValueError("duration must be non-negative.")
    if pattern not in {"constant", "poisson", "bursty"}:
        raise ValueError("pattern must be 'constant', 'poisson', or 'bursty'.")

    # Constant arrivals begin at t = 0 and occur every 1 / rate seconds.
    if pattern == "constant":
        return np.arange(0.0, duration, 1.0 / rate)

    arrivals = []
    t = 0.0

    if pattern == "poisson":
        while True:
            t += rng.exponential(1.0 / rate)
            if t >= duration:
                break
            arrivals.append(t)
    else:
        # Bursty traffic repeats a 10-second cycle:
        #   0-2 s:  3 * rate
        #   2-10 s: 0.5 * rate
        # The rate is selected according to the current time before
        # drawing each inter-arrival gap.
        while t < duration:
            phase = t % 10.0
            current_rate = 3.0 * rate if phase < 2.0 else 0.5 * rate

            t += rng.exponential(1.0 / current_rate)

            if t >= duration:
                break

            arrivals.append(t)

    return np.asarray(arrivals, dtype=float)


def request_mix(times, rng, in_mean=400, out_mean=150):
    """Create request metadata with input/output lengths for each arrival."""
    requests = []

    for request_id, t_arrive in enumerate(times):
        # Draw input length first, then output length, as specified.
        input_len = rng.integers(50, 2 * in_mean)
        output_len = rng.integers(20, 2 * out_mean)

        requests.append({
            "id": request_id,
            "t_arrive": t_arrive,
            "input_len": input_len,
            "output_len": output_len,
        })

    return requests

# Step 3 - ReplicaSim
class ReplicaSim:
    def __init__(self, max_batch, prefill_tokens_per_s, decode_ms_base, decode_ms_per_seq):
        self.max_batch = max_batch
        self.prefill_tokens_per_s = prefill_tokens_per_s
        self.decode_ms_base = decode_ms_base
        self.decode_ms_per_seq = decode_ms_per_seq

    def run(self, requests):
        # Process requests in arrival order and keep a reference to each
        # request's output record so the final result can be returned in
        # the original request order.
        waiting = list(requests)
        active = []
        records = [
            {
                "id": req["id"],
                "t_arrive": req["t_arrive"],
                "token_times": [],
                "t_done": None,
            }
            for req in requests
        ]

        # Map request id to its result record.
        record_by_id = {record["id"]: record for record in records}

        t = 0.0
        next_waiting = 0

        while next_waiting < len(waiting) or active:
            # When the replica is idle, advance directly to the next arrival.
            if not active and next_waiting < len(waiting):
                t = max(t, waiting[next_waiting]["t_arrive"])

            # Admit all requests that have already arrived, subject to
            # the maximum batch size. Admission preserves arrival order.
            prefill_time = 0.0

            while (
                next_waiting < len(waiting)
                and len(active) < self.max_batch
                and waiting[next_waiting]["t_arrive"] <= t
            ):
                req = waiting[next_waiting]
                record = record_by_id[req["id"]]

                active.append({
                    "request": req,
                    "record": record,
                })

                prefill_time += req["input_len"] / self.prefill_tokens_per_s
                next_waiting += 1

            # There should only be no active work when there are no requests
            # left to process. The loop otherwise advances to the next arrival.
            if not active:
                continue

            # A decode step includes all prefill work from newly admitted
            # sequences followed by one decode iteration for the active batch.
            decode_time = (
                self.decode_ms_base
                + self.decode_ms_per_seq * len(active)
            ) / 1000.0

            t += prefill_time + decode_time

            # Every active sequence emits exactly one token at the end of
            # this step, including requests admitted at the beginning of it.
            finished = []

            for item in active:
                req = item["request"]
                record = item["record"]

                record["token_times"].append(t)

                if len(record["token_times"]) >= req["output_len"]:
                    record["t_done"] = t
                    finished.append(item)

            # Remove requests that have generated their complete output.
            if finished:
                finished_set = {id(item) for item in finished}
                active = [
                    item for item in active
                    if id(item) not in finished_set
                ]

        return records

# Step 4 - run_benchmark
def run_benchmark(sim, requests, slo):
    """Run a benchmark and summarize throughput, latency, and goodput."""
    if not requests:
        raise ValueError("requests must contain at least one request.")

    records = sim.run(requests)

    # Compute per-request metrics from the simulator output.
    metrics = [
        request_metrics(record["t_arrive"], record["token_times"])
        for record in records
    ]

    first_arrival = min(record["t_arrive"] for record in records)
    last_completion = max(record["t_done"] for record in records)
    makespan = last_completion - first_arrival

    if makespan <= 0:
        raise ValueError("Benchmark makespan must be positive.")

    total_requests = len(records)
    total_output_tokens = sum(len(record["token_times"]) for record in records)

    ttft_values = [m["ttft"] for m in metrics]
    itl_values = [m["itl_mean"] for m in metrics]
    e2e_values = [m["e2e"] for m in metrics]

    # A request is counted as goodput only when both latency objectives
    # are satisfied.
    good_requests = sum(
        1
        for m in metrics
        if m["ttft"] <= slo["ttft"] and m["itl_mean"] <= slo["itl"]
    )

    return {
        "req_per_s": float(total_requests / makespan),
        "tokens_per_s": float(total_output_tokens / makespan),
        "ttft_p50": float(percentile(ttft_values, 0.50)),
        "ttft_p99": float(percentile(ttft_values, 0.99)),
        "itl_p50": float(percentile(itl_values, 0.50)),
        "itl_p99": float(percentile(itl_values, 0.99)),
        "e2e_p99": float(percentile(e2e_values, 0.99)),
        "goodput": float(good_requests / total_requests),
    }


def format_benchmark(s):
    """Format a benchmark summary as the requested single-line string."""
    return (
        f"{s['req_per_s']:6.2f} req/s "
        f"{s['tokens_per_s']:8.1f} tok/s "
        f"TTFT p50 {s['ttft_p50'] * 1000:6.0f} ms "
        f"p99 {s['ttft_p99'] * 1000:6.0f} ms "
        f"ITL p99 {s['itl_p99'] * 1000:5.1f} ms "
        f"goodput {s['goodput']:5.1%}"
    )

# Step 5 - throughput_latency_curve
def throughput_latency_curve(make_sim, rates, duration, slo, seed=0, pattern='poisson'):
    curve = []

    for rate in rates:
        # Create a fresh RNG for each offered rate, as specified.
        rng = np.random.default_rng(seed)

        times = arrival_times(rate, duration, pattern, rng)
        requests = request_mix(times, rng)

        summary = run_benchmark(make_sim(), requests, slo)

        curve.append({
            "rate": rate,
            "tokens_per_s": summary["tokens_per_s"],
            "ttft_p99": summary["ttft_p99"],
            "goodput": summary["goodput"],
        })

    return curve


def capacity_knee(curve, min_goodput=0.95):
    qualifying_rates = [
        point["rate"]
        for point in curve
        if point["goodput"] >= min_goodput
    ]

    return max(qualifying_rates) if qualifying_rates else 0.0


def format_curve(curve):
    # Return one formatted string per curve point with the exact
    # spacing expected by the project tests.
    return [
        f"offered {point['rate']:5.1f} req/s -> "
        f"{point['tokens_per_s']:8.1f} tok/s  "
        f"TTFT p99 {point['ttft_p99'] * 1000:7.0f} ms  "
        f"goodput {point['goodput']:5.1%}"
        for point in curve
    ]

# Step 6 - cold_start_seconds
def cold_start_breakdown(
    image_gb,
    image_bw_gbps,
    weight_gb,
    weight_bw_gbps,
    init_s,
    warmup_s,
    provision_s=0.0,
):
    """Return the cold-start time broken down by startup stage."""
    return {
        "provision": float(provision_s),
        "image_pull": float(image_gb / image_bw_gbps),
        "weight_load": float(weight_gb / weight_bw_gbps),
        "init": float(init_s),
        "warmup": float(warmup_s),
    }


def cold_start_seconds(**kwargs):
    """Return the total cold-start time across all stages."""
    breakdown = cold_start_breakdown(**kwargs)
    return float(sum(breakdown.values()))


def largest_stage(breakdown):
    """Return the name of the longest cold-start stage."""
    return max(breakdown, key=breakdown.get)


def scale_up_headroom(rise_per_min, cold_start_s, replica_capacity):
    """Return spare replicas needed to absorb traffic during cold start."""
    return math.ceil(
        (rise_per_min / 60.0) * cold_start_s / replica_capacity
    )

# Step 7 - Autoscaler
class Autoscaler:
    def __init__(
        self,
        min_replicas,
        max_replicas,
        target_concurrency,
        window_s,
        scale_down_delay_s,
    ):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.target_concurrency = target_concurrency
        self.window_s = window_s
        self.scale_down_delay_s = scale_down_delay_s

        # The fleet starts at the configured minimum.
        self.replicas = min_replicas

        # Store (time, concurrency) observations for the rolling window.
        self._history = []

        # Time at which the current scale-down deficit began.
        self._scale_down_since = None

    def step(self, t, concurrency):
        # Record the current observation first so it is included in
        # the interval (t - window_s, t].
        self._history.append((t, concurrency))

        window_start = t - self.window_s

        # Keep only observations that are still relevant. The lower
        # boundary is exclusive, matching (t - window_s, t].
        self._history = [
            (obs_t, obs_concurrency)
            for obs_t, obs_concurrency in self._history
            if obs_t > window_start and obs_t <= t
        ]

        # The current observation is always present for a normal step.
        mean_concurrency = sum(
            obs_concurrency for _, obs_concurrency in self._history
        ) / len(self._history)

        # Convert mean concurrency to the desired replica count and clamp
        # it to the configured minimum and maximum.
        desired = math.ceil(
            mean_concurrency / self.target_concurrency
        )
        desired = max(self.min_replicas, min(self.max_replicas, desired))

        # Scale up immediately whenever the desired count exceeds the
        # current replica count.
        if desired > self.replicas:
            self.replicas = desired
            self._scale_down_since = None

        # A desired count at or above the current count means there is no
        # active scale-down deficit, so reset the delay timer.
        elif desired >= self.replicas:
            self._scale_down_since = None

        # Desired replicas are below the current count. Start the timer
        # when the deficit is first observed, and scale down only after
        # the deficit has persisted for the full configured delay.
        else:
            if self._scale_down_since is None:
                self._scale_down_since = t

            if t - self._scale_down_since >= self.scale_down_delay_s:
                self.replicas = desired
                self._scale_down_since = None

        return self.replicas

# Step 8 - traffic_profile
def profile_times(duration_s, dt):
    """Return sampling times from 0 up to duration_s, excluding the endpoint."""
    return np.arange(0.0, duration_s, dt)


def traffic_profile(kind, duration_s, dt, base=20.0, peak=60.0):
    """Generate an offered-load profile sampled every dt seconds."""
    times = profile_times(duration_s, dt)

    if kind == "diurnal":
        # One complete cosine cycle across the requested duration.
        return base + (peak - base) * (
            1.0 - np.cos(2.0 * np.pi * times / duration_s)
        ) / 2.0

    if kind == "spike":
        # Baseline traffic with a 300-second spike beginning at one-third
        # of the total simulation duration.
        spike_start = duration_s / 3.0
        spike_end = spike_start + 300.0

        return np.where(
            (times >= spike_start) & (times < spike_end),
            peak,
            base,
        )

    if kind == "ramp":
        # Linear increase from base at t=0 toward peak at duration_s.
        return base + (peak - base) * (times / duration_s)

    raise ValueError("kind must be 'diurnal', 'spike', or 'ramp'.")

# Step 9 - simulate_fleet
def simulate_fleet(
    load,
    dt,
    autoscaler,
    cold_start_s,
    replica_capacity,
    latency_s=2.0,
    max_wait_s=1.0,
):
    """Run a fluid fleet simulation with autoscaling and cold-start delays."""
    load = np.asarray(load, dtype=float)

    n = len(load)

    # Ready replicas start at the autoscaler's current replica count.
    ready = autoscaler.replicas

    # Boot completion times for replicas that are not ready yet.
    pending = []

    # Backlog, measured in requests.
    queue = 0.0

    ready_series = np.zeros(n, dtype=float)
    queue_series = np.zeros(n, dtype=float)
    wait_series = np.zeros(n, dtype=float)

    replica_seconds = 0.0

    for i, offered_load in enumerate(load):
        t = i * dt

        # Any replica whose cold start has completed becomes ready before
        # this tick's traffic is served.
        completed = [boot_time for boot_time in pending if boot_time <= t]
        if completed:
            ready += len(completed)
            pending = [boot_time for boot_time in pending if boot_time > t]

        capacity = ready * replica_capacity

        # Requests arriving during this tick are added to the existing
        # backlog, then served up to the available replica capacity.
        arrivals = offered_load * dt
        incoming_queue = queue + arrivals
        served = min(incoming_queue, capacity * dt)

        queue = incoming_queue - served

        # Approximate observed concurrency using offered traffic multiplied
        # by latency, plus the currently queued requests.
        concurrency = offered_load * latency_s + queue

        desired = autoscaler.step(t, concurrency)

        current_fleet = ready + len(pending)

        # Scale up by creating only the replicas needed beyond the current
        # ready + pending fleet. Every new replica pays the full cold-start.
        if desired > current_fleet:
            additional = desired - current_fleet
            pending.extend([t + cold_start_s] * additional)

        # Scale down ready replicas immediately when requested. Pending
        # replicas are not cancelled because the specification only reduces
        # the ready count.
        elif desired < ready:
            ready = max(0, desired)

        wait = queue / capacity if capacity > 0 else np.inf

        ready_series[i] = ready
        queue_series[i] = queue
        wait_series[i] = wait

        # Pay for all replicas that are either ready or still booting.
        replica_seconds += (ready + len(pending)) * dt

    violation_fraction = (
        float(np.mean(wait_series > max_wait_s))
        if n > 0
        else 0.0
    )

    peak_queue = float(np.max(queue_series)) if n > 0 else 0.0

    return {
        "ready": ready_series,
        "queue": queue_series,
        "wait": wait_series,
        "replica_seconds": float(replica_seconds),
        "violation_fraction": violation_fraction,
        "peak_queue": peak_queue,
    }

