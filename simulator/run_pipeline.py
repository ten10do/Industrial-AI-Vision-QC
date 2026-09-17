"""Realtime pipeline entry point.

Run:
    .venv/Scripts/python.exe -m simulator.run_pipeline --images 50

Prints a benchmark summary at the end (sample count, interval, workers,
latencies, throughput, queue peak).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from .camera_simulator import CameraSimulator
from .config import OrchestratorConfig, SimulatorConfig
from .orchestrator import InspectionOrchestrator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IndustrialVision-QC realtime pipeline")
    p.add_argument("--source", default=None, help="image source directory")
    p.add_argument("--interval-ms", type=int, default=None, help="capture interval in ms")
    p.add_argument("--line", default=None)
    p.add_argument("--station", default=None)
    p.add_argument("--batch", default=None)
    p.add_argument("--backend-url", default=None)
    p.add_argument("--api-token", default=None, help="operator bearer token for inspection creation (env IVQC_API_TOKEN)")
    p.add_argument("--pipeline-token", default=None, help="pipeline bearer token for telemetry (env IVQC_PIPELINE_TOKEN)")
    p.add_argument("--queue-size", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--retry-max", type=int, default=None)
    p.add_argument("--loop", action="store_true", default=None)
    p.add_argument("--images", type=int, default=None, help="stop after N images (benchmark)")
    return p.parse_args()


def build_configs(args: argparse.Namespace) -> tuple[SimulatorConfig, OrchestratorConfig]:
    sim = SimulatorConfig()
    orch = OrchestratorConfig()
    if args.source:
        sim.source_directory = args.source
    if args.interval_ms is not None:
        sim.interval_ms = args.interval_ms
    if args.line:
        sim.production_line = args.line
    if args.station:
        sim.station = args.station
    if args.batch:
        sim.batch_id = args.batch
    if args.backend_url:
        orch.backend_url = args.backend_url
    if args.api_token:
        orch.api_token = args.api_token
    if args.pipeline_token:
        orch.pipeline_token = args.pipeline_token
    if args.queue_size is not None:
        orch.queue_size = args.queue_size
    if args.workers is not None:
        orch.workers = args.workers
    if args.retry_max is not None:
        orch.retry_max = args.retry_max
    if args.loop:
        sim.loop = True
    return sim, orch


async def _main(args: argparse.Namespace) -> int:
    sim_cfg, orch_cfg = build_configs(args)
    orch_cfg.max_images = args.images
    orchestrator = InspectionOrchestrator(orch_cfg)
    # the simulator must share the orchestrator's bounded queue (backpressure)
    simulator = CameraSimulator(sim_cfg, orchestrator.queue)

    started = time.perf_counter()
    auth_state = "on" if (orch_cfg.api_token and orch_cfg.pipeline_token) else "MISSING (run() will fail fast)"
    print(f"pipeline start | source={sim_cfg.source_directory} interval={sim_cfg.interval_ms}ms "
          f"queue={orch_cfg.queue_size} workers={orch_cfg.workers} max_images={args.images or 'unbounded'} "
          f"auth={auth_state}")
    try:
        await orchestrator.run(simulator, max_images=args.images)
    except KeyboardInterrupt:
        await simulator.stop()
    duration = time.perf_counter() - started
    m = orchestrator.metrics
    e2e = m.e2e_latencies
    throughput = (m.completed_total + m.failed_total) / duration if duration else 0.0

    print("\n===== Phase 3 benchmark =====")
    print(f"sample count           : {m.captured_total}")
    print(f"simulator interval     : {sim_cfg.interval_ms} ms")
    print(f"worker count           : {orch_cfg.workers}")
    print(f"queue maxsize          : {orch_cfg.queue_size}")
    print(f"peak queue depth       : {m.queue_peak_depth}")
    print(f"total duration         : {duration:.2f} s")
    print(f"completed              : {m.completed_total}")
    print(f"failed (system)        : {m.failed_total}")
    print(f"pass/review/fail       : {m.pass_total}/{m.review_total}/{m.fail_total}")
    if e2e:
        e2e_sorted = sorted(e2e)
        print(f"e2e avg latency        : {sum(e2e)/len(e2e):.1f} ms")
        print(f"e2e p50 latency        : {e2e_sorted[len(e2e_sorted)//2]:.1f} ms")
        print(f"e2e p95 latency        : {e2e_sorted[max(0, int(len(e2e_sorted)*0.95)-1)]:.1f} ms")
    if m.inference_latencies:
        inf = m.inference_latencies
        print(f"inference avg latency  : {sum(inf)/len(inf):.1f} ms")
    print(f"throughput             : {throughput:.2f} inspections/s")
    print(f"conservation (drained) : captured({m.captured_total}) == completed({m.completed_total}) + failed({m.failed_total}) -> {m.conservation_ok()}")
    print("===========================")
    return 0 if m.failed_total == 0 else 1


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    return asyncio.run(_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
