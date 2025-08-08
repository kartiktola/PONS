#!/usr/bin/env python
# sweep.py

import argparse, csv, random, statistics, sys, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from pons.simulation import run_simulation
import numpy as np
import pons
from pons.routing.epidemic import EpidemicRouter

def parse_args():
    p = argparse.ArgumentParser(description="Batch sweep of sim.py over (p1,p2,p3)")
    p.add_argument("--nodes",       type=int,   default=100)
    p.add_argument("--duration",    type=int,   default=86400)
    p.add_argument("--grid",        type=int,   default=10)
    p.add_argument("--runs",        type=int,   default=20)
    p.add_argument("--workers",     type=int,   default=None,
                   help="Parallel worker processes (default: auto = min(physical_cores-1, 14))")
    p.add_argument("--energy_thresh", type=float, default=0.0)
    p.add_argument("--pop_thresh",    type=float, default=0.0)
    p.add_argument("--world_width",   type=float, default=1000.0)
    p.add_argument("--world_height",  type=float, default=1000.0)
    p.add_argument("--max_pause",     type=float, default=60.0)
    p.add_argument("--net_range",     type=float, default=50.0,
                   help="Communication radius (m)")
    p.add_argument("--out",           default="results.csv")
    return p.parse_args()

def run_one(params):
    """
    Unpacked task to build router, generate movement, run sim & compute metrics.
    Returns (p1, p2, p3, result_dict).
    """
    (p1, p2, p3, seed,
     nodes, duration, world_size, max_pause,
     net_range, energy_thresh, pop_thresh,
     msggenconfig) = params

    router = EpidemicRouter(
        p1=p1, p2=p2, p3=p3,
        energy_thresh=energy_thresh,
        pop_thresh=pop_thresh
    )
    router.latencies.clear()
    random.seed(seed)
    np.random.seed(seed)

    moves = pons.generate_randomwaypoint_movement(
        duration, nodes,
        int(world_size[0]), int(world_size[1]),
        max_pause=max_pause
    )
    msggens = [msggenconfig]

    stats, node_list = run_simulation(
        router=router,
        num_nodes=nodes,
        sim_time=duration,
        world_size=world_size,
        movements=moves,
        msggens=msggens,
        config={
            "movement_logger": False,
            "peers_logger":   False,
            "event_logging":  False,
            "net": [pons.NetworkSettings("WIFI_50m", range=net_range)]
        }
    )

    used = [n.initial_energy - n.energy for n in node_list]
    energy_used   = sum(used)
    energy_stddev = np.std(used)

    all_lats = []
    for n in node_list:
        all_lats += getattr(n.router, "latencies", [])
    median = float(np.median(all_lats)) if all_lats else 0.0
    p95    = float(np.percentile(all_lats, 95)) if all_lats else 0.0

    return (p1, p2, p3, {
        "F1": stats["delivery_prob"],
        "F2": energy_used,
        "F3": stats["latency_avg"],
        "F4": energy_stddev,
        "F3_med": median,
        "F3_95": p95
    })

def main():
    cfg = parse_args()

    # --- Auto workers: respect user value, otherwise pick min(phys-1, 14)
    if cfg.workers is None:
        try:
            import psutil
            phys = psutil.cpu_count(logical=False)
        except Exception:
            phys = None
        logical = os.cpu_count() or 1
        if not phys:
            phys = max(1, logical // 2)
        cfg.workers = max(1, min(14, phys - 1))
    print(f"[sweep] Using workers={cfg.workers}")

    cfg.world_size = (cfg.world_width, cfg.world_height)
    cfg.msggenconfig = {
        "type":     "single",
        "interval": 30,
        "src":      (0, cfg.nodes),
        "dst":      (0, cfg.nodes),
        "size":     100,
        "id":       "M",
        "ttl":      cfg.duration,
    }

    # Build list of raw task parameters
    vals = [i / cfg.grid for i in range(1, cfg.grid + 1)]
    tasks = []
    for p3 in vals:
        for a in vals:
            for b in vals:
                if a > b:
                    continue
                p1, p2 = p3 * a, p3 * b
                for _ in range(cfg.runs):
                    seed = random.randrange(1_000_000)
                    tasks.append((
                        p1, p2, p3, seed,
                        cfg.nodes, cfg.duration, cfg.world_size, cfg.max_pause,
                        cfg.net_range, cfg.energy_thresh, cfg.pop_thresh,
                        cfg.msggenconfig
                    ))

    total_tasks = len(tasks)
    # number of unique (p1,p2,p3) configs:
    unique_configs = sum(1 for _p3 in vals for _a in vals for _b in vals if _a <= _b)

    header = ["p3","a","b","p1","p2","F1","F2","F3","F3_med","F3_95","F4"]

    # Open CSV with line buffering; write header immediately
    with open(cfg.out, "w", newline="", buffering=1) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        f.flush()

        # Incremental aggregation: write each row as soon as its 'runs' are complete
        aggregates = defaultdict(lambda: {
            "count": 0,
            "sum_F1": 0.0, "sum_F2": 0.0, "sum_F3": 0.0, "sum_F4": 0.0,
            "sum_F3_med": 0.0, "sum_F3_95": 0.0
        })

        completed_tasks = 0
        completed_configs = 0

        print(f"[sweep] Total sims: {total_tasks} | Unique configs: {unique_configs}")

        # Use as_completed so one failed sim doesn't kill the whole run
        with ProcessPoolExecutor(max_workers=cfg.workers) as exe:
            future_to_key = {
                exe.submit(run_one, params): (params[0], params[1], params[2])  # (p1,p2,p3)
                for params in tasks
            }

            for fut in as_completed(future_to_key):
                p1, p2, p3 = future_to_key[fut]
                try:
                    _p1, _p2, _p3, res = fut.result()
                except Exception as e:
                    completed_tasks += 1
                    print(f"[{completed_tasks}/{total_tasks}] ERROR for p1={p1:.3f}, p2={p2:.3f}, p3={p3:.3f}: {e}", file=sys.stderr)
                    continue

                completed_tasks += 1
                key = (p1, p2, p3)
                agg = aggregates[key]
                agg["count"]      += 1
                agg["sum_F1"]     += res["F1"]
                agg["sum_F2"]     += res["F2"]
                agg["sum_F3"]     += res["F3"]
                agg["sum_F4"]     += res["F4"]
                agg["sum_F3_med"] += res["F3_med"]
                agg["sum_F3_95"]  += res["F3_95"]

                print(f"[{completed_tasks}/{total_tasks}] sim done p1={p1:.3f}, p2={p2:.3f}, p3={p3:.3f}  ({agg['count']}/{cfg.runs} for this config)")

                # If this config reached the target count, write the aggregate row now
                if agg["count"] >= cfg.runs:
                    a = p1 / p3 if p3 != 0 else 0.0
                    b = p2 / p3 if p3 != 0 else 0.0
                    writer.writerow([
                        f"{p3:.3f}", f"{a:.3f}", f"{b:.3f}",
                        f"{p1:.3f}", f"{p2:.3f}",
                        f"{(agg['sum_F1']/agg['count']):.4f}",
                        f"{(agg['sum_F2']/agg['count']):.1f}",
                        f"{(agg['sum_F3']/agg['count']):.1f}",
                        f"{(agg['sum_F3_med']/agg['count']):.1f}",
                        f"{(agg['sum_F3_95']/agg['count']):.1f}",
                        f"{(agg['sum_F4']/agg['count']):.1f}",
                    ])
                    f.flush()
                    completed_configs += 1
                    print(f"→ [{completed_configs}/{unique_configs}] wrote aggregate for p1={p1:.3f}, p2={p2:.3f}, p3={p3:.3f}")
                    # free memory for this config
                    del aggregates[key]

        # If anything was partially filled (crash/stop), you could optionally dump partials here.
        if aggregates:
            print(f"[sweep] Note: {len(aggregates)} configs incomplete (less than {cfg.runs} runs each). No rows written for those.")

if __name__ == "__main__":
    main()
