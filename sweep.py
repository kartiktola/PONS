#!/usr/bin/env python
# sweep.py

import argparse, csv, json, random, statistics, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pons.simulation import run_simulation
import numpy as np
import pons
from pons.routing.epidemic import EpidemicRouter
from pons.node import generate_nodes

def parse_args():
    p = argparse.ArgumentParser(description="Batch sweep of sim.py over (p1,p2,p3)")
    p.add_argument("--sim", default=".venv/Scripts/python sim.py", help="Command to invoke sim.py")
    p.add_argument("--nodes",  type=int, default=100)
    p.add_argument("--duration", type=int, default=86400)
    p.add_argument("--grid",   type=int, default=10)
    p.add_argument("--runs",   type=int, default=20)
    p.add_argument("--timeout", type=int, default=300,
                   help="Per‐run timeout in seconds")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel worker threads")
    p.add_argument("--capacity", type=int, default=0, help="Buffer capacity for router")
    p.add_argument("--energy_thresh", type=float, default=0.0,
                   help="Min energy to allow forwarding")
    p.add_argument("--pop_thresh",    type=float, default=0.0,
                   help="Min popularity to allow forwarding")
    p.add_argument("--world_width",   type=float, default=1000.0,
                   help="Width of simulation area")
    p.add_argument("--world_height",  type=float, default=1000.0,
                   help="Height of simulation area")
    p.add_argument("--max_pause",     type=float, default=60.0,
                   help="Max pause time for RandomWaypoint")
    p.add_argument("--net_range", type=float, default=50.0,
              help="Communication radius (m)")
    p.add_argument("--out",    default="results.csv")
    return p.parse_args()

def run_one(p1, p2, p3, seed, cfg):
    # build a router with p1,p2,p3 and thresholds
    router = EpidemicRouter(
        p1=p1, p2=p2, p3=p3,
        energy_thresh=cfg.energy_thresh,
        pop_thresh=cfg.pop_thresh
    )
    random.seed(seed)
    np.random.seed(seed)
    # generate movement & message‐gen config from your args
    moves = pons.generate_randomwaypoint_movement(
        cfg.duration,
        cfg.nodes,
        int(cfg.world_size[0]),    # ← cast to int
        int(cfg.world_size[1]),    # ← cast to int
        max_pause=cfg.max_pause
    )
    msggens = [cfg.msggenconfig]
    # run the sim
    stats, nodes = run_simulation(
        router=router,
        num_nodes=cfg.nodes,
        sim_time=cfg.duration,
        world_size=cfg.world_size,
        movements=moves,
        msggens=msggens,
        config={
            "movement_logger": False,
            "peers_logger":   False,
            "event_logging":  False,
            "net": [ pons.NetworkSettings("WIFI_50m", range=cfg.net_range) ]
            }

    )
    # compute energy & latency percentiles exactly as before
    used = [n.initial_energy - n.energy for n in nodes]
    energy_used   = sum(used)
    energy_stddev = np.std(used)
    latencies = router.latencies
    median = float(np.median(latencies)) if latencies else 0.0
    p95    = float(np.percentile(latencies,95)) if latencies else 0.0

    return {
        "F1": stats["delivery_prob"],
        "F2": energy_used,
        "F3": stats["latency_avg"],
        "F4": energy_stddev,
        "F3_med": median,
        "F3_95": p95
    }

def main():
    cfg = parse_args()
        # pack world_size tuple
    cfg.world_size = (cfg.world_width, cfg.world_height)

    # message generator config (matches sim.py defaults)
    cfg.msggenconfig = {
        "type":     "single",
        "interval": 30,
        "src":      (0, cfg.nodes),
        "dst":      (0, cfg.nodes),
        "size":     100,
        "id":       "M",
        "ttl":      cfg.duration,
    }

    vals = [i/cfg.grid for i in range(1, cfg.grid+1)]
    fieldnames = ["p3","a","b","p1","p2","F1","F2","F3","F3_med","F3_95","F4"]
    with open(cfg.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        # Prepare tasks
        tasks = []
        for p3 in vals:
            for a in vals:
                for b in vals:
                    if a > b: continue
                    p1 = p3*a; p2 = p3*b
                    for _ in range(cfg.runs):
                        seed = random.randrange(1_000_000)
                        tasks.append((p1,p2,p3,seed))

        # Parallel execution
        with ProcessPoolExecutor(max_workers=cfg.workers) as exe:
            futures = {
                exe.submit(run_one, p1, p2, p3, seed, cfg): (p1,p2,p3)
                for (p1,p2,p3,seed) in tasks
                }
            # Aggregate per‐config
            results = {}
            for fut in as_completed(futures):
                p1,p2,p3 = futures[fut]
                res = fut.result()
                if not res: 
                    continue
                key = (p1,p2,p3)
                results.setdefault(key, []).append(res)

        # Write aggregated means
        for (p1,p2,p3), runs in results.items():
            F1s = [r["F1"] for r in runs]
            F2s = [r["F2"] for r in runs]
            F3s = [r["F3"] for r in runs]
            F3m = [r["F3_med"] for r in runs]
            F3_ = [r["F3_95"] for r in runs]
            F4s = [r["F4"] for r in runs]
            writer.writerow([
                f"{p3:.3f}", f"{p3*p1/p3:.3f}", f"{p3*p2/p3:.3f}",
                f"{p1:.3f}", f"{p2:.3f}",
                f"{statistics.mean(F1s):.4f}",
                f"{statistics.mean(F2s):.1f}",
                f"{statistics.mean(F3s):.1f}",
                f"{statistics.mean(F3m):.1f}",
                f"{statistics.mean(F3_):.1f}",
                f"{statistics.mean(F4s):.1f}",
            ])
            print(f"Done p1={p1:.3f} p2={p2:.3f} p3={p3:.3f}")

if __name__=="__main__":
    main()
