#!/usr/bin/env python
# sweep.py

import argparse
import csv
import json
import subprocess
import random
import statistics
import sys
import time
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Comprehensive batch sweep of sim.py over (p3,a,b)")
    p.add_argument("--sim",      default="python sim.py", help="Command to invoke sim.py")
    p.add_argument("--router",   default="p_epidemic",    help="Router to pass to --routing")
    p.add_argument("--nodes",    type=int, default=100,   help="Number of nodes")
    p.add_argument("--duration", type=int, default=86400, help="Simulation duration (s)")
    p.add_argument("--energy_thresh", type=float, default=0.0, help="Minimum energy threshold")
    p.add_argument("--pop_thresh",    type=float, default=0.0, help="Minimum popularity threshold")
    p.add_argument("--grid",     type=int, default=5,    help="Grid resolution per axis")
    p.add_argument("--runs",     type=int, default=10,   help="Runs per grid cell")
    p.add_argument("--out",      default="results.csv", help="Output CSV file")
    p.add_argument("--logfile",  default=None,          help="File to log errors")
    p.add_argument("--extras",   nargs=argparse.REMAINDER, help="Extra flags for sim.py")
    return p.parse_args()

def run_one(p1, p2, p3, seed, cfg):
    cmd = cfg.sim.split() + [
        "--routing",    cfg.router,
        "--nodes",      str(cfg.nodes),
        "--duration",   str(cfg.duration),
        "--p1",         f"{p1:.4f}",
        "--p2",         f"{p2:.4f}",
        "--p3",         f"{p3:.4f}",
        "--seed",       str(seed),
        "--energy_thresh", str(cfg.energy_thresh),
        "--pop_thresh",    str(cfg.pop_thresh)
    ] + (cfg.extras or [])
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        msg = f"[ERROR] cmd={' '.join(cmd)}\n{e.stderr}"
        if cfg.logfile:
            with open(cfg.logfile, "a") as logf:
                logf.write(msg + "\n")
        else:
            print(msg, file=sys.stderr)
        return None

    lines = [l.strip() for l in out.splitlines() if l.strip()]
    stats = json.loads(lines[-3])
    energy_used   = float(next(l for l in lines if l.startswith("energy_used:")).split()[-1])
    energy_stddev = float(next(l for l in lines if l.startswith("energy_stddev:")).split()[-1])
    return {
        "F1": stats["delivery_prob"],
        "F2": energy_used,
        "F3": stats["latency_avg"],
        "F4": energy_stddev,
    }

def main():
    cfg = parse_args()
    start_time = time.time()

    # prepare output file
    out_path = Path(cfg.out)
    first_write = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if first_write:
            writer.writerow(["p3","a","b","p1","p2","F1","F2","F3","F4"])

        vals = [i/cfg.grid for i in range(1, cfg.grid+1)]
        total_cells = sum(1 for p3 in vals for a in vals for b in vals if a <= b)
        cell_count = 0

        for p3 in vals:
            for a in vals:
                for b in vals:
                    if a > b:
                        continue
                    cell_count += 1
                    p1, p2 = p3*a, p3*b

                    # run multiple seeds
                    results = {k: [] for k in ("F1","F2","F3","F4")}
                    t0 = time.time()
                    for _ in range(cfg.runs):
                        seed = random.randrange(1_000_000)
                        res = run_one(p1, p2, p3, seed, cfg)
                        if res:
                            for k,v in res.items():
                                results[k].append(v)
                    elapsed = time.time() - t0

                    if not results["F1"]:
                        continue

                    # average metrics
                    row = [
                        f"{p3:.3f}", f"{a:.3f}", f"{b:.3f}",
                        f"{p1:.3f}", f"{p2:.3f}",
                        f"{statistics.mean(results['F1']):.4f}",
                        f"{statistics.mean(results['F2']):.1f}",
                        f"{statistics.mean(results['F3']):.1f}",
                        f"{statistics.mean(results['F4']):.1f}"
                    ]
                    writer.writerow(row)
                    f.flush()

                    # progress
                    done = cell_count/total_cells*100
                    eta = (time.time() - start_time)/cell_count*(total_cells-cell_count)
                    print(f"[{done:5.1f}%] p3={p3:.3f}, a={a:.3f}, b={b:.3f} "
                          f"({cell_count}/{total_cells}) "
                          f"{elapsed:.1f}s/point, ETA {eta/60:.1f}m")

if __name__ == "__main__":
    main()
