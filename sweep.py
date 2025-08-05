#!/usr/bin/env python
# sweep.py

import argparse, csv, json, subprocess, random, statistics, sys

def parse_args():
    p = argparse.ArgumentParser(description="Batch sweep of sim.py over (p3,a,b)")
    p.add_argument("--sim",      default="python sim.py", help="Command to invoke sim.py")
    p.add_argument("--router",   default="p_epidemic",    help="Router name for --routing")
    p.add_argument("--nodes",    type=int, default=100)
    p.add_argument("--duration", type=int, default=86400)
    p.add_argument("--grid",     type=int, default=10,
                   help="Number of steps in [0,1] for a,b,p3")
    p.add_argument("--runs",     type=int, default=20,
                   help="Number of seeds per config")
    p.add_argument("--out",      default="results.csv",
                   help="Output CSV filename")
    return p.parse_args()

def run_one(p1, p2, p3, seed, cfg):
    cmd = cfg.sim.split() + [
        "--routing", cfg.router,
        "--nodes",   str(cfg.nodes),
        "--duration",str(cfg.duration),
        "--p1",      f"{p1:.4f}",
        "--p2",      f"{p2:.4f}",
        "--p3",      f"{p3:.4f}",
        "--seed",    str(seed),
    ]
    try:
        out = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] cmd={' '.join(cmd)}", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return None

    lines = [l.strip() for l in out.splitlines() if l.strip()]
    stats = json.loads(lines[-3])   # built-in JSON
    used  = next(l for l in lines if l.startswith("energy_used:")).split()[-1]
    std   = next(l for l in lines if l.startswith("energy_stddev:")).split()[-1]
    return {
        "F1": stats["delivery_prob"],
        "F2": float(used),
        "F3": stats["latency_avg"],
        "F4": float(std),
    }

def main():
    cfg = parse_args()
    # prepare CSV
    with open(cfg.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p3","a","b","p1","p2","F1","F2","F3","F4"])
        # grid values ∈ (1/grid, ..., grid/grid)
        vals = [i/cfg.grid for i in range(1, cfg.grid+1)]
        for p3 in vals:
            for a in vals:
                for b in vals:
                    if a > b: 
                        continue
                    p1 = p3 * a
                    p2 = p3 * b
                    results = {"F1":[],"F2":[],"F3":[],"F4":[]}
                    for _ in range(cfg.runs):
                        seed = random.randrange(1_000_000)
                        r = run_one(p1,p2,p3,seed,cfg)
                        if r:
                            for k in results:
                                results[k].append(r[k])
                    if len(results["F1"])==0:
                        continue
                    row = [
                        f"{p3:.3f}", f"{a:.3f}", f"{b:.3f}",
                        f"{p1:.3f}", f"{p2:.3f}",
                        f"{statistics.mean(results['F1']):.4f}",
                        f"{statistics.mean(results['F2']):.1f}",
                        f"{statistics.mean(results['F3']):.1f}",
                        f"{statistics.mean(results['F4']):.1f}"
                    ]
                    w.writerow(row)
                    print(f"Done p3={p3:.3f} a={a:.3f} b={b:.3f}")

if __name__=="__main__":
    main()
