import argparse
import random
import json
import numpy as np

import pons
import pons.routing

# ─── 1. Parse command-line args ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="PONS Simulator with params")
parser.add_argument("--routing",   type=str,   default="epidemic")
parser.add_argument("--nodes",     type=int,   default=10)
parser.add_argument("--duration",  type=int,   default=3600*24)
parser.add_argument("--p1",        type=float, default=0.1, help="Forward prob p1")
parser.add_argument("--p2",        type=float, default=0.2, help="Forward prob p2")
parser.add_argument("--p3",        type=float, default=0.5, help="Forward prob p3")
parser.add_argument("--seed",      type=int,   default=42,  help="RNG seed")
parser.add_argument("--energy_thresh", type=float, default=0.0, help="Min energy to allow forwarding")
parser.add_argument("--pop_thresh",    type=float, default=0.0,help="Min popularity to allow forwarding")
parser.add_argument("--net_range", type=float, default=50.0, help="Communication radius (m)")

args = parser.parse_args()

# ─── 2. Simulation constants ─────────────────────────────────────────────────
RANDOM_SEED = args.seed
SIM_TIME    = args.duration
NUM_NODES   = args.nodes
WORLD_SIZE  = (1000, 1000)
NET_RANGE   = 50
CAPACITY    = 10000

print("Python Opportunistic Network Simulator")
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─── 3. Generate movement ────────────────────────────────────────────────────
moves = pons.generate_randomwaypoint_movement(
    SIM_TIME, NUM_NODES, WORLD_SIZE[0], WORLD_SIZE[1], max_pause=60.0
)

# ─── 4. Configure network & router ───────────────────────────────────────────
net = pons.NetworkSettings("WIFI_50m", range=args.net_range)
epidemic = pons.routing.EpidemicRouter(
     capacity=CAPACITY,
     p1=args.p1,
     p2=args.p2,
     p3=args.p3,
    energy_thresh=args.energy_thresh,
    pop_thresh=args.pop_thresh
 )

nodes = pons.generate_nodes(
    NUM_NODES,
    net=[net],
    router=epidemic
)

config = {
    "movement_logger": False,
    "peers_logger":   False,
    "event_logging":  False
}

msggenconfig = {
    "type":     "single",
    "interval": 30,
    "src":      (0, NUM_NODES),
    "dst":      (0, NUM_NODES),
    "size":     100,
    "id":       "M",
    "ttl":      SIM_TIME,
}

# ─── 5. Setup & run ──────────────────────────────────────────────────────────
netsim = pons.NetSim(
    SIM_TIME,
    nodes,
    world_size=WORLD_SIZE,
    movements=moves,
    config=config,
    msggens=[msggenconfig],
)
netsim.setup()
netsim.run()

# ─── 6. Print built-in stats ─────────────────────────────────────────────────
print(json.dumps(netsim.net_stats,     indent=4))
print(json.dumps(netsim.routing_stats, indent=4))

# ─── 7. Compute & print energy metrics ───────────────────────────────────────
used = [n.initial_energy - n.energy for n in nodes]
energy_used   = float(np.sum(used))
energy_stddev = float(np.std(used))

print(f"energy_used: {energy_used:.3f}")
print(f"energy_stddev: {energy_stddev:.3f}")
# Compute and print latency percentiles
all_latencies = []
for node in nodes:
    # each node.router is a deepcopy of epidemic
    all_latencies.extend(getattr(node.router, "latencies", []))
lat_list = all_latencies

if lat_list:
    median = float(np.median(lat_list))
    p95    = float(np.percentile(lat_list, 95))
else:
    median = p95 = 0.0

print(f"latency_median: {median:.3f}")
print(f"latency_p95:    {p95:.3f}")