PONS - Python Opportunistic Network Simulator
===

A modular DTN simulator in the style of [the ONE](https://github.com/akeranen/the-one).

Features:
# PONS - Python Opportunistic Network Simulator

**PONS** is a modular, Python-native Delay-Tolerant Network (DTN) simulator designed for **surrogate-assisted multi-objective optimization** of routing strategies. It was developed as part of a Master's thesis to explore and optimize probabilistic forwarding parameters in DTNs with four simultaneous objectives:

- **F1:** Maximize delivery probability
- **F2:** Minimize energy consumption
- **F3:** Minimize delivery latency
- **F4:** Maximize energy fairness

Unlike traditional Java-based simulators such as [The ONE](https://github.com/akeranen/the-one), PONS integrates seamlessly with Python-based machine learning workflows, enabling direct surrogate modeling (Polynomial Regression, Gaussian Processes) and large-scale Bayesian optimization.

---

## Key Features

- **Full Python Integration**
  - Built entirely in Python using SimPy, NetworkX, NumPy, and Pandas.
  - Direct compatibility with Python ML/optimization libraries such as scikit-learn, BoTorch, and GPyTorch.

- **DTN Routing Protocols**
  - Epidemic (probabilistic variant with `p1`, `p2`, `p3`)
  - Spray & Wait
  - First Contact
  - Direct Delivery
  - PRoPHET
  - Static

- **Mobility Models**
  - Random Waypoint (built-in)
  - External movement from ONE simulator
  - External movement from ns2

- **Contact Plan Models**
  - Static networkx topologies (GraphML import)
  - Dynamic contact plans (ION, Core Contact Plan)

- **Custom Metrics & Logging**
  - Delivery probability
  - Energy consumption
  - Average/percentile latency
  - Energy fairness (Jain’s fairness index variant)

- **Automation Tools**
  - `sweep.py` – High-throughput parameter sweeps with parallel execution
  - `find_pareto_front.py` – Identify Pareto-optimal routing strategies
  - `train_gp_models.py` – Train Gaussian Process surrogates
  - `Validate_Polynomials_Models.py` – Validate and select polynomial surrogate models
  - `Visualize_paretoFront.py` – Generate trade-off visualizations (pairplots, parallel coordinate plots)
  - `netedit` – Generate/edit network topologies
  - `ponsanim` – Create animated visualizations from contact/event logs

---

## Repository Structure
└── pons/
    ├── README.md
    ├── 90simtiming_test14W.csv
    ├── conv_12h.csv
    ├── conv_24h.csv
    ├── conv_6h.csv
    ├── EDA.py
    ├── find_pareto_front.py
    ├── LICENSE
    ├── plot_converg.py
    ├── pyproject.toml
    ├── requirements.txt
    ├── sim.py
    ├── sweep.py
    ├── sweep_sparse.csv
    ├── sweepSparse_10x50.csv
    ├── test_results.csv
    ├── timing_test.csv
    ├── timing_test12W.csv
    ├── timing_test8W.csv
    ├── train_gp_models.py
    ├── Validate_Polynomials_Models.py
    ├── Visualize_paretoFront.py
    ├── examples/
    │   ├── corecontactplan-asym.py
    │   ├── corecontactplan.py
    │   ├── ext_ns2.py
    │   ├── ext_one.py
    │   ├── ioncontactplan.py
    │   ├── netedit_and_ccp.py
    │   ├── netplan.py
    │   ├── netplan2.py
    │   ├── ping.py
    │   ├── plan_reader.py
    │   ├── run_tests
    │   ├── static_routing.py
    │   ├── static_routing2.py
    │   ├── static_routing3.py
    │   ├── tests.json
    │   ├── two_net.py
    │   ├── data/
    │   │   ├── 3n-asym.ccm
    │   │   ├── 3n-dyn-link.graphml
    │   │   ├── 3n-exported.ccm
    │   │   ├── 3n-exported.graphml
    │   │   ├── 3n-netedit.ccm
    │   │   ├── 3n-netedit.graphml
    │   │   ├── 3n.ccm
    │   │   ├── contactPlan_complex.txt
    │   │   ├── contactPlan_simple.txt
    │   │   ├── scenario1.ns_movements
    │   │   ├── simple.ccm
    │   │   └── topo.graphml
    │   ├── dtnmail/
    │   │   ├── 3n.ccp
    │   │   ├── bpimapd.py
    │   │   ├── bpsmtpd.py
    │   │   ├── dtnmail.py
    │   │   ├── run.sh
    │   │   ├── test_recv.py
    │   │   └── test_send.py
    │   └── scenario/
    │       └── simple_test/
    │           ├── contacts.ccp
    │           ├── contacts.csv
    │           ├── contacts.json
    │           ├── flows.json
    │           └── nodes.json
    ├── outputs/
    │   ├── gp_cv_scores.csv
    │   └── saved_models/
    │       └── scaler.joblib
    ├── outputs_ValidatePolyModels/
    │   ├── model_coeffs_F1.txt
    │   ├── model_coeffs_F2.txt
    │   ├── model_coeffs_F3.txt
    │   ├── model_coeffs_F4.txt
    │   └── polynomial_cv_scores.csv
    ├── pons/
    │   ├── __init__.py
    │   ├── event_log.py
    │   ├── message.py
    │   ├── node.py
    │   ├── simulation.py
    │   ├── apps/
    │   │   ├── __init__.py
    │   │   ├── app.py
    │   │   ├── ping.py
    │   │   └── udpgw.py
    │   ├── mobility/
    │   │   ├── __init__.py
    │   │   ├── movement.py
    │   │   └── ns2_parser.py
    │   ├── net/
    │   │   ├── __init__.py
    │   │   ├── common.py
    │   │   ├── netplan.py
    │   │   └── plans/
    │   │       ├── __init__.py
    │   │       ├── core.py
    │   │       ├── ion.py
    │   │       └── parser.py
    │   ├── routing/
    │   │   ├── __init__.py
    │   │   ├── directdelivery.py
    │   │   ├── epidemic.py
    │   │   ├── firstcontact.py
    │   │   ├── prophet.py
    │   │   ├── router.py
    │   │   ├── sprayandwait.py
    │   │   └── static.py
    │   └── utils/
    │       ├── __init__.py
    │       ├── list_utils.py
    │       ├── misc.py
    │       └── vector.py
    ├── tests/
    │   ├── __init__.py
    │   └── mobility/
    │       ├── README.md
    │       ├── ns2_example_-500_3600_-482_3508.txt
    │       ├── ns2_example_0_3600_18_3035.txt
    │       └── ns2_tests.py
    └── tools/
        ├── netedit/
        │   ├── README.md
        │   ├── Dockerfile
        │   ├── netedit.py
        │   ├── requirements.txt
        │   ├── run-in-docker.sh
        │   ├── docker/
        │   │   ├── entrypoint.sh
        │   │   └── xstartup
        │   └── gui/
        │       ├── __init__.py
        │       └── dialogs.py
        ├── plot_contacts/
        │   ├── plot_contacts.py
        │   └── requirements.txt
        ├── ponsanim/
        │   ├── ponsanim.py
        │   └── requirements.txt
        └── scenariorunner/
            ├── scenariohelper.py
            └── scenariorunner.py

## Requirements

- simpy >= 4.0
- networkx >= 3.2
- plotting:
  - seaborn
  - pandas
  - matplotlib
  - numpy
- tools:
  - pillow
  - opencv-python
  - tkinter


## Example

```python
import random
import json

import pons
import pons.routing

RANDOM_SEED = 42
SIM_TIME = 3600*24
NET_RANGE = 50
NUM_NODES = 10
WORLD_SIZE = (3000, 3000)

# Setup and start the simulation
random.seed(RANDOM_SEED)

moves = pons.generate_randomwaypoint_movement(
    SIM_TIME, NUM_NODES, WORLD_SIZE[0], WORLD_SIZE[1], max_pause=60.0)

net = pons.NetworkSettings("NET1", range=NET_RANGE)
epidemic = pons.routing.EpidemicRouter()

nodes = pons.generate_nodes(NUM_NODES, net=[net], router=epidemic)
config = {"movement_logger": False, "peers_logger": False, "event_logger": True}

msggenconfig = {"type": "single", "interval": 30, 
  "src": (0, NUM_NODES), "dst": (0, NUM_NODES), 
  "size": 100, "id": "M"}

netsim = pons.NetSim(SIM_TIME, WORLD_SIZE, nodes, moves,
                     config=config, msggens=[msggenconfig])

netsim.setup()

netsim.run()

# print results

print(json.dumps(netsim.net_stats, indent=4))
print(json.dumps(netsim.routing_stats, indent=4))
```

Run using `python3` or for improved performance use `pypy3`.

## Magic ENV Variables

Some of the simulation core functions can be set during runtime without having to change your simulation code.

- `LOG_FILE` can be set to change the default event log file from `/tmp/events.log` to something else
- `SIM_DURATION` can be used to override the calculated simulation duration

For `netedit` there are also ways to influence its behavior:
- `BG_IMG` can be set to any image and it while be rendered as a background behind the network topology

